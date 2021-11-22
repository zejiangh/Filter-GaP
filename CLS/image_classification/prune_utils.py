import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from math import cos, pi
from copy import deepcopy
import sys
import torch.nn.functional as F
from image_classification.compute_flops import print_model_param_flops
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from . import logger as log
import time
from torch.distributions import MultivariateNormal
from . import utils


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


class SI(nn.Module):
    def __init__(self, inp, k_sobel):
        super(SI, self).__init__()

        self.inp = inp
        
        sobel_2D = get_sobel_kernel(k_sobel)
        sobel_2D_trans = sobel_2D.T
        sobel_2D = torch.from_numpy(sobel_2D).float().cuda()
        sobel_2D_trans = torch.from_numpy(sobel_2D_trans).float().cuda()
        sobel_2D = sobel_2D.unsqueeze(0).repeat(inp,1,1,1)
        sobel_2D_trans = sobel_2D_trans.unsqueeze(0).repeat(inp,1,1,1)
        
        self.vars = nn.ParameterList()
        self.vars.append(nn.Parameter(sobel_2D, requires_grad=False))
        self.vars.append(nn.Parameter(sobel_2D_trans, requires_grad=False))
        
    def forward(self, x):
        grad_x = F.conv2d(x, self.vars[0], bias=None, stride=1, padding=1, dilation=1, groups=self.inp)
        grad_y = F.conv2d(x, self.vars[1], bias=None, stride=1, padding=1, dilation=1, groups=self.inp)
        value = torch.sqrt(grad_x**2 + grad_y**2)
        # value = 1/1.4142 * (torch.abs(grad_x) + torch.abs(grad_y))
        denom = value.shape[2]*value.shape[3]
        out = torch.sum(value**2, dim=(2,3))/denom - (torch.sum(value, dim=(2,3))/denom)**2
        return out ** 0.5


def SI_pruning(model, data_loader):
    model = deepcopy(model.module)

    list_conv=[]
    def conv_hook(self, input, output):
        SIfeature = SI(output.shape[1], 3)
        list_conv.append(SIfeature(output))

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    with torch.no_grad():
        for idx, (data, target) in enumerate (data_loader):
            if idx >= 100:
                break
            data, target = Variable(data.cuda()), Variable(target.cuda())
            model(data)
            if idx == 0:
                score = [torch.mean(m, dim=0, keepdim=True) for m in list_conv]
            else:
                temp = [torch.mean(m, dim=0, keepdim=True) for m in list_conv]
                score = [x+y for x, y in zip(score, temp)]
            list_conv = []
    full_score = [m.squeeze(0).detach().cpu().numpy().tolist() for m in score]
    full_rank = [np.argsort(m) for m in full_score]

    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    layer_id = 1
    score = []
    rank = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if layer_id in l1 + l2 + skip:
                score.append(full_score[layer_id-1])
                rank.append(full_rank[layer_id-1])
                layer_id += 1
                continue
            layer_id += 1
    return score, rank


def L1_norm(layer):
    weight_copy = layer.weight.data.abs().clone().cpu().numpy()
    norm = np.sum(weight_copy, axis=(1,2,3))
    return norm


def Laplacian(layer):
    weight = layer.weight.data.detach()
    x = weight.view(weight.shape[0], -1)
    X_inner = torch.matmul(x, x.t())
    X_norm = torch.diag(X_inner, diagonal=0)
    X_dist_sq = X_norm + torch.reshape(X_norm, [-1,1]) - 2 * X_inner
    X_dist = torch.sqrt(X_dist_sq)
    laplace = torch.sum(X_dist, dim=0).cpu().numpy()
    return laplace


def CSS(layer, k):
    '''
    k: pruning rate, i.e. select (1-k)*C columns
    '''
    weight = layer.weight.data.detach()
    X = weight.view(weight.shape[0], -1)
    X = torch.transpose(X, 0, 1)
    if X.shape[0] >= X.shape[1]:
        _, _, V = torch.svd(X, some=True)
        Vk = V[:,:int((1-k)*X.shape[1])]
        lvs = torch.norm(Vk, dim=1)
        lvs = lvs.cpu().numpy()
        return lvs
    else:
        weight_copy = layer.weight.data.abs().clone().cpu().numpy()
        norm = np.sum(weight_copy, axis=(1,2,3))
        return norm


def get_layer_ratio (model, sparsity):
    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    total = 0
    bn_count = 1
    for m in model.module.modules():
        if isinstance(m, nn.BatchNorm2d):
            if bn_count in l1 + l2 + skip:
                total += m.weight.data.shape[0]
                bn_count += 1
                continue
            bn_count += 1
    bn = torch.zeros(total)
    index = 0
    bn_count = 1
    for m in model.module.modules():
        if isinstance(m, nn.BatchNorm2d):
            if bn_count in l1 + l2 + skip:
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size
                bn_count += 1
                continue
            bn_count += 1
    y, i = torch.sort(bn)
    thre_index = int(total * sparsity)
    thre = y[thre_index]
    layer_ratio = []
    bn_count = 1
    for m in model.module.modules():
        if isinstance(m, nn.BatchNorm2d):
            if bn_count in l1 + l2 + skip:
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
                layer_ratio.append((mask.shape[0] - torch.sum(mask).item()) / mask.shape[0])
                bn_count += 1
                continue
            bn_count += 1
    return layer_ratio
    

def init_channel_mask(model, ratio):
    prev_model = deepcopy(model.module)
    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    layer_id = 1
    cfg_mask = []
    for m in model.module.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if layer_id in l1 + l2 + skip:
                num_keep = int(out_channels * (1 - ratio))
                rank = np.argsort(L1_norm(m))
                arg_max_rev = rank[::-1][:num_keep]
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)
                layer_id += 1
                continue
            layer_id += 1
    return cfg_mask, prev_model


def update_channel_mask(model, layer_ratio_up, layer_ratio_down, old_model, Rank_=None):

    # # regrow EMA weight
    # old_model = deepcopy(model.module)
    # ema.copy_to(old_model.parameters())

    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    layer_id = 1
    idx = 0
    cfg_mask = []
    for [m, m0] in zip(model.module.modules(), old_model.modules()):
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if layer_id in l1:
                num_keep = int(out_channels*(1-layer_ratio_down[idx]))
                num_free = int(out_channels*(1-layer_ratio_up[idx])) - num_keep

                rank = np.argsort(CSS(m, layer_ratio_down[idx]))
                # rank = np.argsort(L1_norm(m))
                # rank = Rank_[idx]

                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                grow = np.random.permutation(freedom)[:num_free]
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask.append(mask)
                
                copy_idx = np.where(L1_norm(m) == 0)[0]
                w = m0.weight.data[copy_idx.tolist(), :, :, :].clone()
                m.weight.data[copy_idx.tolist(),:,:,:] = w.clone()
                
                layer_id += 1
                idx += 1
                continue
            if layer_id in l2:
                num_keep = int(out_channels*(1-layer_ratio_down[idx]))
                num_free = int(out_channels*(1-layer_ratio_up[idx])) - num_keep

                rank = np.argsort(CSS(m, layer_ratio_down[idx]))
                # rank = np.argsort(L1_norm(m))
                # rank = Rank_[idx]

                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                grow = np.random.permutation(freedom)[:num_free]
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask.append(mask)
                
                prev_copy_idx = deepcopy(copy_idx)
                copy_idx = np.where(L1_norm(m) == 0)[0]
                w = m0.weight.data[:,prev_copy_idx.tolist(),:,:].clone()
                m.weight.data[:,prev_copy_idx.tolist(),:,:] = w.clone()
                w = m0.weight.data[copy_idx.tolist(),:,:,:].clone()
                m.weight.data[copy_idx.tolist(),:,:,:] = w.clone()
                
                layer_id += 1
                idx += 1
                continue
            if layer_id in l3:
                
                w = m0.weight.data[:,copy_idx.tolist(),:,:].clone()
                m.weight.data[:,copy_idx.tolist(),:,:] = w.clone()
                
                layer_id += 1
                continue
            if layer_id in skip:
                num_keep = int(out_channels*(1-layer_ratio_down[idx]))
                num_free = int(out_channels*(1-layer_ratio_up[idx])) - num_keep

                rank = np.argsort(CSS(m, layer_ratio_down[idx]))
                # rank = np.argsort(L1_norm(m))
                # rank = Rank_[idx]

                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                grow = np.random.permutation(freedom)[:num_free]
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask.append(mask)
                
                copy_idx = np.where(L1_norm(m) == 0)[0]
                w = m0.weight.data[copy_idx.tolist(), :, :, :].clone()
                m.weight.data[copy_idx.tolist(),:,:,:] = w.clone()
                
                layer_id += 1
                idx += 1
                continue
            layer_id += 1
        elif isinstance(m, nn.BatchNorm2d):
            if layer_id - 1 in l1 + l2 + skip:
                w = m0.weight.data[copy_idx.tolist()].clone()
                m.weight.data[copy_idx.tolist()] = w.clone()
                b = m0.bias.data[copy_idx.tolist()].clone()
                m.bias.data[copy_idx.tolist()] = b.clone()
                rm = m0.running_mean[copy_idx.tolist()].clone()
                m.running_mean[copy_idx.tolist()] = rm.clone()
                rv = m0.running_var[copy_idx.tolist()].clone()
                m.running_var[copy_idx.tolist()] = rv.clone()
                continue
    prev_model = deepcopy(model.module)
    return cfg_mask, prev_model


def apply_channel_mask(model, cfg_mask):
    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    layer_id_in_cfg = 0
    conv_count = 1
    for m in model.module.modules():
        if isinstance(m, nn.Conv2d):
            if conv_count in l1:
                mask = cfg_mask[layer_id_in_cfg].float().cuda()
                mask = mask.view(m.weight.data.shape[0],1,1,1)
                m.weight.data.mul_(mask)
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count in l2:
                mask = cfg_mask[layer_id_in_cfg].float().cuda()
                mask = mask.view(m.weight.data.shape[0],1,1,1)
                m.weight.data.mul_(mask)
                prev_mask = cfg_mask[layer_id_in_cfg-1].float().cuda()
                prev_mask = prev_mask.view(1,m.weight.data.shape[1],1,1)
                m.weight.data.mul_(prev_mask)
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count in l3:
                prev_mask = cfg_mask[layer_id_in_cfg-1].float().cuda()
                prev_mask = prev_mask.view(1,m.weight.data.shape[1],1,1)
                m.weight.data.mul_(prev_mask)
                conv_count += 1
                continue
            if conv_count in skip:
                mask = cfg_mask[layer_id_in_cfg].float().cuda()
                mask = mask.view(m.weight.data.shape[0],1,1,1)
                m.weight.data.mul_(mask)
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            conv_count += 1
        elif isinstance(m, nn.BatchNorm2d):
            if conv_count-1 in l1 + l2 + skip:
                mask = cfg_mask[layer_id_in_cfg-1].float().cuda()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                continue


def detect_channel_zero (model):
    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    total_zero = 0
    total_c = 0
    conv_count = 1
    for m in model.module.modules():
        if isinstance(m, nn.Conv2d):
            if conv_count in l1 + l2 + skip:
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                norm = np.sum(weight_copy, axis=(1,2,3))
                total_zero += len(np.where(norm == 0)[0])
                total_c += m.weight.data.shape[0]
                conv_count += 1
                continue
            conv_count += 1
    return total_zero / total_c


def projection_formula(M):
    scatter = torch.matmul(M.t(), M)
    inv = torch.pinverse(scatter)
    return torch.matmul(torch.matmul(M, inv), M.t())


def IS_update_channel_mask(model, layer_ratio_up, layer_ratio_down, old_model):
    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    layer_id = 1
    idx = 0
    cfg_mask = []
    for [m, m0] in zip(model.module.modules(), old_model.modules()):
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if layer_id in l1:
                # number of channels
                num_keep = int(out_channels*(1-layer_ratio_down[idx]))
                num_free = int(out_channels*(1-layer_ratio_up[idx])) - num_keep
                # pruning criterion
                rank = np.argsort(CSS(m, layer_ratio_down[idx]))
                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                # restore MRU
                copy_idx = np.where(L1_norm(m) == 0)[0]
                w = m0.weight.data[copy_idx.tolist(), :, :, :].clone()
                m.weight.data[copy_idx.tolist(),:,:,:] = w.clone()
                # importance sampling
                weight_copy = m.weight.data.detach().cpu()
                weight_copy = weight_copy.view(weight_copy.shape[0], -1)
                weight_copy = torch.transpose(weight_copy, 0, 1)
                base_weight = weight_copy[:, selected.tolist()]
                proj = projection_formula(base_weight)
                candidate = weight_copy[:, freedom.tolist()]
                candidate_prime = torch.matmul(proj, candidate)
                sampling_prob = F.softmax(torch.norm(candidate - candidate_prime, dim=0))
                if num_free <= 0:
                    grow = np.random.permutation(freedom)[:num_free]
                else: 
                    grow = freedom[np.unique(torch.multinomial(sampling_prob, num_free).numpy())]
                # channel mask
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask.append(mask)
                layer_id += 1
                idx += 1
                continue
            if layer_id in l2:
                # number of channels
                num_keep = int(out_channels*(1-layer_ratio_down[idx]))
                num_free = int(out_channels*(1-layer_ratio_up[idx])) - num_keep
                # pruning criterion
                rank = np.argsort(CSS(m, layer_ratio_down[idx]))
                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                # restore MRU
                prev_copy_idx = deepcopy(copy_idx)
                copy_idx = np.where(L1_norm(m) == 0)[0]
                w = m0.weight.data[:,prev_copy_idx.tolist(),:,:].clone()
                m.weight.data[:,prev_copy_idx.tolist(),:,:] = w.clone()
                w = m0.weight.data[copy_idx.tolist(),:,:,:].clone()
                m.weight.data[copy_idx.tolist(),:,:,:] = w.clone()
                # importance sampling
                weight_copy = m.weight.data.detach().cpu()
                weight_copy = weight_copy.view(weight_copy.shape[0], -1)
                weight_copy = torch.transpose(weight_copy, 0, 1)
                base_weight = weight_copy[:, selected.tolist()]
                proj = projection_formula(base_weight)
                candidate = weight_copy[:, freedom.tolist()]
                candidate_prime = torch.matmul(proj, candidate)
                sampling_prob = F.softmax(torch.norm(candidate - candidate_prime, dim=0))
                if num_free <= 0:
                    grow = np.random.permutation(freedom)[:num_free]
                else: 
                    grow = freedom[np.unique(torch.multinomial(sampling_prob, num_free).numpy())]
                # channel mask
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask.append(mask)
                layer_id += 1
                idx += 1
                continue
            if layer_id in l3:
                w = m0.weight.data[:,copy_idx.tolist(),:,:].clone()
                m.weight.data[:,copy_idx.tolist(),:,:] = w.clone()
                layer_id += 1
                continue
            if layer_id in skip:
                # number of channels
                num_keep = int(out_channels*(1-layer_ratio_down[idx]))
                num_free = int(out_channels*(1-layer_ratio_up[idx])) - num_keep
                # pruning criterion
                rank = np.argsort(CSS(m, layer_ratio_down[idx]))
                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                # restore MRU
                copy_idx = np.where(L1_norm(m) == 0)[0]
                w = m0.weight.data[copy_idx.tolist(), :, :, :].clone()
                m.weight.data[copy_idx.tolist(),:,:,:] = w.clone()
                # importance sampling
                weight_copy = m.weight.data.detach().cpu()
                weight_copy = weight_copy.view(weight_copy.shape[0], -1)
                weight_copy = torch.transpose(weight_copy, 0, 1)
                base_weight = weight_copy[:, selected.tolist()]
                proj = projection_formula(base_weight)
                candidate = weight_copy[:, freedom.tolist()]
                candidate_prime = torch.matmul(proj, candidate)
                sampling_prob = F.softmax(torch.norm(candidate - candidate_prime, dim=0))
                if num_free <= 0:
                    grow = np.random.permutation(freedom)[:num_free]
                else: 
                    grow = freedom[np.unique(torch.multinomial(sampling_prob, num_free).numpy())]
                # channel mask
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask.append(mask)
                layer_id += 1
                idx += 1
                continue
            layer_id += 1
        elif isinstance(m, nn.BatchNorm2d):
            if layer_id-1 in l1 +l2 + skip:
                w = m0.weight.data[copy_idx.tolist()].clone()
                m.weight.data[copy_idx.tolist()] = w.clone()
                b = m0.bias.data[copy_idx.tolist()].clone()
                m.bias.data[copy_idx.tolist()] = b.clone()
                rm = m0.running_mean[copy_idx.tolist()].clone()
                m.running_mean[copy_idx.tolist()] = rm.clone()
                rv = m0.running_var[copy_idx.tolist()].clone()
                m.running_var[copy_idx.tolist()] = rv.clone()
                continue
    prev_model = deepcopy(model.module)
    return cfg_mask, prev_model





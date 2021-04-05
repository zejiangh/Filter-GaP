# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import logger as log
from . import resnet as models
from . import utils
import dllogger
from math import cos, pi
from copy import deepcopy
import sys
import torch.nn.functional as F

# from prune_utils import *
from torch_ema import ExponentialMovingAverage

import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )

ACC_METADATA = {"unit": "%", "format": ":.2f"}
IPS_METADATA = {"unit": "img/s", "format": ":.2f"}
TIME_METADATA = {"unit": "s", "format": ":.5f"}
LOSS_METADATA = {"format": ":.5f"}


class ModelAndLoss(nn.Module):
    def __init__(
        self,
        arch,
        loss,
        pretrained_weights=None,
        cuda=True,
        fp16=False,
        memory_format=torch.contiguous_format,
    ):
        super(ModelAndLoss, self).__init__()
        self.arch = arch

        ###########################################################################################
        print("=> creating model '{}'".format(arch))
        model = models.build_resnet(arch[0], arch[1], arch[2], arch[3])
        print(model)
        print('PARAMS:', sum([param.nelement() for param in model.parameters()]))
        
        def process_state_dict(state_dict):
            for k in list(state_dict.keys()):
                state_dict[k.replace('module.', '')] = state_dict.pop(k)
            return state_dict
        
        if pretrained_weights is not None:
            print("=> using pre-trained model from a file '{}'".format(arch))
            model.load_state_dict(process_state_dict(pretrained_weights['state_dict']))
        ###########################################################################################

        if cuda:
            model = model.cuda().to(memory_format=memory_format)
        if fp16:
            model = network_to_half(model)

        # define loss function (criterion) and optimizer
        criterion = loss()

        if cuda:
            criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)

        return loss, output

    def distributed(self):
        self.model = DDP(self.model)

    def load_model_state(self, state):
        if not state is None:
            self.model.load_state_dict(state)


def get_optimizer(
    parameters,
    fp16,
    lr,
    momentum,
    weight_decay,
    nesterov=False,
    state=None,
    static_loss_scale=1.0,
    dynamic_loss_scale=False,
    bn_weight_decay=False,
):

    if bn_weight_decay:
        print(" ! Weight decay applied to BN parameters ")
        optimizer = torch.optim.SGD(
            [v for n, v in parameters],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    else:
        print(" ! Weight decay NOT applied to BN parameters ")
        bn_params = [v for n, v in parameters if "bn" in n]
        rest_params = [v for n, v in parameters if not "bn" in n]
        print(len(bn_params))
        print(len(rest_params))
        optimizer = torch.optim.SGD(
            [
                {"params": bn_params, "weight_decay": 0},
                {"params": rest_params, "weight_decay": weight_decay},
            ],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    if fp16:
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=static_loss_scale,
            dynamic_loss_scale=dynamic_loss_scale,
            verbose=False,
        )

    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer


def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric(
            "lr", log.LR_METER(), verbosity=dllogger.Verbosity.VERBOSE
        )

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric("lr", lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1 - (e / es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(
    base_lr, warmup_length, epochs, final_multiplier=0.001, logger=None
):
    es = epochs - warmup_length
    epoch_decay = np.power(2, np.log2(final_multiplier) / es)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay ** e)
        return lr

    return lr_policy(_lr_fn, logger=logger)


def get_train_step(model_and_loss, optimizer, fp16, use_amp=False, batch_size_multiplier=1):
    
    def _step(input, target, optimizer_step=True):
        
        input_var = Variable(input)
        target_var = Variable(target)
        loss, output = model_and_loss(input_var, target_var)
        if torch.distributed.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        if fp16:
            optimizer.backward(loss)
        elif use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if optimizer_step:
            opt = (
                optimizer.optimizer
                if isinstance(optimizer, FP16_Optimizer)
                else optimizer
            )
            for param_group in opt.param_groups:
                for param in param_group["params"]:
                    param.grad /= batch_size_multiplier
                    
            # # for choosing where to grow by gradient magnitude
            # prune_update_grad(opt)

            optimizer.step()
            optimizer.zero_grad()
            
            # ############################################
            # '''
            # maintain weight sparsity during training
            # '''
            # prune_apply_masks()
            # ############################################

        torch.cuda.synchronize()

        return reduced_loss

    return _step


def train(
    train_loader,
    model_and_loss,
    ema,
    optimizer,
    lr_scheduler,
    fp16,
    logger,
    epoch,
    use_amp=False,
    prof=-1,
    batch_size_multiplier=1,
    register_metrics=True,
    cfg_mask=None,
):

    if register_metrics and logger is not None:
        logger.register_metric(
            "train.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
        )
        logger.register_metric(
            "train.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "train.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "train.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "train.compute_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )

    step = get_train_step(
        model_and_loss,
        optimizer,
        fp16,
        use_amp=use_amp,
        batch_size_multiplier=batch_size_multiplier,
    )

    model_and_loss.train()
    end = time.time()

    optimizer.zero_grad()

    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter)
    if prof > 0:
        data_iter = utils.first_n(prof, data_iter)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        lr_scheduler(optimizer, i, epoch)
        data_time = time.time() - end

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss = step(input, target, optimizer_step=optimizer_step)
        
        ########################################################
        '''
        maintain channel sparsity
        '''
        if not cfg_mask is None:
            apply_channel_mask(model_and_loss.model, cfg_mask)
        ########################################################

        # EMA weight update
        if not ema is None:
            ema.update(model_and_loss.model.module.parameters())

        it_time = time.time() - end

        if logger is not None:
            logger.log_metric("train.loss", to_python_float(loss), bs)
            logger.log_metric("train.compute_ips", calc_ips(bs, it_time - data_time))
            logger.log_metric("train.total_ips", calc_ips(bs, it_time))
            logger.log_metric("train.data_time", data_time)
            logger.log_metric("train.compute_time", it_time - data_time)

        end = time.time()


def get_val_step(model_and_loss):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            loss, output = model_and_loss(input_var, target_var)

        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data)
            prec1 = utils.reduce_tensor(prec1)
            prec5 = utils.reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(
    val_loader, model_and_loss, fp16, logger, epoch, prof=-1, register_metrics=True
):
    if register_metrics and logger is not None:
        logger.register_metric(
            "val.top1",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
        )
        logger.register_metric(
            "val.top5",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
        )
        logger.register_metric(
            "val.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
        )
        logger.register_metric(
            "val.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "val.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "val.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at100",
            log.LAT_100(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at99",
            log.LAT_99(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at95",
            log.LAT_95(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )

    step = get_val_step(model_and_loss)

    top1 = log.AverageMeter()
    # switch to evaluate mode
    model_and_loss.eval()

    end = time.time()

    data_iter = enumerate(val_loader)
    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, val=True)
    if prof > 0:
        data_iter = utils.first_n(prof, data_iter)

    count = 0
    for i, (input, target) in data_iter:
        count += 1
        bs = input.size(0)
        data_time = time.time() - end

        loss, prec1, prec5 = step(input, target)

        it_time = time.time() - end

        top1.record(to_python_float(prec1), bs)
        if logger is not None:
            logger.log_metric("val.top1", to_python_float(prec1), bs)
            logger.log_metric("val.top5", to_python_float(prec5), bs)
            logger.log_metric("val.loss", to_python_float(loss), bs)
            logger.log_metric("val.compute_ips", calc_ips(bs, it_time - data_time))
            logger.log_metric("val.total_ips", calc_ips(bs, it_time))
            logger.log_metric("val.data_time", data_time)
            logger.log_metric("val.compute_latency", it_time - data_time)
            logger.log_metric("val.compute_latency_at95", it_time - data_time)
            logger.log_metric("val.compute_latency_at99", it_time - data_time)
            logger.log_metric("val.compute_latency_at100", it_time - data_time)

        end = time.time()
    print('*'*50, count)

    return top1.get_val()


# Train loop {{{
def calc_ips(batch_size, time):
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    tbs = world_size * batch_size
    return tbs / time
            
# ======================================================================================================================
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

def regrow_allocation(model, delta_sparsity, layer_ratio_down):
    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    bn_count = 1
    idx = 0
    layer_ratio = []
    for m in model.module.modules():
        if isinstance(m, nn.BatchNorm2d):
            out_channel = m.weight.data.shape[0]
            if bn_count in l1 + l2 + skip:
                num_remain = out_channel*(1-layer_ratio_down[idx])
                num_regrow = int(delta_sparsity * out_channel)
                num_prune = out_channel - num_remain - num_regrow
                if num_prune <= 0:
                    num_prune = 0
                layer_ratio.append(num_prune / out_channel)
                idx += 1
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

def update_channel_mask(model, layer_ratio_up, layer_ratio_down, ema, args=None):
    if args.EMA:
        # regrow EMA weight
        old_model = deepcopy(model.module)
        ema.copy_to(old_model.parameters())
    else:
        old_model = ema
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
    
# def detect_weight_zero (model):
#     total_param = 0
#     total_zero = 0
#     for k, m in enumerate(model.module.modules()):
#         if isinstance(m, nn.Conv2d):
#             total_zero += torch.sum(m.weight.data.eq(0))
#             total_param += m.weight.data.numel()
#         if isinstance(m, nn.Linear):
#             total_zero += torch.sum(m.weight.data.eq(0))
#             total_param += m.weight.data.numel()
#     return total_zero.item() / total_param

# def GMP_mask(model, ratio):
#     l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
#     l2 = (np.asarray(l1)+1).tolist()
#     l3 = (np.asarray(l2)+1).tolist()
#     skip = [5,15,28,47]
#     layer_id = 1
#     cfg_mask = []
#     for m in model.module.modules():
#         if isinstance(m, nn.Conv2d):
#             out_channels = m.weight.data.shape[0]
#             if layer_id in l1 + l2 + skip:
#                 num_keep = int(out_channels * (1 - ratio))
#                 rank = np.argsort(L1_norm(m))
#                 arg_max_rev = rank[::-1][:num_keep]
#                 mask = torch.zeros(out_channels)
#                 mask[arg_max_rev.tolist()] = 1
#                 cfg_mask.append(mask)
#                 layer_id += 1
#                 continue
#             layer_id += 1
#     return cfg_mask

# def Pruning(model, ratio, train_loader):
#     print('CSS pruning...')

#     # _,Rank_ = SI_pruning(model, train_loader)

#     l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
#     l2 = (np.asarray(l1)+1).tolist()
#     l3 = (np.asarray(l2)+1).tolist()
#     skip = [5,15,28,47]
#     layer_id = 1
#     cfg_mask = []
#     idx = 0
#     for m in model.module.modules():
#         if isinstance(m, nn.Conv2d):
#             out_channels = m.weight.data.shape[0]
#             if layer_id in l1 + l2 + skip:
#                 num_keep = int(out_channels * (1 - ratio))

#                 # rank = np.argsort(L1_norm(m))
#                 rank = np.argsort(CSS(m, ratio))
#                 # rank = Rank_[idx]

#                 arg_max_rev = rank[::-1][:num_keep]
#                 mask = torch.zeros(out_channels)
#                 mask[arg_max_rev.tolist()] = 1
#                 cfg_mask.append(mask)
#                 layer_id += 1
#                 idx += 1
#                 continue
#             layer_id += 1
#     return cfg_mask

def sampling_update_channel_mask(model, layer_ratio_up, layer_ratio_down, old_model): # orthogonality based sampling (filter diversity)
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
                # pruning rank
                rank = np.argsort(CSS(m, layer_ratio_down[idx]))
                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                # MRU weight
                copy_idx = np.where(L1_norm(m) == 0)[0]
                w = m0.weight.data[copy_idx.tolist(), :, :, :].clone()
                m.weight.data[copy_idx.tolist(),:,:,:] = w.clone()
                # sampling prob
                W_ = m.weight.data.detach()
                W_ = W_.view(W_.shape[0], -1)
                W_ = torch.transpose(W_, 0, 1)
                W_ = W_.cpu().numpy()
                sampling_prob = []
                for H in freedom:
                    W_temp = W_[:,selected.tolist()+[H]]
                    sampling_prob.append(np.exp(-np.linalg.norm(np.matmul(W_temp.T, W_temp) - np.eye(W_temp.shape[1]))))
                # for H in freedom:
                #     W_temp = W_[:,selected.tolist()]
                #     y_ = W_[:,H]
                #     sampling_prob.append(np.linalg.norm(np.matmul(W_temp, np.matmul(np.linalg.inv(np.matmul(W_temp.T, W_temp) + 1e-5*np.eye(W_temp.shape[1])), np.matmul(W_temp.T, y_))) - y_))
                # sampling
                if num_free <= 0:
                    grow = np.random.permutation(freedom)[:num_free]
                else:
                    prob = np.array(sampling_prob) # sampling prob (unnormalized)
                    multinomial_output = torch.multinomial(torch.from_numpy(prob), num_free).numpy()
                    grow = freedom[np.unique(multinomial_output)]
                # mask
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask.append(mask)
                # misc
                layer_id += 1
                idx += 1
                continue
            if layer_id in l2:
                num_keep = int(out_channels*(1-layer_ratio_down[idx]))
                num_free = int(out_channels*(1-layer_ratio_up[idx])) - num_keep
                # pruning rank
                rank = np.argsort(CSS(m, layer_ratio_down[idx]))
                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                # MRU weight
                prev_copy_idx = deepcopy(copy_idx)
                copy_idx = np.where(L1_norm(m) == 0)[0]
                w = m0.weight.data[:,prev_copy_idx.tolist(),:,:].clone()
                m.weight.data[:,prev_copy_idx.tolist(),:,:] = w.clone()
                w = m0.weight.data[copy_idx.tolist(),:,:,:].clone()
                m.weight.data[copy_idx.tolist(),:,:,:] = w.clone()
                # sampling prob
                W_ = m.weight.data.detach()
                W_ = W_.view(W_.shape[0], -1)
                W_ = torch.transpose(W_, 0, 1)
                W_ = W_.cpu().numpy()
                sampling_prob = []
                for H in freedom:
                    W_temp = W_[:,selected.tolist()+[H]]
                    sampling_prob.append(np.exp(-np.linalg.norm(np.matmul(W_temp.T, W_temp) - np.eye(W_temp.shape[1]))))
                # for H in freedom:
                #     W_temp = W_[:,selected.tolist()]
                #     y_ = W_[:,H]
                #     sampling_prob.append(np.linalg.norm(np.matmul(W_temp, np.matmul(np.linalg.inv(np.matmul(W_temp.T, W_temp) + 1e-5*np.eye(W_temp.shape[1])), np.matmul(W_temp.T, y_))) - y_))
                # sampling
                if num_free <= 0:
                    grow = np.random.permutation(freedom)[:num_free]
                else:
                    prob = np.array(sampling_prob) # sampling prob (unnormalized)
                    multinomial_output = torch.multinomial(torch.from_numpy(prob), num_free).numpy()
                    grow = freedom[np.unique(multinomial_output)]
                # mask
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask.append(mask)
                # misc
                layer_id += 1
                idx += 1
                continue
            if layer_id in l3:
                # MRU weight
                w = m0.weight.data[:,copy_idx.tolist(),:,:].clone()
                m.weight.data[:,copy_idx.tolist(),:,:] = w.clone()
                # misc
                layer_id += 1
                continue
            if layer_id in skip:
                num_keep = int(out_channels*(1-layer_ratio_down[idx]))
                num_free = int(out_channels*(1-layer_ratio_up[idx])) - num_keep
                # pruning rank
                rank = np.argsort(CSS(m, layer_ratio_down[idx]))
                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                # MRU weight
                copy_idx = np.where(L1_norm(m) == 0)[0]
                w = m0.weight.data[copy_idx.tolist(), :, :, :].clone()
                m.weight.data[copy_idx.tolist(),:,:,:] = w.clone()
                # sampling prob
                W_ = m.weight.data.detach()
                W_ = W_.view(W_.shape[0], -1)
                W_ = torch.transpose(W_, 0, 1)
                W_ = W_.cpu().numpy()
                sampling_prob = []
                for H in freedom:
                    W_temp = W_[:,selected.tolist()+[H]]
                    sampling_prob.append(np.exp(-np.linalg.norm(np.matmul(W_temp.T, W_temp) - np.eye(W_temp.shape[1]))))
                # for H in freedom:
                #     W_temp = W_[:,selected.tolist()]
                #     y_ = W_[:,H]
                #     sampling_prob.append(np.linalg.norm(np.matmul(W_temp, np.matmul(np.linalg.inv(np.matmul(W_temp.T, W_temp) + 1e-5*np.eye(W_temp.shape[1])), np.matmul(W_temp.T, y_))) - y_))
                # sampling
                if num_free <= 0:
                    grow = np.random.permutation(freedom)[:num_free]
                else:
                    prob = np.array(sampling_prob) # sampling prob (unnormalized)
                    multinomial_output = torch.multinomial(torch.from_numpy(prob), num_free).numpy()
                    grow = freedom[np.unique(multinomial_output)]
                # mask
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask.append(mask)
                # misc
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


def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(n_params, acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):

    best_x = None
    best_acquisition_value = 1

    for starting_point in np.random.uniform(bounds[:,0], bounds[:,1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=(((bounds[:,0], bounds[:,1]), ) * n_params),
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(
    model_and_loss, val_loader, fp16, prof, sparsity, n_iters, x0=None, gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7
    ):

    x_list = []
    y_list = []

    n_params = x0.shape[0]
    lb = np.maximum(x0 - 0.2, np.zeros(n_params))
    ub = np.minimum(x0 + 0.2, np.ones(n_params))
    # bounds = np.array([0,1]) # pruning ratio range
    bounds = np.concatenate((np.expand_dims(lb, axis=1),np.expand_dims(ub, axis=1)), axis=1)

    x_list.append(x0)
    y_list.append(GP_evaluator(model_and_loss.model, x0, sparsity, val_loader, fp16, prof))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=False)

    for n in range(n_iters):

        model.fit(xp, yp)

        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(bounds[:,0], bounds[:,1], size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            pass
            # next_sample = sample_next_hyperparameter(n_params, expected_improvement, model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:,0], bounds[:,1], n_params)

        # Sample loss for new set of parameters
        cv_score = GP_evaluator(model_and_loss.model, next_sample, sparsity, val_loader, fp16, prof)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

    return xp, yp

def GP_get_channel_mask(model, ratio):
    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    layer_id = 1
    cfg_mask = []
    idx = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if layer_id in l1 + l2 + skip:
                num_keep = int(out_channels * (1 - ratio[idx]))
                rank = np.argsort(L1_norm(m))
                arg_max_rev = rank[::-1][:num_keep]
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)
                layer_id += 1
                idx += 1
                continue
            layer_id += 1
    return cfg_mask


def GP_apply_channel_mask(model, cfg_mask):
    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    layer_id_in_cfg = 0
    conv_count = 1
    for m in model.modules():
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

def GP_detect_channel_zero (model):
    l1 = [2,6,9, 12,16,19,22, 25,29,32,35,38,41, 44,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [5,15,28,47]
    total_zero = 0
    total_c = 0
    conv_count = 1
    for m in model.modules():
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

def GP_get_val_step(model_and_loss):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)
        with torch.no_grad():
            output = model_and_loss(input_var)
        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))
        if torch.distributed.is_initialized():
            prec1 = utils.reduce_tensor(prec1)
            prec5 = utils.reduce_tensor(prec5)
        else:
            pass
        torch.cuda.synchronize()
        return prec1, prec5
    return _step

def GP_validate(
    val_loader, model_and_loss, fp16, logger, epoch, prof=-1, register_metrics=True
):
    step = GP_get_val_step(model_and_loss)
    top5 = log.AverageMeter()
    # switch to evaluate mode
    model_and_loss.eval()
    end = time.time()
    data_iter = enumerate(val_loader)
    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end
        prec1, prec5 = step(input, target)
        it_time = time.time() - end
        top5.record(to_python_float(prec5), bs)
        end = time.time()
    return top5.get_val()

def GP_evaluator(model, layer_ratio, sparsity, val_loader, fp16, prof):
    aux_model = deepcopy(model.module)
    cfg_mask = GP_get_channel_mask(aux_model, layer_ratio)
    GP_apply_channel_mask(aux_model, cfg_mask)
    # print('apply mask | detect channel zero: {}'.format(GP_detect_channel_zero(aux_model)))
    accuracy, _ = GP_validate(val_loader, aux_model, fp16, None, 0, prof=prof, register_metrics=False,)
    penalty = np.log(np.abs(GP_detect_channel_zero(aux_model)-sparsity)+1)
    return accuracy - penalty

# =======================================================================================================================

def train_loop(
    model_and_loss,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    fp16,
    logger,
    should_backup_checkpoint,
    use_amp=False,
    batch_size_multiplier=1,
    best_prec1=0,
    start_epoch=0,
    end_epoch=0,
    prof=-1,
    skip_training=False,
    skip_validation=False,
    save_checkpoints=True,
    checkpoint_dir="./",
    checkpoint_filename="checkpoint.pth.tar",
    args=None,
    # weight_sparsity=0.8,
    # init_weight_ratio=0.1,
    T_max=250,
    channel_sparsity=0.2,
    init_channel_ratio=0.2,
    delta_T=2,
    # early_bird=1.0,
):

    prec1 = -1
    print(f"RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}")

    # EMA weight initialization 
    if args.EMA:
        ema = ExponentialMovingAverage(model_and_loss.model.module.parameters(), decay=0.999)
    else:
        ema = None

    #########################################################################
    '''
    initialize mask
    '''
    cfg_mask = None
    if args.grow_prune:
        # filter-wise
        if args.EMA:
            cfg_mask, prev_model = init_channel_mask(model_and_loss.model, channel_sparsity-init_channel_ratio)
        else:
            cfg_mask, _ = init_channel_mask(model_and_loss.model, channel_sparsity-init_channel_ratio) # regrow EMA weight
        apply_channel_mask(model_and_loss.model, cfg_mask)
        # #weight-wise
        # prune_init(args, model_and_loss.model)
        # prune_apply_masks()
        print('apply init. mask | detect channel zero: {}'.format(detect_channel_zero(model_and_loss.model)))

    if args.GMP:
        cfg_mask = GMP_mask(model_and_loss.model, channel_sparsity-init_channel_ratio)
        apply_channel_mask(model_and_loss.model, cfg_mask)
        print('apply init. mask | detect channel zero: {}'.format(detect_channel_zero(model_and_loss.model)))

    if args.prune_dense:
        cfg_mask = Pruning(model_and_loss.model, channel_sparsity, train_loader)
        apply_channel_mask(model_and_loss.model, cfg_mask)
        print('apply init. mask | detect channel zero: {}'.format(detect_channel_zero(model_and_loss.model)))
    #########################################################################
    
    for epoch in range(start_epoch, end_epoch):
        if logger is not None:
            logger.start_epoch()

        #####################################################################
        '''
        update mask
        '''
        if args.grow_prune:
            # channel-wise
            if epoch >= 1 and epoch <= int(T_max) and epoch % delta_T == 0:
                channel_ratio = init_channel_ratio * (1 + cos(pi * (epoch) / (int(T_max)))) / 2 # cosine decay
                # channel_ratio = init_channel_ratio * (1 - epoch / T_max) # linear decay
                # channel_ratio = init_channel_ratio # constant decay
                # if epoch == T_max: # constant decay
                #     channel_ratio = 0 # constant decay
                layer_ratio_down = get_layer_ratio(model_and_loss.model, channel_sparsity) # for prune
                if args.GP and epoch != T_max:
                    xp, yp = bayesian_optimisation(model_and_loss, val_loader, fp16, prof, channel_sparsity, n_iters=50, x0=np.array(layer_ratio_down), random_search=1000000)
                    layer_ratio_down = xp[np.argmax(yp)].tolist()
                # layer_ratio_up = regrow_allocation(model_and_loss.model, channel_ratio, layer_ratio_down) # for grow
                layer_ratio_up = get_layer_ratio(model_and_loss.model, channel_sparsity-channel_ratio)
                print('layer ratio up:', layer_ratio_up)
                print('layer ratio down:', layer_ratio_down)
                # _,rank = SI_pruning(model_and_loss.model, train_loader)
                if args.sampling:
                    # if sampling is used, layer_ratio_up should be generated by get_layer_ratio() function
                    cfg_mask, prev_model = sampling_update_channel_mask(model_and_loss.model, layer_ratio_up, layer_ratio_down, prev_model)
                else:
                    if args.EMA:
                        cfg_mask, _ = update_channel_mask(model_and_loss.model, layer_ratio_up, layer_ratio_down, ema) # regrow EMA weight
                    else:
                        cfg_mask, prev_model = update_channel_mask(model_and_loss.model, layer_ratio_up, layer_ratio_down, prev_model, args)
                apply_channel_mask(model_and_loss.model, cfg_mask)
            # # weigth-wise
            # lb_v = weight_sparsity
            # ub_v = weight_sparsity - init_weight_ratio * (1 + cos(pi * (epoch) / (T_max))) / 2
            # prune_update(epoch, lb_v, ub_v)
            # prune_apply_masks()
                print('apply updated mask | detect channel zero: {}'.format(detect_channel_zero(model_and_loss.model)))

        if args.GMP:
            if epoch >= 1 and epoch <= int(T_max) and epoch % delta_T == 0:
                channel_ratio = init_channel_ratio * (1 + cos(pi * (epoch) / (int(T_max)))) / 2
                cfg_mask = GMP_mask(model_and_loss.model, channel_sparsity-channel_ratio)
                apply_channel_mask(model_and_loss.model, cfg_mask)
                print('apply updated mask | detect channel zero: {}'.format(detect_channel_zero(model_and_loss.model)))
        #####################################################################
            
        if not skip_training:
            train(
                train_loader,
                model_and_loss,
                ema,
                optimizer,
                lr_scheduler,
                fp16,
                logger,
                epoch,
                use_amp=use_amp,
                prof=prof,
                register_metrics=epoch == start_epoch,
                batch_size_multiplier=batch_size_multiplier,
                cfg_mask=cfg_mask,
            )

        print('****** training ******| detect channel zero: {}'.format(detect_channel_zero(model_and_loss.model)))

        if not skip_validation:
            prec1, nimg = validate(
                val_loader,
                model_and_loss,
                fp16,
                logger,
                epoch,
                prof=prof,
                register_metrics=epoch == start_epoch,
            )
        if logger is not None:
            logger.end_epoch()

        if not skip_validation:
            is_best = logger.metrics["val.top1"]["meter"].get_epoch() > best_prec1
            best_prec1 = max(
                logger.metrics["val.top1"]["meter"].get_epoch(), best_prec1
            )
        else:
            is_best = False
            best_prec1 = 0

        if should_backup_checkpoint(epoch):
            backup_filename = "checkpoint-{}.pth.tar".format(epoch + 1)
        else:
            backup_filename = None
        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": model_and_loss.arch,
                "state_dict": model_and_loss.model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            checkpoint_dir=checkpoint_dir,
            backup_filename=backup_filename,
            filename=checkpoint_filename,
        )


# }}}
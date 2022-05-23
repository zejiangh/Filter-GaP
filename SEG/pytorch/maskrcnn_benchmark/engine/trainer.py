# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist
from copy import deepcopy
import sys
import torch.nn as nn
import numpy as np
from math import cos, pi


from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

try:
    from apex import amp
    use_amp = True
except ImportError:
    print('Use APEX for multi-precision via apex.amp')
    use_amp = False

# ==================================================================================================================================================================
def L1_norm(layer):
    weight_copy = layer.weight.data.abs().clone().cpu().numpy()
    norm = np.sum(weight_copy, axis=(1,2,3))
    return norm

def CSS(layer, k):
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
    l1 = [3,6,9, 13,16,19,22, 26,29,32,35,38,41, 45,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [2,12,25,44]
    total = 0
    bn_count = 1
    for m in model.modules():
        if isinstance(m, nn.GroupNorm):
            if bn_count in l1 + l2 + skip:
                total += m.weight.data.shape[0]
                bn_count += 1
                continue
            bn_count += 1
    bn = torch.zeros(total)
    index = 0
    bn_count = 1
    for m in model.modules():
        if isinstance(m, nn.GroupNorm):
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
    for m in model.modules():
        if isinstance(m, nn.GroupNorm):
            if bn_count in l1 + l2 + skip:
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
                layer_ratio.append((mask.shape[0] - torch.sum(mask).item()) / mask.shape[0])
                bn_count += 1
                continue
            bn_count += 1
    # segmentation demonstrates no meaningful channel config; use uniform
    layer_ratio = [sparsity] * len(layer_ratio)
    return layer_ratio

def regrow_allocation(model, delta_sparsity, layer_ratio_down):
    l1 = [3,6,9, 13,16,19,22, 26,29,32,35,38,41, 45,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [2,12,25,44]
    bn_count = 1
    idx = 0
    layer_ratio = []
    for m in model.modules():
        if isinstance(m, nn.GroupNorm):
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
    prev_model = deepcopy(model)
    l1 = [3,6,9, 13,16,19,22, 26,29,32,35,38,41, 45,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [2,12,25,44]
    layer_id = 1
    cfg_mask = []
    for m in model.modules():
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

def update_channel_mask(model, layer_ratio_up, layer_ratio_down, old_model):
    l1 = [3,6,9, 13,16,19,22, 26,29,32,35,38,41, 45,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [2,12,25,44]
    layer_id = 1
    idx = 0
    cfg_mask = []
    for [m, m0] in zip(model.modules(), old_model.modules()):
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if layer_id in l1:
                num_keep = int(out_channels*(1-layer_ratio_down[idx]))
                num_free = int(out_channels*(1-layer_ratio_up[idx])) - num_keep
                rank = np.argsort(CSS(m, layer_ratio_down[idx]))
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
        elif isinstance(m, nn.GroupNorm):
            if layer_id-1 in l1 + l2 + skip:
                w = m0.weight.data[copy_idx.tolist()].clone()
                m.weight.data[copy_idx.tolist()] = w.clone()
                b = m0.bias.data[copy_idx.tolist()].clone()
                m.bias.data[copy_idx.tolist()] = b.clone()
#                rm = m0.running_mean[copy_idx.tolist()].clone()
#                m.running_mean[copy_idx.tolist()] = rm.clone()
#                rv = m0.running_var[copy_idx.tolist()].clone()
#                m.running_var[copy_idx.tolist()] = rv.clone()
                continue
    prev_model = deepcopy(model)
    return cfg_mask, prev_model

def apply_channel_mask(model, cfg_mask):
    l1 = [3,6,9, 13,16,19,22, 26,29,32,35,38,41, 45,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [2,12,25,44]
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
        elif isinstance(m, nn.GroupNorm):
            if conv_count-1 in l1 + l2 + skip:
                mask = cfg_mask[layer_id_in_cfg-1].float().cuda()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                continue

def detect_channel_zero (model):
    l1 = [3,6,9, 13,16,19,22, 26,29,32,35,38,41, 45,48,51]
    l2 = (np.asarray(l1)+1).tolist()
    l3 = (np.asarray(l2)+1).tolist()
    skip = [2,12,25,44]
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
# ==================================================================================================================================================================

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    use_amp,
    cfg,
    dllogger,
    per_iter_end_callback_fn=None,
):
    dllogger.log(step="PARAMETER", data={"train_start": True})
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()


    ###########################################################################################
    T_max = int(max_iter * 0.75)
    channel_sparsity = 0.5
    init_channel_ratio = 0.5
    delta_T = 1000
    ### initialize mask
    cfg_mask, prev_model = init_channel_mask(model.module.backbone.body, channel_sparsity - init_channel_ratio)
    apply_channel_mask(model.module.backbone.body, cfg_mask)
    print('apply init. mask | detect channel sparsity: {}'.format(detect_channel_zero(model.module.backbone.body)))
    ###########################################################################################


    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):

        #########################################################################################################
        ### update mask
        if iteration >= 1 and iteration <= T_max and iteration % delta_T == 0:
            channel_ratio = init_channel_ratio * (1 + cos(pi * (iteration) / (T_max))) / 2
            layer_ratio_down = get_layer_ratio(model.module.backbone.body, channel_sparsity) # for prune
            layer_ratio_up = regrow_allocation(model.module.backbone.body, channel_ratio, layer_ratio_down) # for grow
            print('layer ratio up:', layer_ratio_up)
            print('layer ratio down:', layer_ratio_down)
            cfg_mask, prev_model = update_channel_mask(model.module.backbone.body, layer_ratio_up, layer_ratio_down, prev_model)
            apply_channel_mask(model.module.backbone.body, cfg_mask)
            print('apply updated mask | detect channel sparsity: {}'.format(detect_channel_zero(model.module.backbone.body)))
        #########################################################################################################


        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)


        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        if use_amp:        
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()
        else:
            losses.backward()

        if not cfg.SOLVER.ACCUMULATE_GRAD:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            #########################################################################################################
            ### maintain channel sparsity
            apply_channel_mask(model.module.backbone.body, cfg_mask)
            #########################################################################################################

        else:
            if (iteration + 1) % cfg.SOLVER.ACCUMULATE_STEPS == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.div_(cfg.SOLVER.ACCUMULATE_STEPS)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            log_data = {"eta":eta_string, "learning_rate":optimizer.param_groups[0]["lr"],
                        "memory": torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 }
            log_data.update(meters.get_dict())
            dllogger.log(step=(iteration,), data=log_data)

        if cfg.SAVE_CHECKPOINT:
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

        # per-epoch work (testing)
        if per_iter_end_callback_fn is not None:
            early_exit = per_iter_end_callback_fn(iteration=iteration)
            if early_exit:
                break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    dllogger.log(step=tuple(), data={"e2e_train_time": total_training_time,
                                                   "train_perf_fps": max_iter * cfg.SOLVER.IMS_PER_BATCH / total_training_time})
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info(
    "Total training time: {} ({:.4f} s / it)".format(
        total_time_str, total_training_time / (max_iter)
        )
    )


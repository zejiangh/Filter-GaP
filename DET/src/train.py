# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.autograd import Variable
import torch
import time
from SSD import _C as C

from apex import amp

import numpy as np
import torch.nn as nn

def apply_mask(model, cfg_mask):
    model = model.module.feature_extractor
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
            if conv_count in l2:
                mask = cfg_mask[layer_id_in_cfg-1].float().cuda()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                continue
            if conv_count in l3:
                mask = cfg_mask[layer_id_in_cfg-1].float().cuda()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                continue
            if conv_count-1 in skip:
                mask = cfg_mask[layer_id_in_cfg-1].float().cuda()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                continue

def train_loop(model, loss_func, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std, cfg_mask):
#     for nbatch, (img, _, img_size, bbox, label) in enumerate(train_dataloader):
    for nbatch, data in enumerate(train_dataloader):
        img = data[0][0][0]
        bbox = data[0][1][0]
        label = data[0][2][0]
        label = label.type(torch.cuda.LongTensor)
        bbox_offsets = data[0][3][0]
        # handle random flipping outside of DALI for now
        bbox_offsets = bbox_offsets.cuda()
        img, bbox = C.random_horiz_flip(img, bbox, bbox_offsets, 0.5, False)
        img.sub_(mean).div_(std)
        if not args.no_cuda:
            img = img.cuda()
            bbox = bbox.cuda()
            label = label.cuda()
            bbox_offsets = bbox_offsets.cuda()

        N = img.shape[0]
        if bbox_offsets[-1].item() == 0:
            print("No labels in batch")
            continue
        bbox, label = C.box_encoder(N, bbox, bbox_offsets, label, encoder.dboxes.cuda(), 0.5)
        # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
        M = bbox.shape[0] // N
        bbox = bbox.view(N, M, 4)
        label = label.view(N, M)

        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()

        trans_bbox = bbox.transpose(1, 2).contiguous().cuda()

        if not args.no_cuda:
            label = label.cuda()
        gloc = Variable(trans_bbox, requires_grad=False)
        glabel = Variable(label, requires_grad=False)

        loss = loss_func(ploc, plabel, gloc, glabel)

        if args.local_rank == 0:
            logger.update_iter(epoch, iteration, loss.item())

        if args.amp:
            with amp.scale_loss(loss, optim) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()

        if args.warmup is not None:
            warmup(optim, args.warmup, iteration, args.learning_rate)

        optim.step()
        optim.zero_grad()
        iteration += 1
        

        if not cfg_mask is None:
            apply_mask(model, cfg_mask)
            

    return iteration


def benchmark_train_loop(model, loss_func, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std):
    start_time = None
    # tensor for results
    result = torch.zeros((1,)).cuda()
    for i, data in enumerate(loop(train_dataloader)):
        if i >= args.benchmark_warmup:
            torch.cuda.synchronize()
            start_time = time.time()

        img = data[0][0][0]
        bbox = data[0][1][0]
        label = data[0][2][0]
        label = label.type(torch.cuda.LongTensor)
        bbox_offsets = data[0][3][0]
        # handle random flipping outside of DALI for now
        bbox_offsets = bbox_offsets.cuda()
        img, bbox = C.random_horiz_flip(img, bbox, bbox_offsets, 0.5, False)

        if not args.no_cuda:
            img = img.cuda()
            bbox = bbox.cuda()
            label = label.cuda()
            bbox_offsets = bbox_offsets.cuda()
        img.sub_(mean).div_(std)

        N = img.shape[0]
        if bbox_offsets[-1].item() == 0:
            print("No labels in batch")
            continue
        bbox, label = C.box_encoder(N, bbox, bbox_offsets, label, encoder.dboxes.cuda(), 0.5)

        M = bbox.shape[0] // N
        bbox = bbox.view(N, M, 4)
        label = label.view(N, M)





        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()

        trans_bbox = bbox.transpose(1, 2).contiguous().cuda()

        if not args.no_cuda:
            label = label.cuda()
        gloc = Variable(trans_bbox, requires_grad=False)
        glabel = Variable(label, requires_grad=False)

        loss = loss_func(ploc, plabel, gloc, glabel)



        # loss scaling
        if args.amp:
            with amp.scale_loss(loss, optim) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()

        optim.step()
        optim.zero_grad()

        if i >= args.benchmark_warmup + args.benchmark_iterations:
            break

        if i >= args.benchmark_warmup:
            torch.cuda.synchronize()
            logger.update(args.batch_size, time.time() - start_time)


    result.data[0] = logger.print_result()
    if args.N_gpu > 1:
        torch.distributed.reduce(result, 0)
    if args.local_rank == 0:
        print('Training performance = {} FPS'.format(float(result.data[0])))



def loop(dataloader, reset=True):
    while True:
        for data in dataloader:
            yield data
        if reset:
            dataloader.reset()

def benchmark_inference_loop(model, loss_func, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std):
    assert args.N_gpu == 1, 'Inference benchmark only on 1 gpu'
    start_time = None
    model.eval()

    i = -1
    val_datas = loop(val_dataloader, False)

    while True:
        i += 1
        torch.cuda.synchronize()
        if i >= args.benchmark_warmup:
            start_time = time.time()

        data = next(val_datas)

        with torch.no_grad():
            img = data[0]
            if not args.no_cuda:
                img = img.cuda()
            if args.amp:
                img = img.half()
            img.sub_(mean).div_(std)
            img = Variable(img, requires_grad=False)
            _ = model(img)
            torch.cuda.synchronize()

            if i >= args.benchmark_warmup + args.benchmark_iterations:
                break

            if i >= args.benchmark_warmup:
                logger.update(args.eval_batch_size, time.time() - start_time)

    logger.print_result()

def warmup(optim, warmup_iters, iteration, base_lr):
    if iteration < warmup_iters:
        new_lr = 1. * base_lr / warmup_iters * iteration
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr


def load_checkpoint(model, checkpoint):
    """
    Load model from checkpoint.
    """
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'N.' from checkpoint that comes from DDP wrapper
    saved_model = od["model"]
    model.load_state_dict(saved_model)


def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]

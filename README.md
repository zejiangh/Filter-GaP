# FilterExpo: CNN Model Compression via Filter Exploration

This repository is the official implementation of the paper "FilterExpo: CNN Model Compression via Filter Exploration".

## Highlights
Filter-wise network pruning has been broadly recognized as an effective technique to reduce the computation and memory cost of deep convolutional neural networks. However, conventional pruning methods require a fully pre-trained large model, and remove filters in one-shot or iterative manners unidirectionally, which result to sub-optimal model quality, large memory footprint and expensive training cost. In this paper, we propose a novel Filter Exploration methodology, dubbed as FilterExpo. It repeatedly prunes and regrows the filters throughout the training process, which reduces the risk of pruning important filters prematurely. It also balances the number of filters across all layers with a  sparsity constraint. In addition, we convert the filter pruning problem to the well known column subset selection (CSS) problem, which produces better results than previous heuristic pruning methods. All the exploration process is done in a single training from scratch without the need of a pre-trained large model.  Experimental results demonstrate that our method can effectively reduce the FLOPs of diverse CNN architectures on a variety of computer vision tasks, including image classification, object detection, instance segmentation, and 3D vision. For example, our compressed ResNet-50 model on ImageNet dataset achieves 76% Top-1 accuracy with only 25% FLOPs of the original ResNet-50 model, improving previous state-of-the-art filter pruning method by 0.7%.

<div align="center">
  <img width="100%" src="figs/overview.png">
</div>

## Dependency

<!-- This repository contains Dockerfile which extends the PyTorch NGC container and encapsulates some dependencies. In addition, ensure you have the following components: -->

<!-- * [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) -->
<!-- * [PyTorch 20.12-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch) or newer -->
<!-- * Supported GPUs: -->
<!--     * [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) -->

## ImageNet Classification

### Prepare the dataset

* We use ImageNet-1K, a widely used image classification dataset from the ILSVRC challenge. 
* [Download the images](http://image-net.org/download-images).
* Extract the training data
```Shell
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```

* Extract the validation data and move the images to subfolders
```Shell
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

* Docker setup
```Shell
docker build . -t nvidia_resnet50
nvidia-docker run --rm -it -v <path to imagenet>:/imagenet --ipc=host nvidia_resnet50
```

### Checkpoints
<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>linear</th>
    <th>k-nn</th>
    <th colspan="1">download</th>
    <th colspan="3">logs</th>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>23M</td>
    <td>75.7%</td>
    <td>71.3%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/resnet/resnet50/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/resume_from_ckpt0200/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/resnet/resnet50/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/resume_from_ckpt0200/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/resnet/resnet50/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/resume_from_ckpt0200/lincls/epoch_last/lr0.01/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/resnet/resnet50/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/resume_from_ckpt0200/features/epoch0300/log.txt">knn</a></td>    
  </tr>  
  <tr>
    <td>EsViT (Swin-T, W=7)</td>
    <td>28M</td>
    <td>78.0%</td>
    <td>75.7%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/checkpoint_best.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/lincls/epoch0300/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/features/epoch0280/log.txt">knn</a></td>    
  </tr>
  <tr>
    <td>EsViT (Swin-S, W=7)</td>
    <td>49M</td>
    <td>79.5%</td>
    <td>77.7%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/checkpoint_best.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/lincls/epoch0300/lr_0.003_n_last_blocks4/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/features/epoch0280/log.txt">knn</a></td>   
  </tr>
  <tr>
    <td>EsViT (Swin-B, W=7)</td>
    <td>87M</td>
    <td>80.4%</td>
    <td>78.9%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug/continued_from0200_dense/checkpoint_best.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug/continued_from0200_dense/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug/continued_from0200_dense/lincls/epoch0260/lr_0.001_n_last_blocks4/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug/continued_from0200_dense/features/epoch0260/log.txt">knn</a></td> 

### Evaluation
* We released the pruned model at ```./RN50/logs/resnet50_2g_0.774.pth.tar``` (ResNet50 with 2GFLOPs and 77.4% Top-1) for direct evaluation.
* Start inference
```Shell
python ./main.py --data-backend pytorch --arch resnet50 --evaluate --pruned_model ./logs/resnet50_2g_0.774.pth.tar -b 128 /data/imagenet
```
* FLOPs checking
```Shell
python check_flops.py --checkpoint_path ./logs/resnet50_2g_0.774.pth.tar
```

### Training from scratch
```Shell
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend pytorch --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --mixup 0. --grow_prune --delta_T 2 --T_max 0.72 --init_channel_ratio 0.2 --channel_sparsity 0.5 --sampling
```


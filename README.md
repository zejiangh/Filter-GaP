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
We provide the checkpoints of the compressed ResNet models on ImageNet. You can download and evaluate them directly.

<table>
  <tr>
    <th>Model</th>
    <th>FLOPs</th>
    <th>FLOPs reduction</th>
    <th>Top-1</th>
    <th colspan="1">Download</th>
  </tr>
  <tr>
    <td>ResNet-18</td>
    <td>1.03G</td>
    <td>43%</td>
    <td>69.4%</td>
    <td><a href="">resnet18_1gflops_69.4top1 ckpt</a></td>   
  </tr>
  <tr>
    <td>ResNet-34</td>
    <td>2G</td>
    <td>46%</td>
    <td>73.5%</td>
    <td><a href="">resnet34_2gflops_73.5top1 ckpt</a></td>   
  </tr>
  <tr>
    <td rowspan="4">ResNet-50</td>
    <td>3G</td>
    <td>27%</td>
    <td>77.9%</td>
    <td><a href="">resnet50_3gflops_77.9top1 ckpt</a></td>   
  </tr>
  <tr>
    <td>2G</td>
    <td>50%</td>
    <td>77.4%</td>
    <td><a href="">resnet50_2gflops_77.4top1 ckpt</a></td>   
  </tr> 
  <tr>
    <td>1G</td>
    <td>75%</td>
    <td>76%</td>
    <td><a href="">resnet50_1gflops_76top1 ckpt</a></td>   
  </tr> 
  <tr>
    <td>0.8G</td>
    <td>80%</td>
    <td>74.4%</td>
    <td><a href="">resnet50_0.8gflops_74.4top1 ckpt</a></td>   
  </tr> 
  <tr>
    <td rowspan="2">ResNet-101</td>
    <td>3.4G</td>
    <td>55%</td>
    <td>78.8%</td>
    <td><a href="">resnet101_3.4gflops_78.8top1 ckpt</a></td>   
  </tr> 
  <tr>
    <td>1.9G</td>
    <td>75%</td>
    <td>77.6%</td>
    <td><a href="">resnet101_1.9gflops_77.6top1 ckpt</a></td>   
  </tr> 
</table>

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


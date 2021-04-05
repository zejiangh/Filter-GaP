# Filter Explorations for Convolutional NeuralNetworks

This repository is an official PyTorch implementation of the paper "Filter Explorations for Convolutional NeuralNetworks".

## Dependency

This repository contains Dockerfile which extends the PyTorch NGC container and encapsulates some dependencies. In addition, out code runs with the following components:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 20.12-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch) or newer
* Supported GPUs:
    * [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)

## ImageNet Classification

### Download the dataset

1. The script operates on ImageNet-1K, a widely popular image classification dataset from the ILSVRC challenge. 
2. [Download the images](http://image-net.org/download-images).

### Training

```Shell
python main.py --model DDHRN --scale 4 --patch_size 192  --save DDHRN_scalex4 --reset --epochs 800 --n_GPUs 2 --n_resblocks 10 --n_resgroups 10 --n_feats 64 --data_range '1-800/801-810'
```

### Testing

```Shell
python main.py --model DDHRN --scale 4 --reset --n_GPUs 1 --n_resblocks 10 --n_resgroups 10 --n_feats 64 --data_test Set5+Set14+B100+Urban100 --pre_train ../model/DDHRN_scalex4.pt --test_only --save_results
```

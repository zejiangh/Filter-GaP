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

* The script operates on ImageNet-1K, a widely popular image classification dataset from the ILSVRC challenge. 
* [Download the images](http://image-net.org/download-images).



### Training

```Shell
git clone https://github.com/zejiangh/Filter-GaP
cd RN50/

python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 250 --mixup 0.2 --grow_prune --delta_T 2 --T_max 180 --init_channel_ratio 0.3 --channel_sparsity 0.7

```

### Testing

```Shell
python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained-weights ./logs/model_best.pth.tar -b <batch size> <path to imagenet>
```

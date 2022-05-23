docker build . -t nvidia_rn50

nvidia-docker run --rm -it -v /mnt/dataset/imagenet:/data/imagenet -v /mnt/zejiang.hou/CHEX:/workspace/rn50 --ipc=host nvidia_rn50

### evaluation
CUDA_VISIBLE_DEVICES=0 python ./main.py --data-backend pytorch --arch resnet50 --evaluate --epochs 1 -b 100 /data/imagenet --pretrained-weights /workspace/rn50/checkpoints/resnet50_1gflops_76top1.pth.tar 

### training
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend pytorch --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 250 --grow_prune --delta_T 2 --T_max 180 --init_channel_ratio 0.2 --channel_sparsity 0.5
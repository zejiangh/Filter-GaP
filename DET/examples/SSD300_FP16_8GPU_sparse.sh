### evaluation
python ./main.py --backbone resnet50 --mode evaluation --checkpoint /workspace/checkpoints/ssd_50%flops_25.9ap.pth.tar --data /coco

### training
python -m torch.distributed.launch --nproc_per_node=8 ./main.py --backbone resnet50 --warmup 300 --bs 64 --amp --data /coco --epoch 650 --multistep 430 540 --grow_prune --channel_sparsity 0.7 --init_channel_ratio 0.2 --delta_T 2 --T_max 470 --save ./logs

python -m torch.distributed.launch --nproc_per_node=8 ./main.py --backbone resnet50 --warmup 300 --bs 64 --amp --data /coco --epoch 650 --multistep 430 540 --grow_prune --channel_sparsity 0.5 --init_channel_ratio 0.2 --delta_T 2 --T_max 470 --save ./logs
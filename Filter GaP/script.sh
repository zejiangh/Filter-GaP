docker build . -t nvidia_rn50

# gpu 02-04
nvidia-docker run --rm -it -v /mnt/ssd_old/minghai/dataset/imagenet:/data/imagenet --ipc=host nvidia_rn50
# gpu 01
nvidia-docker run --rm -it -v /mnt/ssd_old/minghai.qin/dataset/imagenet:/data/imagenet --ipc=host nvidia_rn50
# gpu 05
nvidia-docker run --rm -it -v /mnt/ssd/dataset/imagenet:/data/imagenet --ipc=host nvidia_rn50
# aliyun
nvidia-docker run --rm -it -v /mnt/dataset/imagenet:/data/imagenet --ipc=host nvidia_rn50
nvidia-docker run --rm -it -v /mnt/dataset/imagenet:/data/imagenet -v /mnt/princeton.univ/RN50_gap:/workspace/rn50 --ipc=host nvidia_rn50


scp /Users/zejiangh/Downloads/grow_and_prune/RN50_gap/image_classification/training.py princeton.univ@8.140.119.55:/mnt/princeton.univ/RN50_gap/image_classification/training.py
scp /Users/zejiangh/Downloads/grow_and_prune/RN50_gap/main.py princeton.univ@8.140.119.55:/mnt/princeton.univ/RN50_gap/main.py

# baseline
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 250 --mixup 0.2

# baseline: resnet50-D + SE + dropout (+ random augment)
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend randomaug --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 250 --mixup 0.2
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 250 --mixup 0.2

# filter-wise random gap
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 250 --mixup 0.2 --grow_prune --delta_T 2 --T_max 180 --init_channel_ratio 0.2 --channel_sparsity 0.5 --gather-checkpoints

# sampling based regrowth
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 300 --mixup 0.2 --grow_prune --delta_T 2 --T_max 216 --init_channel_ratio 0.3 --channel_sparsity 0.7 --sampling

# expand and grow-and-prune (for comparison with efficientnet)
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 350 --mixup 0.2 --grow_prune --delta_T 2 --T_max 250 --init_channel_ratio 0.2 --channel_sparsity 0.5 --expand

# Gaussian Process
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 250 --mixup 0.2 --grow_prune --delta_T 2 --T_max 180 --init_channel_ratio 0.2 --channel_sparsity 0.7 --GP

# regrow EMA weight
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 250 --mixup 0.2 --grow_prune --delta_T 2 --T_max 180 --init_channel_ratio 0.2 --channel_sparsity 0.7 --EMA

# A100 small epochs
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend pytorch --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 5 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 150 --mixup 0 --grow_prune --delta_T 2 --T_max 100 --init_channel_ratio 0.3 --channel_sparsity 0.7 --gather-checkpoints

python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend pytorch --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 5 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 150 --mixup 0 --grow_prune --delta_T 2 --T_max 100 --init_channel_ratio 0.3 --channel_sparsity 0.5 --gather-checkpoints

# prune pretrained model
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 5 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 100 --mixup 0 --prune_dense --channel_sparsity 0.7 --pruned_model ./ep_250/model_best.pth.tar

python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr 2.048 --optimizer-batch-size 2048 --warmup 5 --arch resnet50 -c fanin --label-smoothing 0 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 256 --amp --static-loss-scale 128 --epochs 300 --mixup 0 --prune_dense --channel_sparsity 0.7 --pruned_model ./ep_250/model_best.pth.tar
















# weight-wise random gap
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend pytorch --raport-file raport.json -j8 -p 100 --lr 2.048 --optimizer-batch-size 2048 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 256 --amp --static-loss-scale 128 --epochs 250 --mixup 0.2 

--sp-backbone --sp-mask-update-freq 2 --sp-update-init-method zero --sp-retrain --retrain-mask-pattern=random --retrain-mask-seed=0 --sp-config-file=profiles/p90_95.yaml 

--weight_sparsity 0.9 --init_weight_ratio 0.05 --T_max 240 --sp-global-magnitude --retrain --pruned_model ./logs/refined_resnet50_from_1.25x_scratch.pth.tar

# joint
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend pytorch --raport-file raport.json -j8 -p 100 --lr 2.048 --optimizer-batch-size 2048 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 256 --amp --static-loss-scale 128 --epochs 250 --mixup 0.2 --sp-backbone --sp-mask-update-freq 1 --sp-update-init-method zero --sp-retrain --retrain-mask-pattern=random --retrain-mask-seed=0 --sp-config-file=profiles/p92_85.yaml --weight_sparsity 0.925 --init_weight_ratio 0.075 --T_max 240 --sp-global-magnitude --channel_sparsity 0.2 --init_channel_ratio 0.2 --delta_T 5 --early_bird 1.0

python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend pytorch --raport-file raport.json -j8 -p 100 --lr 2.048 --optimizer-batch-size 2048 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 256 --amp --static-loss-scale 128 --epochs 250 --mixup 0.2 --sp-backbone --sp-mask-update-freq 1 --sp-update-init-method zero --sp-retrain --retrain-mask-pattern=random --retrain-mask-seed=0 --sp-config-file=profiles/p92_85.yaml --weight_sparsity 0.94 --init_weight_ratio 0.05 --T_max 240 --sp-global-magnitude --channel_sparsity 0.29 --init_channel_ratio 0.29 --delta_T 5 --early_bird 1.0

# finetuning
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend pytorch --raport-file raport.json -j8 -p 100 --lr 1.024 --optimizer-batch-size 1024 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 128 --amp --static-loss-scale 128 --epochs 250 --mixup 0.2 --retrain --pruned_model ./pruned/dynamic_slimming_0.7_0.2_pruneskip_gp2r_scratch.pth.tar --sparse_training

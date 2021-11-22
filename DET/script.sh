docker build . -t nvidia_ssd

nvidia-docker run --rm -it --ulimit memlock=-1 --ulimit stack=67108864 -v /mnt/dataset/COCO2017/data:/coco -v /mnt/zejiang.hou/CHEX:/workspace --ipc=host nvidia_ssd
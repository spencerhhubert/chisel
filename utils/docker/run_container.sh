#! /bin/bash
IMAGE=$1
sudo docker run -d \
    -v /home/spencer/code/chisel/:/chisel/ \
    -v /mnt/red/ABC-Dataset:/chisel/data/ABC-Dataset \
    --gpus all -it $IMAGE

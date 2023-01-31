#! /bin/bash
IMAGE="chisel_training"
sudo docker run -d \
    -v /home/spencer/code/chisel/:/chisel/ \
    -v /mnt/red/ABC-Dataset:/chisel/data/ABC-Dataset \
    --gpus all \
    --name "chisel_training" \
    -it $IMAGE

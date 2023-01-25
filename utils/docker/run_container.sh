#! /bin/bash
IMAGE=$1
sudo docker run -d -v /mnt/red/ABC-Dataset/:/chisel/data/ABC-Dataset/ --gpus all -it $IMAGE

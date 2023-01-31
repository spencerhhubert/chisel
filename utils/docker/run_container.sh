#! /bin/bash
IMAGE="chisel_training"
CONTAINER="chisel_training"

docker stop $CONTAINER
docker rm $CONTAINER

sudo docker run -d \
    -v /home/spencer/code/chisel/:/chisel/ \
    -v /mnt/red/ABC-Dataset:/chisel/data/ABC-Dataset \
    --gpus all \
    --name $CONTAINER \
    -it $IMAGE

xhost +local:root
DATA_PATH=~/
docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --shm-size 16G \
    -v $DATA_PATH:/ws \
    --network=host --name 3dda -it 3dda

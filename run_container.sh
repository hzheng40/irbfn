#!/usr/bin/env bash

xhost +
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
docker run --gpus all -it \
    -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name irbfn_container \
    -v $SCRIPT_DIR/:/home/irbfn \
    irbfn

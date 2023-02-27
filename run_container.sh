#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
docker run --gpus all -it \
    --name irbfn_container \
    -v $SCRIPT_DIR/:/home/irbfn \
    irbfn

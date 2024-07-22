#!/usr/bin/env bash

# xhost +
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
docker run --gpus all -it \
    --name irbfn_container \
    -v $SCRIPT_DIR/:/home/irbfn \
    -v /data/billyz/irbfn_explicit_mpc_tables:/data/tables \
    irbfn

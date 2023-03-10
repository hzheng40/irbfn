# Interpolating Radial Basis Function Networks
Code for IROS 2023 paper: Differentiable Trajectory Generation for Car-like Robots by Interpolating Radial Basis Function Networks
![Trajectory](traj_out.png "Generated Trajectories")

## Prerequisite

1. NVIDIA driver version 520.61.5+ is required.

## Installation

1. Clone this repo, then `cd irbfn`
2. Update submodule. `git submodule sync && git submodule update --init --force`
3. Build docker image from Dockerfile:
```bash
sudo docker build -t irbfn -f Dockerfile .
```
1. Run the docker container:
```bash
sudo ./run_container.sh
```

## Run evaluation

```bash
python3 evaluate.py
```

## Re-run training

1. Download training dataset. See instructions [here](data/download_data.md).
2. Run train script `python3 train.py`. Updated config and checkpoint files should be saved to `ckpts/` and `configs/`.
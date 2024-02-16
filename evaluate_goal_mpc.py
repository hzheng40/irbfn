# MIT License

# Copyright (c) 2023 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author: Hongrui Zheng
# Last Modified: 11/28/2022

import os
import argparse
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax.training import train_state, checkpoints
import matplotlib.pyplot as plt
from flax_rbf import *
from model import WCRBFNet
from tqdm import tqdm
import time


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

flax.config.update('flax_use_orbax_checkpointing', False)

config_f = "configs/goal_mpc_20240214_043527.yaml"
ckpt = "ckpts/goal_mpc_20240214_043527/checkpoint_0"
with open(config_f, "r") as f:
    config_dict = yaml.safe_load(f)
conf = argparse.Namespace(**config_dict)

# # uncomment the following line (53) and line 206 to create memory profile, will slow down inference significantly
# # see https://jax.readthedocs.io/en/latest/profiling.html#programmatic-capture for how to inspect
# # jax.profiler.start_trace('./tensorboard_profiler')


# pred one step
@jax.jit
def pred_step(state, x):
    y = state.apply_fn(state.params, x)
    return y


# load checkpoint
wcrbf = WCRBFNet(
    in_features=conf.in_features,
    out_features=conf.out_features,
    num_kernels=conf.num_kernels,
    basis_func=eval(conf.basis_func),
    num_regions=conf.num_regions,
    lower_bounds=conf.lower_bounds,
    upper_bounds=conf.upper_bounds,
    dimension_ranges=conf.dimension_ranges,
    activation_idx=conf.activation_idx,
    delta=conf.delta,
)

rng = jax.random.PRNGKey(conf.seed)
rng, init_rng = jax.random.split(rng)
params = wcrbf.init(init_rng, jnp.ones((1, conf.in_features)))
optim = optax.adam(conf.lr)
state = train_state.TrainState.create(apply_fn=wcrbf.apply, params=params, tx=optim)
restored_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt, target=state)

print('Importing data...')
data = np.load(f'goal_mpc_lookup_table_tiny.npz')
table = data['table']
v_s = table[:, 0].flatten()
x_g = table[:, 1].flatten()
y_g = table[:, 2].flatten()
t_g = table[:, 3].flatten()
v_g = table[:, 4].flatten()
speed = table[:, 5].flatten()
steer = table[:, 6].flatten()

v_s = np.concatenate((v_s, v_s))
x_g = np.concatenate((x_g, x_g))
y_g = np.concatenate((y_g, -y_g))
t_g = np.concatenate((t_g, -t_g))
v_g = np.concatenate((v_g, v_g))
speed = np.concatenate((speed, speed))
steer = np.concatenate((steer, -steer))

flattened_input = np.vstack([v_s, x_g, y_g, t_g, v_g]).T
flattened_output = np.vstack([speed, steer]).T

# predictions
# inputs = [arr.reshape(1, 75 for arr in flattened_input[:1000]]
# start = time.time()
# st = time.process_time()
# for inp in inputs:
#     pred_step(restored_state, inp)
# end = time.time()
# et = time.process_time()
# print('Execution time:', end - start, 'seconds')
# print('CPU Execution time:', et - st, 'seconds')
print('Prediction running...')
pred_u = pred_step(restored_state, flattened_input)
print('Prediction done')
speed_ae = np.abs(flattened_output[:, 0] - pred_u[:, 0])
steer_ae = np.abs(flattened_output[:, 1] - pred_u[:, 1])

speed_mae = np.average(speed_ae)
steer_mae = np.average(steer_ae)

print('Speed MAE:', speed_mae)
print('Steer MAE:', steer_mae)

print('Speed Median:', np.median(speed_ae))
print('Steer Median:', np.median(steer_ae))
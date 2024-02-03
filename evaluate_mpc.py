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


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

flax.config.update('flax_use_orbax_checkpointing', False)


config_f = "configs/basic_mpc_2000x2000.yaml"
ckpt = "ckpts/basic_mpc_2000x2000/checkpoint_0"
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


# predictions
test_samples = 100
test_data = np.loadtxt(f'data/valid_data_{test_samples}x{test_samples}.csv', delimiter=',')
test_x = test_data[:, 0].flatten()
test_y = test_data[:, 1].flatten()
test_u = test_data[:, 2].flatten()

test = jnp.hstack((test_x.reshape(-1, 1), test_y.reshape(-1, 1)))

pred_u = pred_step(restored_state, test)

valid_data = np.transpose(np.vstack([test_x, test_y, pred_u.flatten()]))
np.savetxt(f'data/pred_2000_data_{test_samples}x{test_samples}.csv', valid_data, fmt='%f', delimiter=',')

# xlim = 15

# # plot
# fig = plt.figure(figsize=(24, 8))
# ax1 = fig.add_subplot(1, 3, 1, projection='3d')
# ax1.scatter(test_x, test_y, test_u, c='red', label='actual')
# ax1.set_xlabel('x1')
# ax1.set_ylabel('x2')
# ax1.set_zlabel('u')
# ax1.legend()

# ax2 = fig.add_subplot(1, 3, 2, projection='3d')
# ax2.scatter(test_x, test_y, pred_u, c='blue', label='predicted')
# ax2.set_xlabel('x1')
# ax2.set_ylabel('x2')
# ax2.set_zlabel('u')
# ax2.legend()

# plt.subplot(1, 3, 3)
# plt.scatter(test_x, test_y, c=np.subtract(pred_u.flatten(), test_u), cmap='RdYlBu')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.grid(True)
# # Add box constraint on x plot
# plt.plot([-xlim, -xlim], [-xlim, xlim], color="black")
# plt.plot([xlim, xlim], [-xlim, xlim], color="black")
# plt.plot([-xlim, xlim], [-xlim, -xlim], color="black")
# plt.plot([-xlim, xlim], [xlim, xlim], color="black", label="Constraints")
# plt.colorbar()
# plt.legend()

# plt.show()


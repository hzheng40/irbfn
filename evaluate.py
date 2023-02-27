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

#%%
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp
import numpy as np
import argparse
from flax.training import train_state, checkpoints
from model import WCRBFNet
from flax_rbf import *
import optax
import yaml
from utils import integrate_path_mult

#%%
# parser = argparse.ArgumentParser()
# parser.add_argument('--ckpt', type=str, required=True)
# parser.add_argument('--config', type=str, required=True)
# args = parser.parse_args()

config_f = '/home/irbfn/configs/default.yaml'
ckpt = '/home/irbfn/ckpts/checkpoint_0'
with open(config_f, 'r') as f:
    config_dict = yaml.safe_load(f)
conf = argparse.Namespace(**config_dict)


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
params = wcrbf.init(init_rng, jnp.ones((conf.batch_size, 4)))
optim = optax.adam(conf.lr)
state = train_state.TrainState.create(apply_fn=wcrbf.apply, params=params, tx=optim)
restored_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt, target=state)

#%%
# test_x = jnp.linspace(6, 6, 20)
test_x = jnp.array([4.])
test_y = jnp.linspace(-4.0, 4.0, 20)
test_t = jnp.linspace(-0.3, 0.3, 10)
# test kappas are zeros

# stacking testing points, final matrix should be shape (batch_size, 4)
xm, ym, tm = jnp.meshgrid(test_x, test_y, test_t, indexing='ij')
test_idx = jnp.stack((xm, ym, tm), axis=-1)
test_flat = test_idx.reshape((-1, 3))
test_kappa = jnp.zeros((test_flat.shape[0], 1))
test = jnp.hstack((test_flat, test_kappa))

y_pred = jnp.zeros((test.shape[0], 5))

# jit
y_pred_raw = pred_step(restored_state, test)
# y_pred should have columns (0, k1, k2, 0, s)
# y_pred_raw has (s, k1, k2)
y_pred = y_pred.at[:, -1].set(y_pred_raw[:, 0])
y_pred = y_pred.at[:, 1:3].set(y_pred_raw[:, 1:])

all_states = integrate_path_mult(y_pred)


#%%
y_pred_raw = pred_step(restored_state, test)
y_pred = jnp.zeros((test.shape[0], 5))
y_pred = y_pred.at[:, -1].set(y_pred_raw[:, 0])
y_pred = y_pred.at[:, 1:3].set(y_pred_raw[:, 1:])

all_states = integrate_path_mult(y_pred)

print(all_states.shape)

#%%
import matplotlib.pyplot as plt
for p_i in range(all_states.shape[0]):
    plt.plot(all_states[p_i, :, 0], all_states[p_i, :, 1])
plt.show()
# %%

import time
start = time.time()
y_pred_raw = pred_step(restored_state, test)
y_pred = jnp.zeros((test.shape[0], 5))
y_pred = y_pred.at[:, -1].set(y_pred_raw[:, 0])
y_pred = y_pred.at[:, 1:3].set(y_pred_raw[:, 1:])

all_states = integrate_path_mult(y_pred)
end = time.time()
print('elapsed:', end-start)

# %%

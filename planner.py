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
# Last Modified: 08/29/2023
# Simple lattice-based planner using IRBFNs

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training import checkpoints, train_state

from flax_rbf import *
from model import WCRBFNet
from utils import N, integrate_path_mult

config_f = "/home/irbfn/configs/20230414_170913_bs2000_lr0.001_nk100_11x10y8t_inverse_quadratic_kernel_l2_allkappalut.yaml"
ckpt = "/home/irbfn/ckpts/20230414_170913_bs2000_lr0.001_nk100_11x10y8t_inverse_quadratic_kernel_l2_allkappalut/checkpoint_0"
with open(config_f, "r") as f:
    config_dict = yaml.safe_load(f)
conf = argparse.Namespace(**config_dict)

@jax.jit
def pred_step(
    state: train_state.TrainState, x: jnp.DeviceArray
) -> jnp.DeviceArray:
    y = state.apply_fn(state.params, x)
    return y

@jax.jit
def gen_traj(state: train_state.TrainState, x: jnp.DeviceArray):
    return integrate_path_mult(pred_step(state, x))

@jax.jit
def softmin(x):
    a = jnp.exp(-100.0 * x)
    b = jnp.sum(jnp.exp(-100.0 * x))
    return a/b

@jax.jit
def softargmin(x):
    sm = softmin(x)
    pos = jnp.arange(len(x))
    return jnp.sum(sm * pos)

@jax.jit
def sample_grid(
    x_range: jnp.DeviceArray,
    y_range: jnp.DeviceArray,
    t_range: jnp.DeviceArray,
) -> jnp.DeviceArray:
    xm, ym, tm = jnp.meshgrid(x_range, y_range, t_range, indexing="ij")
    idx = jnp.stack((xm, ym, tm), axis=-1)
    idx_flat = idx.reshape((-1, 3))
    return idx_flat

class WCRBFPlanner(object):
    def __init__(self, waypoints, config, ckpt) -> None:
        # load config
        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)
        conf = argparse.Namespace(**config_dict)
        conf.x_min = 2.0
        conf.x_max = 6.0
        conf.num_x = 10
        conf.y_min = -4.0
        conf.y_max = 4.0
        conf.num_y = 10
        conf.t_min = -1.3
        conf.t_max = 1.3
        conf.num_t = 5

        # load model
        self.wcrbf = WCRBFNet(
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

        self.rng = jax.random.PRNGKey(conf.seed)
        self.rng, init_rng = jax.random.split(self.rng)
        params = self.wcrbf.init(init_rng, jnp.ones((1, 3)))
        optim = optax.adam(conf.lr)
        state = train_state.TrainState.create(
            apply_fn=self.wcrbf.apply, params=params, tx=optim
        )
        self.model_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt, target=state)

        # jit
        grid = sample_grid(
            jnp.linspace(conf.x_min, conf.x_max, conf.num_x),
            jnp.linspace(conf.y_min, conf.y_max, conf.num_y),
            jnp.linspace(conf.t_min, conf.t_max, conf.num_t),
        )
        _ = integrate_path_mult(pred_step(self.model_state, grid))


    def plan(self, x, y, theta):
        y_pred = pred_step(self.model_state, self.grid)
        all_states = integrate_path_mult(y_pred)

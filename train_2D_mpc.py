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
# Last Modified: 10/27/2022

import os
import yaml
from datetime import datetime

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import flax
from flax.core import unfreeze
from flax.training import train_state, checkpoints

import matplotlib.pyplot as plt

from flax_rbf import (
    gaussian,
    inverse_quadratic,
    inverse_multiquadric,
    quadratic,
    multiquadric,
    gaussian_wide,
    gaussian_wider,
)
from model import WCRBFNet

jax.config.update("jax_enable_x64", True)

# tensorboard logging
lr = 0.001
batch_size = 20000
numk = 10

# number of regions: 1x1 = 1
num_split_x1 = 1
num_split_x2 = 1

# loading raw data
print('Loading data...')
samples = 2000
table = np.load(f'valid_data_{samples}x{samples}.npz')['table']
print('Data Size:', table.shape)
X1 = table[:, 0].flatten()
X2 = table[:, 1].flatten()
u = table[:, 2].flatten()
print('Data import completed')

print('Generating bounds...')
x1_bounds = [min(X1), max(X1)]
x2_bounds = [min(X2), max(X2)]
print('Bounds defined')

print('Generating input and output...')
flattened_input = np.vstack([X1, X2]).T
flattened_output = np.atleast_2d(u).T
print('Data processing done')

# model parameters
in_features = flattened_input.shape[1]
out_features = flattened_output.shape[1]
num_regions = num_split_x1 * num_split_x2
lower_bounds = [x1_bounds[:-1], x2_bounds[:-1]]
upper_bounds = [x1_bounds[1:], x2_bounds[1:]]
activation_idx = [0, 1]
delta = [15.0, 15.0]
basis_function = inverse_quadratic
seed = 0
bound_ranges = [np.arange(len(curr_bounds)) for curr_bounds in lower_bounds]
dimension_ranges = (
    np.stack(np.meshgrid(*bound_ranges, indexing='ij'), axis=-1)
    .reshape(-1, len(bound_ranges))
    .tolist()
)

# rng
rng = jax.random.PRNGKey(seed)
rng, init_rng = jax.random.split(rng)

# model init
wcrbf = WCRBFNet(
    in_features=in_features,
    out_features=out_features,
    num_kernels=numk,
    basis_func=basis_function,
    num_regions=num_regions,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    dimension_ranges=dimension_ranges,
    activation_idx=activation_idx,
    delta=delta,
)
params = wcrbf.init(init_rng, jnp.ones((batch_size, in_features)))
params_shape = jax.tree_util.tree_map(jnp.shape, unfreeze(params))
print("Initialized parameter shapes:\n", params_shape)

# optimizer
optim = optax.adam(lr)

# train state
state = train_state.TrainState.create(apply_fn=wcrbf.apply, params=params, tx=optim)


# train one step
@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        y_pred = wcrbf.apply(params, x)
        loss = optax.l2_loss(predictions=y_pred, targets=y).mean()
        return loss
    # def loss_fn(params):
    #     y_pred = wcrbf.apply(params, x)
    #     chex.assert_type([y_pred], float)
    #     chex.assert_equal_shape((y_pred, y))
    #     loss = jnp.abs(y_pred - y).mean()
    #     return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss_, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_


# training parameters
num_points = flattened_output.shape[0]
epochs = 1000

# ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# train_work_dir = "./runs/goal_mpc_{}".format(ts)
train_work_dir = "./runs/2D_quadrotor_mpc"
writer = SummaryWriter(train_work_dir)

# config logging
yaml_dir = "./configs/" + train_work_dir[7:] + ".yaml"

# config logging
config_dict = {
    "in_features": in_features,
    "out_features": out_features,
    "num_kernels": numk,
    "basis_func": basis_function.__name__,
    "num_regions": num_regions,
    "lower_bounds": [[float(l) for l in ll] for ll in lower_bounds],
    "upper_bounds": [[float(u) for u in uu] for uu in upper_bounds],
    "dimension_ranges": dimension_ranges,
    "activation_idx": activation_idx,
    "delta": delta,
    "epochs": epochs,
    "lr": lr,
    "batch_size": batch_size,
    "seed": seed,
}
with open(yaml_dir, "w+") as outfile:
    yaml.dump(config_dict, outfile, default_flow_style=False)


def train_epoch(train_state, train_x, train_y, bs, epoch, epoch_rng, summary_writer):
    # batching data
    num_train = train_x.shape[0]
    num_steps = num_train // bs

    # random permutations
    perms = jax.random.permutation(epoch_rng, num_train)
    perms = perms[:num_steps*bs]
    perms = perms.reshape((num_steps, bs))
    batch_losses = []
    for b, perm in enumerate(perms):
        batch_x = train_x[perm, :]
        batch_y = train_y[perm, :]
        train_state, batch_loss = train_step(train_state, batch_x, batch_y)
        batch_losses.append(batch_loss)
        summary_writer.add_scalar(
            'train_loss_batch', jax.device_get(batch_loss), b + (epoch * len(perms))
        )
    
    batch_losses_np = jax.device_get(batch_losses)
    
    summary_writer.add_scalar('train_loss', np.mean(batch_losses_np), epoch)

    return train_state


# training
for e in tqdm(range(epochs)):
    rng, perm_rng = jax.random.split(rng)
    state = train_epoch(
        state,
        flattened_input,
        flattened_output,
        batch_size,
        e,
        perm_rng,
        writer
    )


CKPT_DIR = 'ckpts/' + train_work_dir[7:]


flax.config.update('flax_use_orbax_checkpointing', False)
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)
checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=0)


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
numk = 100

# number of regions: 1x1x2x2x1 = 4
num_split_v_car = 1
num_split_x_g = 1
num_split_y_g = 2
num_split_t_g = 2
num_split_v_g = 1

# loading raw data
print('Loading data...')
data = np.load('nmpc_lookup_table.npz')
inputs, outputs = data['inputs'], data['outputs']
v_c = inputs[:, 0].flatten()
x_g = inputs[:, 1].flatten()
y_g = inputs[:, 2].flatten()
t_g = inputs[:, 3].flatten()
v_g = inputs[:, 4].flatten()
accel = outputs[:, :, 0]
deltv = outputs[:, :, 1]
print('Data import completed')

print('Mirroring data...')
v_c_m = np.concatenate((v_c,  v_c), axis=0)
x_g_m = np.concatenate((x_g,  x_g), axis=0)
y_g_m = np.concatenate((y_g, -y_g), axis=0)
t_g_m = np.concatenate((t_g, -t_g), axis=0)
v_g_m = np.concatenate((v_g,  v_g), axis=0)
accel_m = np.concatenate((accel,  accel), axis=0)
deltv_m = np.concatenate((deltv, -deltv), axis=0)
print('Mirroring completed')

print('Generating bounds...')
v_c_bounds = [ 0.0, 7.3]
x_g_bounds = [ 0.0, 3.6]
y_g_bounds = [-3.6, 3.6]
t_g_bounds = [-3.2, 3.2]
v_g_bounds = [ 0.0, 7.3]
print('Bounds defined')

print('Generating input and output...')
flattened_input = np.vstack([v_c_m, x_g_m, y_g_m, t_g_m, v_g_m]).T
flattened_output = np.hstack([accel_m, deltv_m])
print('Data processing done')

# model parameters
in_features = flattened_input.shape[1]
out_features = flattened_output.shape[1]
num_regions = num_split_v_car * num_split_x_g * num_split_y_g * num_split_t_g * num_split_v_g
lower_bounds = [v_c_bounds[:-1], x_g_bounds[:-1], [-3.6, -0.2], [-3.2, -0.2], v_g_bounds[:-1]]
upper_bounds = [v_c_bounds[1:] , x_g_bounds[1:] , [ 0.2,  3.6], [ 0.2,  3.2], v_g_bounds[1:] ]
activation_idx = [0, 1, 2, 3, 4]
delta = [10.0, 15.0, 15.0, 100.0, 10.0]
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
    # def loss_fn(params):
    #     y_pred = wcrbf.apply(params, x)
    #     loss = optax.l2_loss(predictions=y_pred, targets=y).mean()
    #     return loss
    def loss_fn(params):
        DT = 0.1
        WB = 0.33
        MAX_SPEED = 7.0
        MIN_SPEED = 0.0
        MAX_STEER = 0.4189

        y_predictions = wcrbf.apply(params, x)

        batch_size = x.shape[0]
        x_ =    jnp.zeros(batch_size)
        y_ =    jnp.zeros(batch_size)
        delta = jnp.zeros(batch_size)
        v =     jnp.clip(x[:, 0], a_min=MIN_SPEED, a_max=MAX_SPEED)
        yaw =   jnp.zeros(batch_size)

        x_actual =     x_
        y_actual =     y_
        delta_actual = delta
        v_actual =     v
        yaw_actual =   yaw
        first_states_actual = None
        final_states_actual = None
        for i in range(5):
            a = y[:, i]
            delta_v = y[:, i+5]
            x_actual = x_actual + v_actual * jnp.cos(yaw_actual) * DT
            y_actual = y_actual + v_actual * jnp.sin(yaw_actual) * DT
            delta_actual = delta_actual + delta_v * DT
            delta_actual = jnp.clip(delta_actual, a_min=-MAX_STEER, a_max=MAX_STEER)
            v_actual = v_actual + a * DT
            v_actual = jnp.clip(v_actual, a_min=MIN_SPEED, a_max=MAX_SPEED)
            yaw_actual = yaw_actual + (v_actual / WB) * jnp.tan(delta_actual) * DT
            if i == 0:
                first_states_actual = jnp.vstack([
                    x_actual, y_actual, delta_actual, v_actual, yaw_actual
                ]).T
            if i == 4:
                final_states_actual = jnp.vstack([
                    x_actual, y_actual, delta_actual, v_actual, yaw_actual
                ]).T

        x_pred =     x_
        y_pred =     y_
        delta_pred = delta
        v_pred =     v
        yaw_pred =   yaw
        first_states_pred = None
        final_states_pred = None
        for i in range(5):
            a = y_predictions[:, i]
            delta_v = y_predictions[:, i+5]
            x_pred = x_pred + v_pred * jnp.cos(yaw_pred) * DT
            y_pred = y_pred + v_pred * jnp.sin(yaw_pred) * DT
            delta_pred = delta_pred + delta_v * DT
            delta_pred = jnp.clip(delta_pred, a_min=-MAX_STEER, a_max=MAX_STEER)
            v_pred = v_pred + a * DT
            v_pred = jnp.clip(v_pred, a_min=MIN_SPEED, a_max=MAX_SPEED)
            yaw_pred = yaw_pred + (v_pred / WB) * jnp.tan(delta_pred) * DT
            if i == 0:
                first_states_pred = jnp.vstack([
                    x_pred, y_pred, delta_pred, v_pred, yaw_pred
                ]).T
            if i == 4:
                final_states_pred = jnp.vstack([
                    x_pred, y_pred, delta_pred, v_pred, yaw_pred
                ]).T

        loss = (
            optax.l2_loss(predictions=first_states_pred, targets=first_states_actual).mean() +
            optax.l2_loss(predictions=final_states_pred, targets=final_states_actual).mean()
        )
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss_, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_


# training parameters
num_points = flattened_output.shape[0]
epochs = 1000

train_work_dir = "./runs/nmpc_4_regions"
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


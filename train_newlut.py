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
from datetime import datetime

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.tensorboard import SummaryWriter
from flax.training import train_state
from tqdm import tqdm

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

import yaml

from flax.core import freeze, unfreeze

# tensorboard logging
lr = 0.001
batch_size = 2000
numk = 100

# number of regions: 20x40x20=16000
num_splitx = 11
num_splity = 10
num_splitt = 8
num_overlap = 1

# loading raw data
data = np.load("lut_allkappa.npz")
lut = jnp.array(data["lut"])
xlut = data["xlut"]
ylut = data["ylut"]
tlut = data["tlut"]

# stride tricks
ast = np.lib.stride_tricks.as_strided
# strides should be the same for all dimension indices
# stride on x
lut_strides = xlut.strides
num_points_in_splitx = int(xlut.shape[0] / num_splitx + 1)
xlut_strided = ast(
    xlut,
    shape=(num_splitx, num_points_in_splitx),
    strides=(lut_strides[0] * (num_points_in_splitx - num_overlap), lut_strides[0]),
).flatten()
# lowers and uppers on x
ux, cx = np.unique(xlut_strided, return_counts=True)
lowers_x = [xlut_strided[0], *(ux[cx > 1])]
uppers_x = [*(ux[cx > 1]), xlut_strided[-1]]

# stride on y
num_points_in_splity = int(ylut.shape[0] / num_splity + 1)
ylut_strided = ast(
    ylut,
    shape=(num_splity, num_points_in_splity),
    strides=(lut_strides[0] * (num_points_in_splity - num_overlap), lut_strides[0]),
).flatten()
# lowers and uppers on y
uy, cy = np.unique(ylut_strided, return_counts=True)
lowers_y = [ylut_strided[0], *(uy[cy > 1])]
uppers_y = [*(uy[cy > 1]), ylut_strided[-1]]

# stride on theta
num_points_in_splitt = int(tlut.shape[0] / num_splitt + 1)
tlut_strided = ast(
    tlut,
    shape=(num_splitt, num_points_in_splitt),
    strides=(lut_strides[0] * (num_points_in_splitt - num_overlap), lut_strides[0]),
).flatten()
# lowers and uppers on y
ut, ct = np.unique(tlut_strided, return_counts=True)
lowers_t = [tlut_strided[0], *(ut[ct > 1])]
uppers_t = [*(ut[ct > 1]), tlut_strided[-1]]


xlutm, ylutm, tlutm = jnp.meshgrid(xlut, ylut, tlut, indexing="ij")
idxlut = jnp.stack((xlutm, ylutm, tlutm), axis=-1)

# flattened into (N, in_features) and (N, out_features)
# in_features: (x, y, theta)
# out_features: (k0, k1, k2, k3, s)
flattened_lut = lut.reshape((-1, 5))
flattened_idxlut = idxlut.reshape((-1, 3))

# model params
# in_features: (x, y, theta)
# out_features: (k0, k1, k2, k3, s)
in_features = 3
out_features = 5
# num_kernels = 20
num_regions = num_splitx * num_splity * num_splitt
lower_bounds = [lowers_x, lowers_y, lowers_t]
uppers_bounds = [uppers_x, uppers_y, uppers_t]
activation_idx = [0, 1, 2]
delta = [15.0, 15.0, 100.0]
basis_func = inverse_quadratic
seed = 0
bound_ranges = [np.arange(len(curr_bounds)) for curr_bounds in lower_bounds]
dimension_ranges = (
    np.stack(np.meshgrid(*bound_ranges, indexing="ij"), axis=-1)
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
    basis_func=basis_func,
    num_regions=num_regions,
    lower_bounds=lower_bounds,
    upper_bounds=uppers_bounds,
    dimension_ranges=dimension_ranges,
    activation_idx=activation_idx,
    delta=delta,
)
params = wcrbf.init(init_rng, jnp.ones((batch_size, 3)))
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
        # loss = optax.huber_loss(predictions=y_pred, targets=y).mean()
        # loss = jnp.mean(jnp.abs(y_pred - y))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss_, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_


# training params
num_points = flattened_lut.shape[0]
epochs = 2000
# batch_size = 50000

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
train_work_dir = "./runs/{}_bs{}_lr{}_nk{}_{}x{}y{}t_{}_kernel_l2_allkappalut".format(
    ts, batch_size, lr, numk, num_splitx, num_splity, num_splitt, basis_func.__name__
)
writer = SummaryWriter(train_work_dir)

# config logging
yaml_dir = "./configs/" + train_work_dir[7:] + ".yaml"

# config logging
config_dict = {
    "in_features": in_features,
    "out_features": out_features,
    "num_kernels": numk,
    "basis_func": basis_func.__name__,
    "num_splitx": num_splitx,
    "num_splity": num_splity,
    "num_splitt": num_splitt,
    "num_regions": num_regions,
    "lower_bounds": [[float(l) for l in ll] for ll in lower_bounds],
    "upper_bounds": [[float(u) for u in uu] for uu in uppers_bounds],
    "dimension_ranges": dimension_ranges,
    "activation_idx": activation_idx,
    "delta": delta,
    "epochs": epochs,
    "lr": lr,
    "batch_size": batch_size,
    "seed": seed,
    # "params_shape": params_shape,
}
with open(yaml_dir, "w+") as outfile:
    yaml.dump(config_dict, outfile, default_flow_style=False)


def train_epoch(train_state, train_x, train_y, bs, epoch, epoch_rng, summary_writer):
    # batching data
    num_train = train_x.shape[0]
    num_steps = num_train // bs
    # random permutations
    perms = jax.random.permutation(epoch_rng, num_train)
    perms = perms[: num_steps * bs]
    perms = perms.reshape((num_steps, bs))
    batch_losses = []
    for b, perm in enumerate(perms):
        batch_x = train_x[perm, :]
        batch_y = train_y[perm, :]
        train_state, batch_loss = train_step(train_state, batch_x, batch_y)
        batch_losses.append(batch_loss)
        # print(
        #     "Training Epoch: %d, batch: %d, batch loss: %.4f"
        #     % (epoch, b, jax.device_get(batch_loss))
        # )
        summary_writer.add_scalar(
            "train_loss_batch", jax.device_get(batch_loss), b + (epoch * len(perms))
        )
    batch_losses_np = jax.device_get(batch_losses)
    # print("Training Epoch: %d, training loss: %.4f" % (epoch, np.mean(batch_losses_np)))

    summary_writer.add_scalar("train_loss", np.mean(batch_losses_np), epoch)
    return train_state


# training
for e in tqdm(range(epochs)):
    rng, perm_rng = jax.random.split(rng)
    state = train_epoch(
        state, flattened_idxlut, flattened_lut, batch_size, e, perm_rng, writer
    )

from flax.training import checkpoints

CKPT_DIR = "ckpts/" + train_work_dir[7:]
import os

if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)
checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=0)

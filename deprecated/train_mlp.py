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

import wandb

from flax.training import train_state
from tqdm import tqdm

from utils import integrate_path_mult

import yaml

from flax.core import freeze, unfreeze

jax.config.update("jax_enable_x64", True)


class MLP(nn.Module):
    # num parameters 581
    @nn.compact
    def __call__(self, x):
        # input shape 3
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=5)(x)
        # output shape 5
        return x


epochs = 4000

# tensorboard logging
lr = 0.001
batch_size = 20000

# loading raw data
data = np.load("lut_allkappa.npz")
lut = jnp.array(data["lut"])
xlut = data["xlut"]
ylut = data["ylut"]
tlut = data["tlut"]

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

seed = 0

# rng
rng = jax.random.PRNGKey(seed)
rng, init_rng = jax.random.split(rng)

# model init
mlp = MLP()

params = mlp.init(init_rng, jnp.ones((batch_size, 3)))
params_shape = jax.tree_util.tree_map(jnp.shape, unfreeze(params))
print("Initialized parameter shapes:\n", params_shape)

# optimizer
optim = optax.adam(lr)

# train state
state = train_state.TrainState.create(apply_fn=mlp.apply, params=params, tx=optim)


# train one step
@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        y_pred = mlp.apply(params, x)
        # loss on polynomial parameters
        # loss = optax.l2_loss(predictions=y_pred, targets=y).mean()
        # loss = optax.huber_loss(predictions=y_pred, targets=y).mean()
        # loss = optax.huber_loss(predictions=y_pred, targets=y)
        loss = jnp.mean(jnp.abs(y_pred - y))
        # loss on actual endpoint
        all_states = integrate_path_mult(y_pred)
        end_states = all_states[:, -1, :3]
        # end_loss = optax.huber_loss(predictions=end_states, targets=x).mean()
        # end_loss = jnp.mean(jnp.abs(end_states - x))
        end_loss = optax.l2_loss(predictions=end_states, targets=x).mean()
        return loss + end_loss
        # return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss_, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_


# training params
num_points = flattened_lut.shape[0]
# batch_size = 50000

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
train_work_dir = "./runs/{}_bs{}_lr{}_mlp_l1+end100stepl2_allkappalut_doubleprecision".format(
    ts, batch_size, lr,
)

# config logging
yaml_dir = "./configs/" + train_work_dir[7:] + ".yaml"

# config logging
config_dict = {
    "in_features": in_features,
    "out_features": out_features,
    "num_kernels": None,
    "basis_func": "MLP",
    "num_splitx": None,
    "num_splity": None,
    "num_splitt": None,
    "num_regions": None,
    "lower_bounds": None,
    "upper_bounds": None,
    "dimension_ranges": None,
    "activation_idx": None,
    "delta": None,
    "epochs": epochs,
    "lr": lr,
    "batch_size": batch_size,
    "seed": seed,
    # "params_shape": params_shape,
}
with open(yaml_dir, "w+") as outfile:
    yaml.dump(config_dict, outfile, default_flow_style=False)

wandb.init(
    project="irbfn",
    config=config_dict,
    notes="trained on actual car size LUT, L1 loss, endpoint loss, double precision, MLP baseline",
)


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

        wandb.log({"train_loss_batch": jax.device_get(batch_loss)})
    batch_losses_np = jax.device_get(batch_losses)

    wandb.log({"train_loss": np.mean(batch_losses_np)})
    return train_state


# training
for e in tqdm(range(epochs)):
    rng, perm_rng = jax.random.split(rng)
    state = train_epoch(
        state, flattened_idxlut, flattened_lut, batch_size, e, perm_rng, None
    )

from flax.training import checkpoints
import flax

flax.config.update("flax_use_orbax_checkpointing", True)

CKPT_DIR = "ckpts/" + train_work_dir[7:]
import os

if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)
checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=0)

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

# %%
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
from utils import integrate_path_mult, N

# %%

config_f = "configs/default.yaml"
ckpt = "ckpts/checkpoint_0"
with open(config_f, "r") as f:
    config_dict = yaml.safe_load(f)
conf = argparse.Namespace(**config_dict)

# uncomment the following line (53) and line 206 to create memory profile, will slow down inference significantly
# see https://jax.readthedocs.io/en/latest/profiling.html#programmatic-capture for how to inspect
# jax.profiler.start_trace('./tensorboard_profiler')


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
params = wcrbf.init(init_rng, jnp.ones((1, 4)))
optim = optax.adam(conf.lr)
state = train_state.TrainState.create(apply_fn=wcrbf.apply, params=params, tx=optim)
# empty_state = train_state.TrainState.create(apply_fn=wcrbf.apply, params=jax.tree_map(np.zeros_like, params['params']), tx=optim)
restored_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt, target=state)
# %%

# plotting points
# test_x = jnp.linspace(6, 6, 20)
test_x = jnp.array([5.0])
test_y = jnp.linspace(-4.0, 4.0, 20)
test_t = jnp.linspace(-0.3, 0.3, 10)

# stacking testing points, final matrix should be shape (batch_size, 4)
xm, ym, tm = jnp.meshgrid(test_x, test_y, test_t, indexing="ij")
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


# %%
y_pred_raw = pred_step(restored_state, test)
y_pred = jnp.zeros((test.shape[0], 5))
y_pred = y_pred.at[:, -1].set(y_pred_raw[:, 0])
y_pred = y_pred.at[:, 1:3].set(y_pred_raw[:, 1:])

all_states = integrate_path_mult(y_pred)

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams["figure.figsize"] = [15, 10]
plt.rcParams["figure.dpi"] = 150
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 30

col = mpl.colormaps["tab20"]

fig, ax = plt.subplots()
for p_i in range(all_states.shape[0]):
    ax.plot(
        all_states[p_i, :, 1],
        all_states[p_i, :, 0],
        color=col((p_i / all_states.shape[0])),
        linewidth=2.0,
    )
ax.scatter(
    test_y,
    np.repeat(test_x, test_y.shape[0]),
    marker="o",
    s=500.0,
    facecolors="none",
    edgecolors="tab:blue",
    linewidth=3,
    label="Goal Points",
)
ax.set_aspect("equal", "box")
ax.xaxis.set_tick_params(width=5, length=10)
ax.yaxis.set_tick_params(width=5, length=10)
plt.xlabel("Y", labelpad=-2)
plt.ylabel("X", labelpad=-10)
plt.legend(loc="lower left")
fig.tight_layout()
plt.savefig("traj_out.pdf", format="pdf", bbox_inches="tight")
plt.savefig("traj_out.png", format="png", bbox_inches="tight")
# %%

import chex

chex.clear_trace_counter()
# profile: time of random evals

# profiling points
test_x = jnp.linspace(2, 6, 10)
test_y = jnp.linspace(-4.0, 4.0, 10)
test_t = jnp.linspace(-0.3, 0.3, 5)
xm, ym, tm = jnp.meshgrid(test_x, test_y, test_t, indexing="ij")
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

import time

num_eval = 1000
print("-----------------------------")
print(
    "Profiling "
    + str(num_eval)
    + " evaluation of generating "
    + str(test.shape[0])
    + " Random Trajectories with IRBFN."
)
noise = jax.random.normal(rng, (num_eval, *test.shape))

start = time.time()
for ei in range(num_eval):
    y_pred_raw = pred_step(restored_state, test + noise[ei])
    y_pred = jnp.zeros((test.shape[0], 5))
    y_pred = y_pred.at[:, -1].set(y_pred_raw[:, 0])
    y_pred = y_pred.at[:, 1:3].set(y_pred_raw[:, 1:])

    all_states = integrate_path_mult(y_pred)

# jax.profiler.stop_trace()
end = time.time()
print("Total elapsed:", end - start)
print("Elapsed per eval:", str((end - start) / num_eval))
print("TrajGen frequency:", str(num_eval / (end - start)), "Hz")

# %%

# Profiling generation through optimization online
# https://pyclothoids.readthedocs.io/en/latest/
from pyclothoids import Clothoid


def sample_traj(clothoid):
    traj = np.empty((N, 4))
    k0 = clothoid.Parameters[3]
    dk = clothoid.Parameters[4]

    for i in range(N):
        s = i * (clothoid.length / max(N - 1, 1))
        traj[i, 0] = clothoid.X(s)
        traj[i, 1] = clothoid.Y(s)
        traj[i, 2] = clothoid.Theta(s)
        traj[i, 3] = np.sqrt(clothoid.XDD(s) ** 2 + clothoid.YDD(s) ** 2)
    return traj


# same evaluation goals
test_np = np.array(test)
num_eval = 100
noise_np = np.array(noise)

print("-----------------------------")
print(
    "Profiling "
    + str(num_eval)
    + " evaluation of generating "
    + str(test_np.shape[0])
    + " Random Trajectories with Online Optimization."
)
# profile once
start = time.time()
for i in range(num_eval):
    all_traj = []
    test_np_noised = test_np + noise_np[i]
    for p in test_np_noised:
        clothoid = Clothoid.G1Hermite(0, 0, 0, p[0], p[1], p[2])
        traj = sample_traj(clothoid)
        all_traj.append(traj)
    all_traj_np = np.array(all_traj)

end = time.time()
print("Total elapsed:", end - start)
print("Elapsed per eval:", str((end - start) / num_eval))
print("TrajGen frequency:", str(num_eval / (end - start)), "Hz")
print("-----------------------------")
# %%
# error calculation
print("Evaluating average error across region:")
print("x: ", str(test_x.min()), "meters to", str(test_x.max()), "meters")
print("y: ", str(test_y.min()), "meters to", str(test_y.max()), "meters")
print("theta: ", str(test_t.min()), "radians to", str(test_t.max()), "radians")
y_pred_raw = pred_step(restored_state, test)
y_pred = jnp.zeros((test.shape[0], 5))
y_pred = y_pred.at[:, -1].set(y_pred_raw[:, 0])
y_pred = y_pred.at[:, 1:3].set(y_pred_raw[:, 1:])
all_states = integrate_path_mult(y_pred)

error = test[:, :3] - np.array(all_states)[:, -1, :3]
avg_err = np.sum(np.abs(error), axis=0) / test.shape[0]

print("-----------------------------")
print("Experimental Errors")
print("Trajectory Endpoint Error on x:", str(avg_err[0]), "meters")
print("Trajectory Endpoint Error on y:", str(avg_err[1]), "meters")
print("Trajectory Endpoint Error on theta:", str(avg_err[2]), "radians")
print("-----------------------------")
print("Theoretical Errors")
training_spacing = 0.1
arc_lengths = np.array(y_pred[:, -1])
infty_norm_func = arc_lengths.max()
integration_spacing = infty_norm_func / N
N_training_points = 3213142
alpha = 1
L = 1000
param_err_the = (1 / (N_training_points ^ alpha)) * (
    L * 2 ** (alpha / 2 + 1) * training_spacing**alpha
    + 2 ** (alpha / 2) * training_spacing**alpha * infty_norm_func
    + infty_norm_func
)

theory_end_point_err_x = 0
theory_end_point_err_y = 0
theory_end_point_err_theta = 0

for i_te in range(N):
    dtheta = param_err_the * (
        integration_spacing
        + integration_spacing**2
        + integration_spacing**3
        + integration_spacing**4
    )
    theory_end_point_err_theta += dtheta
    dx = np.cos(np.pi/2 - theory_end_point_err_theta)
    dy = np.sin(theory_end_point_err_theta)
    theory_end_point_err_x += dx
    theory_end_point_err_y += dy

print(
    "Theoretical Trajectory Endpoint Error on x:", str(theory_end_point_err_x), "meters"
)
print(
    "Theoretical Trajectory Endpoint Error on y:", str(theory_end_point_err_y), "meters"
)
print(
    "Theoretical Trajectory Endpoint Error on theta:",
    str(theory_end_point_err_theta),
    "radians",
)
print("-----------------------------")
# %%

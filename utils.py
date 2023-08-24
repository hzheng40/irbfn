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
# Last Modified: 10/12/2022
# Trajectory Generator Utilities implemented in JAX
#%%
import jax
import jax.numpy as jnp
import numpy as np
import chex
from functools import partial

N = 100

PARAM_MAT = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [-11.0 / 2, 9.0, -9.0 / 2, 1.0],
        [9.0, -45.0 / 2, 18.0, -9.0 / 2],
        [-9.0 / 2, 27.0 / 2, -27.0 / 2, 9.0 / 2],
    ]
)

@jax.jit
def params_to_coefs(params):
    s = params[-1]
    s2 = s**2
    s3 = s**3
    coefs = jnp.matmul(PARAM_MAT, params[:-1])
    coefs = coefs.at[1].divide(s)
    coefs = coefs.at[2].divide(s2)
    coefs = coefs.at[3].divide(s3)
    return coefs

@jax.jit
def get_curvature_theta(coefs, s_cur):
    out = 0.0
    out2 = 0.0
    for i in range(coefs.shape[0]):
        temp = coefs[i] * s_cur**i
        out += temp
        out2 += temp * s_cur / (i + 1)

    return out, out2

@jax.jit
@chex.assert_max_traces(n=1)
def integrate_one_step(state_init, seq, coefs):
    kappa_k, theta_k = get_curvature_theta(coefs, seq[0])
    dx = state_init[4] * (1 - 1 / seq[1]) + (jnp.cos(theta_k) + jnp.cos(state_init[2])) / 2 / seq[1]
    dy = state_init[5] * (1 - 1 / seq[1]) + (jnp.sin(theta_k) + jnp.sin(state_init[2])) / 2 / seq[1]
    x = seq[0] * dx
    y = seq[0] * dy
    state_new = jnp.stack([x, y, theta_k, kappa_k, dx, dy])
    return state_new, state_new

@jax.jit
@partial(jax.vmap, in_axes=(0, ))
@chex.assert_max_traces(n=1)
def integrate_path_mult(params):
    coefs = params_to_coefs(params)
    state_init = jnp.zeros(6,)
    state_init = state_init.at[3].set(coefs[0])
    sk_seq = jnp.linspace(0., params[-1], num=N)
    k_seq = jnp.arange(1, N+1)
    seq = jnp.vstack((sk_seq, k_seq)).T
    last_state, all_states = jax.lax.scan(partial(integrate_one_step, coefs=coefs), state_init, seq)
    return all_states

@jax.jit
@chex.assert_max_traces(n=1)
def integrate_path(params):
    coefs = params_to_coefs(params)
    states = jnp.empty((N, 4))
    states = states.at[0].set(jnp.zeros(4,))
    states = states.at[0, 3].set(coefs[0])
    dx = 0.
    dy = 0.
    x = 0.
    y = 0.
    ds = params[-1] / N
    theta_old = 0.
    for k in range(1, N):
        sk = k * ds
        kappa_k, theta_k = get_curvature_theta(coefs, sk)
        dx = dx * (1 - 1 / k) + (jnp.cos(theta_k) + jnp.cos(theta_old)) / 2 / k
        dy = dy * (1 - 1 / k) + (jnp.sin(theta_k) + jnp.sin(theta_old)) / 2 / k
        x = sk * dx
        y = sk * dy
        states = states.at[k].set(jnp.stack([x, y, theta_k, kappa_k]))
        theta_old = theta_k
    return states

@jax.jit
@chex.assert_max_traces(n=1)
def nearest_point_on_trajectory(point, trajectory):
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = jnp.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = jnp.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = jnp.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = jnp.sqrt(jnp.sum(temp * temp))
    min_dist_segment = jnp.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )
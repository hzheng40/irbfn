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
# Based on clean-rl's implementation of PPO here:
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
# and: https://github.com/MyNameIsArko/RL-Flax/blob/main/PPO.py
# Last Modified: 07/25/2023
# Basic RL algorithms in JAX

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import os
import random
from datetime import datetime
from distutils.util import strtobool
from functools import partial
from typing import Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from distrax import Categorical
from flax import struct
from flax.core import FrozenDict
from flax.training import train_state
from jax.lax import stop_gradient
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def parse_args():
    # experiments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py")
    )
    parser.add_argument("--seed", type=int, default=12345)

    # algorithm specific
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--lr", type=float, default=2.5e-4, help="the learning rate of the optimizer"
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=4, help="the number of mini-batches"
    )
    parser.add_argument(
        "--update-epochs", type=int, default=4, help="the K epochs to update the policy"
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="coefficient of the value function"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="the target KL divergence threshold",
    )
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def linear_layer_init(features, std=np.sqrt(2), bias_const=0.0):
    layer = nn.Dense(
        features=features,
        kernel_init=nn.initializers.orthogonal(std),
        bias_init=nn.initializers.constant(bias_const),
    )
    return layer


class Actor(nn.Module):
    """
    PPO Actor
    """

    hidden_size: int
    act_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                linear_layer_init(self.hidden_size),
                nn.tanh,
                linear_layer_init(self.hidden_size),
                nn.tanh,
                linear_layer_init(self.act_dim, std=0.01),
            ]
        )(x)


class Critic(nn.Module):
    """
    PPO Critic
    """

    hidden_size: int

    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                linear_layer_init(self.hidden_size),
                nn.tanh,
                linear_layer_init(self.hidden_size),
                nn.tanh,
                linear_layer_init(1, std=1.0),
            ]
        )(x)


class AgentState(train_state.TrainState):
    # makes TrainState work in jitted functions?
    actor_fn: Callable = struct.field(pytree_node=False)
    critic_fn: Callable = struct.field(pytree_node=False)


@struct.dataclass
class AgentParams:
    actor_params: FrozenDict
    critic_params: FrozenDict


@struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@jax.jit
def get_action_and_value(
    agent_state: AgentState, params: AgentParams, x: np.ndarray, action: np.ndarray
):
    logits = agent_state.actor_fn(params.actor_params, x)
    value = agent_state.critic_fn(params.critic_params, x)
    probs = Categorical(logits=logits)
    return probs.log_prob(action), probs.entropy(), value.squeeze()


@jax.jit
def get_action_and_value_and_sample(
    agent_state: AgentState,
    next_obs: np.ndarray,
    next_done: np.ndarray,
    storage: Storage,
    step: int,
    key: jax.random.PRNGKey,
):
    logits = agent_state.actor_fn(agent_state.params.actor_params, next_obs)
    value = agent_state.critic_fn(agent_state.params.critic_params, next_obs)
    probs = Categorical(logits=logits)
    key, subkey = jax.random.split(key)
    action = probs.sample(seed=subkey)
    logprob = probs.log_prob(action)
    storage = storage.replace(
        obs=storage.obs.at[step].set(next_obs),
        dones=storage.dones.at[step].set(next_done),
        actions=storage.actions.at[step].set(action),
        logprobs=storage.logprobs.at[step].set(logprob),
        values=storage.values.at[step].set(value.squeeze()),
    )
    return storage, action, key


@jax.jit
def anneal_schedule(step):
    """
    Anneals learning rate
    """
    frac = 1.0 - (update - 1.0) / num_updates
    lrnow = frac * args.lr


@partial(
    jax.jit,
    static_argnums=[
        4,
    ],
)
@chex.assert_max_traces(n=2)
def calc_gae(
    agent_state: AgentState,
    next_obs: np.ndarray,
    next_done: np.ndarray,
    storage: Storage,
    args: argparse.Namespace,
):
    storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
    next_value = agent_state.critic_fn(
        agent_state.params.critic_params, next_obs
    ).squeeze()
    lastgaelam = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - storage.dones[t + 1]
            nextvalues = storage.values[t + 1]
        delta = (
            storage.rewards[t]
            + args.gamma * nextvalues * nextnonterminal
            - storage.values[t]
        )
        lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))
    storage = storage.replace(returns=storage.advantages + storage.values)
    return storage


@partial(
    jax.jit,
    static_argnums=[
        8,
    ],
)
@chex.assert_max_traces(n=2)
def calc_ppo_loss(
    agent_state: AgentState,
    params: AgentParams,
    obs: np.ndarray,
    act: np.ndarray,
    logp: np.ndarray,
    adv: np.ndarray,
    ret: np.ndarray,
    val: np.ndarray,
    args: argparse.Namespace,
):
    newlogprob, entropy, newvalue = get_action_and_value_and_sample(
        agent_state, params, obs, act
    )
    logratio = newlogprob - logp
    ratio = jnp.exp(logratio)

    approx_kl = ((ratio - 1) - logratio).mean()

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    v_loss_unclipped = (newvalue - ret) ** 2
    v_clipped = val + jnp.clip(newvalue - val, -args.clip_coef, args.clip_coef)
    v_loss_clipped = (v_clipped - ret) ** 2
    v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
    v_loss = 0.5 * v_loss_max.mean()

    entropy_loss = entropy.mean()

    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
    return loss, (pg_loss, v_loss, entropy_loss, stop_gradient(approx_kl))


def update_ppo(agent_state: AgentState, storage: Storage, key: jax.random.PRNGKeyArray):
    pass  # TODO


if __name__ == "__main__":
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rl_{args.exp_name}_{args.seed}_{ts}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    # agent

    # start
    global_step = 0
    num_updates = args.global_timesteps // args.batch_size

    # annealing schedule and optimizer
    if args.anneal_lr:
        lr_max = args.learning_rate
        lr_min = (1.0 - (num_updates - 1.0) / num_updates) * args.learning_rate
        lr_schedule = optax.linear_schedule(
            init_value=lr_max, end_value=lr_min, transition_steps=num_updates
        )
        optimizer = optax.adam(lr_schedule, eps=1e-5)
    else:
        optimizer = optax.adam(args.lr, eps=1e-5)

    # main loop
    for update in range(1, num_updates + 1):
        for step in range(0, args.num_steps):
            pass

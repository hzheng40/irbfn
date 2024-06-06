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
from tqdm import tqdm
import time
from goal_mpc_node import solve_mpc


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

flax.config.update('flax_use_orbax_checkpointing', False)

config_f = "configs/goal_mpc_normalized_1_region_l1.yaml"
ckpt = "ckpts/goal_mpc_normalized_1_region_l1/checkpoint_0"
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

irbfn_input = np.array([0.0, 1.0, 0.0, 0.0, 1.0]).reshape(1, 5)
solution = solve_mpc(*irbfn_input.flatten())[-2:]
print('Prediction running...')
irbfn_output = pred_step(restored_state, irbfn_input)
print('Prediction done')

print('Input:', irbfn_input)
print('Actual Solution:', solution)
print('Output:', irbfn_output)
print('Error:', np.abs(solution - irbfn_output))
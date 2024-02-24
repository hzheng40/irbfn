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


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

flax.config.update('flax_use_orbax_checkpointing', False)

config_f = "configs/goal_mpc_4_region_l1_split_y_t.yaml"
ckpt = "ckpts/goal_mpc_4_region_l1_split_y_t/checkpoint_0"
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


if __name__ == "__main__":
    print('Loading data...')
    table = np.load('goal_mpc_lookup_table_tiny_2.npz')['table']
    v_car = table[:, 0].flatten()
    x_goal = table[:, 1].flatten()
    y_goal = table[:, 2].flatten()
    t_goal = table[:, 3].flatten()
    v_goal = table[:, 4].flatten()
    speed = table[:, 5].flatten()
    steer = table[:, 6].flatten()
    print('Data import completed')

    print('Mirroring data...')
    v_car_m = np.concatenate((v_car, v_car))
    x_goal_m = np.concatenate((x_goal, x_goal))
    y_goal_m = np.concatenate((y_goal, -y_goal))
    t_goal_m = np.concatenate((t_goal, -t_goal))
    v_goal_m = np.concatenate((v_goal, v_goal))
    speed_m = np.concatenate((speed, speed))
    steer_m = np.concatenate((steer, -steer))
    print('Mirroring completed')

    print('Assembling table...')
    full_table = np.vstack((v_car_m, x_goal_m, y_goal_m, t_goal_m, v_goal_m, speed_m, steer_m)).T
    print('Table assembled:', len(full_table), 'rows')

    print('Sampling full table...')
    random_sample = np.random.default_rng().choice(full_table, size=len(full_table) // 2, replace=False, axis=0)
    print('Sampling completed')

    print('Number of Samples:', len(random_sample))
    print('Table Shape:', random_sample.shape)

    print('Shifting table...')
    random_sample[:, 0] += 0.25
    random_sample[:, 1] += 0.05
    random_sample[:, 2] += 0.05
    random_sample[:, 3] += 0.05
    random_sample[:, 4] += 0.25
    print('Table shift completed')

    print('Prediction running...')
    pred_u = pred_step(restored_state, random_sample[:, :5])
    print('Prediction completed')

    print('Calculating error...')
    results = np.concatenate((random_sample, pred_u, np.abs(random_sample[:, -2:] - pred_u)), axis=1)
    print('Error calculations completed')
    print('Final Table Shape:', results.shape)

    np.savez(f'goal_mpc_4_region_l1_split_y_t_error.npz', table=results)
    print('Data saved')
    print('Max Velocity Error:', np.max(results[:, -2]))
    print('Max Steering Angle Error:', np.max(results[:, -1]))


import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax.training import train_state, checkpoints
import matplotlib.pyplot as plt
from flax_rbf.flax_rbf import *
from irbfn_mpc.model import WCRBFNet
from tqdm import tqdm
import time
from irbfn_mpc.arg_utils import irbfn_dnmpc_eval_args

# pred one step
@jax.jit
def pred_step(state, x):
    y = state.apply_fn(state.params, x)
    return y


def main(args: argparse.Namespace):
    flax.config.update('flax_use_orbax_checkpointing', False)

    with open(args.config_f, "r") as f:
        config_dict = yaml.safe_load(f)
    conf = argparse.Namespace(**config_dict)

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
    optim = optax.chain(optax.clip_by_global_norm(0.1), optax.adam(conf.lr))
    state = train_state.TrainState.create(apply_fn=wcrbf.apply, params=params, tx=optim)
    restored_state = checkpoints.restore_checkpoint(ckpt_dir=args.ckpt, target=state)

    print('Loading data...')
    data = np.load('data/test_nmpc_lookup_table.npz')
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

    flattened_input = np.vstack([v_c_m, x_g_m, y_g_m, t_g_m, v_g_m]).T
    flattened_output = np.hstack([accel_m, deltv_m])

    # predictions
    # inputs = [arr.reshape(1, 75 for arr in flattened_input[:1000]]
    # start = time.time()
    # st = time.process_time()
    # for inp in inputs:
    #     pred_step(restored_state, inp)
    # end = time.time()
    # et = time.process_time()
    # print('Execution time:', end - start, 'seconds')
    # print('CPU Execution time:', et - st, 'seconds')
    print('Prediction running...')
    pred_u = pred_step(restored_state, flattened_input)
    print('Prediction done')

    print('Propagating controls...')
    DT = 0.1
    WB = 0.33
    MAX_SPEED = 7.0
    MIN_SPEED = 0.0
    MAX_STEER = 0.4189

    batch_size = flattened_input.shape[0]
    x_ =    np.zeros(batch_size)
    y_ =    np.zeros(batch_size)
    delta = np.zeros(batch_size)
    v =     np.clip(flattened_input[:, 0], a_min=MIN_SPEED, a_max=MAX_SPEED)
    yaw =   np.zeros(batch_size)

    x_actual =     x_
    y_actual =     y_
    delta_actual = delta
    v_actual =     v
    yaw_actual =   yaw
    first_states_actual = None
    final_states_actual = None
    for i in tqdm(range(5)):
        a = flattened_output[:, i]
        delta_v = flattened_output[:, i+5]
        x_actual = x_actual + v_actual * np.cos(yaw_actual) * DT
        y_actual = y_actual + v_actual * np.sin(yaw_actual) * DT
        delta_actual = delta_actual + delta_v * DT
        delta_actual = np.clip(delta_actual, a_min=-MAX_STEER, a_max=MAX_STEER)
        v_actual = v_actual + a * DT
        v_actual = np.clip(v_actual, a_min=MIN_SPEED, a_max=MAX_SPEED)
        yaw_actual = yaw_actual + (v_actual / WB) * np.tan(delta_actual) * DT
        if i == 0:
            first_states_actual = np.vstack([
                x_actual, y_actual, delta_actual, v_actual, yaw_actual
            ]).T
        if i == 4:
            final_states_actual = np.vstack([
                x_actual, y_actual, delta_actual, v_actual, yaw_actual
            ]).T

    x_pred =     x_
    y_pred =     y_
    delta_pred = delta
    v_pred =     v
    yaw_pred =   yaw
    first_states_pred = None
    final_states_pred = None
    for i in tqdm(range(5)):
        a = pred_u[:, i]
        delta_v = pred_u[:, i+5]
        x_pred = x_pred + v_pred * np.cos(yaw_pred) * DT
        y_pred = y_pred + v_pred * np.sin(yaw_pred) * DT
        delta_pred = delta_pred + delta_v * DT
        delta_pred = np.clip(delta_pred, a_min=-MAX_STEER, a_max=MAX_STEER)
        v_pred = v_pred + a * DT
        v_pred = np.clip(v_pred, a_min=MIN_SPEED, a_max=MAX_SPEED)
        yaw_pred = yaw_pred + (v_pred / WB) * np.tan(delta_pred) * DT
        if i == 0:
            first_states_pred = np.vstack([
                x_pred, y_pred, delta_pred, v_pred, yaw_pred
            ]).T
        if i == 4:
            final_states_pred = np.vstack([
                x_pred, y_pred, delta_pred, v_pred, yaw_pred
            ]).T
    print('States calculated')

    print('First Position MSE:',  np.average(np.linalg.norm(first_states_actual[:, [0, 1]] - first_states_pred[:, [0, 1]], axis=1)))
    print('Final Position MSE:',  np.average(np.linalg.norm(final_states_actual[:, [0, 1]] - final_states_pred[:, [0, 1]], axis=1)))
    print('First State Heading MAE:', np.average(np.abs(first_states_actual[:, 4] - first_states_pred[:, 4])))
    print('Final State Heading MAE:', np.average(np.abs(final_states_actual[:, 4] - final_states_pred[:, 4])))
    print('First State Velocity MAE:', np.average(np.abs(first_states_actual[:, 3] - first_states_pred[:, 3])))
    print('Final State Velocity MAE:', np.average(np.abs(final_states_actual[:, 3] - final_states_pred[:, 3])))

    # accel_ae = np.abs(flattened_output[:, 0] - pred_u[:, 0])
    # deltv_ae = np.abs(flattened_output[:, 1] - pred_u[:, 5])

    # accel_mae = np.average(accel_ae)
    # deltv_mae = np.average(deltv_ae)

    # print('Accel MAE:', accel_mae)
    # print('Deltv MAE:', deltv_mae)

    # print('Accel Median:', np.median(accel_ae))
    # print('Deltv Median:', np.median(deltv_ae))

    # DT = 0.1

    # print('Speed MAE:', accel_mae*DT)
    # print('Steer MAE:', deltv_mae*DT)

    # print('Speed Median:', np.median(accel_ae)*DT)
    # print('Steer Median:', np.median(deltv_ae)*DT)

if __name__ == "__main__":
    args = irbfn_dnmpc_eval_args()
    main(args)
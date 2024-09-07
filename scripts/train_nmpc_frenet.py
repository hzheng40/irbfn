import os
import yaml
from datetime import datetime

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex

# jax.config.update("jax_debug_nans", True)

import wandb
from tqdm import tqdm

import flax
from flax.core import unfreeze
from flax.training import train_state, checkpoints

import matplotlib.pyplot as plt

from flax_rbf.flax_rbf import (
    gaussian,
    gaussian_narrow,
    gaussian_narrower,
    inverse_quadratic,
    inverse_multiquadric,
    quadratic,
    multiquadric,
    gaussian_wide,
    gaussian_wider,
)
from irbfn_mpc.model import WCRBFNet
from irbfn_mpc.arg_utils import dnmpc_train_args
from irbfn_mpc.dynamics import integrate_st_mult, dynamic_st_onestep_aux


def main(args):
    if args.use_float64:
        jax.config.update("jax_enable_x64", True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # if not os.path.exists(args.npz_path + "_cleaned.npz"):
    # loading raw data
    print("Loading data...")
    data = np.load(args.npz_path)
    inputs, outputs = data["inputs"], data["outputs"]
    v_c = inputs[:, 0].flatten()
    x_g = inputs[:, 1].flatten()
    y_g = inputs[:, 2].flatten()
    t_g = inputs[:, 3].flatten()
    v_g = inputs[:, 4].flatten()
    beta = inputs[:, 5].flatten()
    angvz = inputs[:, 6].flatten()
    accel = outputs[:, :, 0]
    deltv = outputs[:, :, 1]
    print("Data import completed")

    if args.mirror_data:
        print("Mirroring data...")
        v_c_m = np.concatenate((v_c, v_c), axis=0)
        x_g_m = np.concatenate((x_g, x_g), axis=0)
        y_g_m = np.concatenate((y_g, -y_g), axis=0)
        t_g_m = np.concatenate((t_g, -t_g), axis=0)
        v_g_m = np.concatenate((v_g, v_g), axis=0)
        beta_m = np.concatenate((beta, beta), axis=0)
        angvz_m = np.concatenate((angvz, angvz), axis=0)
        accel_m = np.concatenate((accel, accel), axis=0)
        deltv_m = np.concatenate((deltv, -deltv), axis=0)
        print("Mirroring completed")
    else:
        v_c_m = v_c
        x_g_m = x_g
        y_g_m = y_g
        t_g_m = t_g
        v_g_m = v_g
        beta_m = beta
        angvz_m = angvz
        accel_m = accel
        deltv_m = deltv

    print("Generating bounds...")

    v_c_bounds = np.sort(np.unique(v_c_m))
    x_g_bounds = np.sort(np.unique(x_g_m))
    y_g_bounds = np.sort(np.unique(y_g_m))
    t_g_bounds = np.sort(np.unique(t_g_m))
    v_g_bounds = np.sort(np.unique(v_g_m))
    beta_bounds = np.sort(np.unique(beta_m))
    angvz_bounds = np.sort(np.unique(angvz_m))

    vc_ind = np.linspace(
        start=0, stop=len(v_c_bounds) - 1, num=args.num_v + 1, endpoint=True, dtype=int
    )
    xg_ind = np.linspace(
        start=0, stop=len(x_g_bounds) - 1, num=args.num_x + 1, endpoint=True, dtype=int
    )
    yg_ind = np.linspace(
        start=0, stop=len(y_g_bounds) - 1, num=args.num_y + 1, endpoint=True, dtype=int
    )
    tg_ind = np.linspace(
        start=0, stop=len(t_g_bounds) - 1, num=args.num_t + 1, endpoint=True, dtype=int
    )
    vg_ind = np.linspace(
        start=0,
        stop=len(v_g_bounds) - 1,
        num=args.num_vgoal + 1,
        endpoint=True,
        dtype=int,
    )
    beta_ind = np.linspace(
        start=0,
        stop=len(beta_bounds) - 1,
        num=args.num_beta + 1,
        endpoint=True,
        dtype=int,
    )
    angvz_ind = np.linspace(
        start=0,
        stop=len(angvz_bounds) - 1,
        num=args.num_angvz + 1,
        endpoint=True,
        dtype=int,
    )

    lower_bounds = [
        list(v_c_bounds[vc_ind[:-1]]),
        list(x_g_bounds[xg_ind[:-1]]),
        list(y_g_bounds[yg_ind[:-1]]),
        list(t_g_bounds[tg_ind[:-1]]),
        list(v_g_bounds[vg_ind[:-1]]),
        # list(d_v_bounds[dv_ind[:-1]]),
        list(beta_bounds[beta_ind[:-1]]),
        list(angvz_bounds[angvz_ind[:-1]]),
    ]
    upper_bounds = [
        list(v_c_bounds[vc_ind[1:]]),
        list(x_g_bounds[xg_ind[1:]]),
        list(y_g_bounds[yg_ind[1:]]),
        list(t_g_bounds[tg_ind[1:]]),
        list(v_g_bounds[vg_ind[1:]]),
        # list(d_v_bounds[dv_ind[1:]]),
        list(beta_bounds[beta_ind[1:]]),
        list(angvz_bounds[angvz_ind[1:]]),
    ]

    print("Bounds defined")

    print("Generating input and output...")
    flattened_input = np.vstack([v_c_m, x_g_m, y_g_m, t_g_m, v_g_m, beta_m, angvz_m]).T
    flattened_output = np.hstack([accel_m, deltv_m])
    if args.only_onestep:
        flattened_output = flattened_output[:, [0, 5]]

    print("Data processing done")
    # bound ranges
    bound_ranges = [np.arange(len(curr_bounds)) for curr_bounds in lower_bounds]
    dimension_ranges = (
        np.stack(np.meshgrid(*bound_ranges, indexing="ij"), axis=-1)
        .reshape(-1, len(bound_ranges))
        .tolist()
    )
    # # save processed data
    # np.savez_compressed(args.npz_path + "_cleaned.npz", inputs=flattened_input, outputs=flattened_output)
    # print(f"Processed data saved at {args.npz_path + "_cleaned.npz"}")

    # model parameters
    in_features = flattened_input.shape[1]
    out_features = flattened_output.shape[1]
    num_regions = (
        args.num_v
        * args.num_x
        * args.num_y
        * args.num_t
        * args.num_vgoal
        * args.num_beta
        * args.num_angvz
    )

    # activation_idx = [0, 1, 2, 3, 4, 5, 6]
    activation_idx = np.arange(in_features).tolist()
    delta = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    basis_function = eval(args.basis_function)

    # rng
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    # model init
    wcrbf = WCRBFNet(
        in_features=in_features,
        out_features=out_features,
        num_kernels=args.num_k,
        basis_func=basis_function,
        num_regions=num_regions,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        dimension_ranges=dimension_ranges,
        activation_idx=activation_idx,
        delta=delta,
    )
    params = wcrbf.init(init_rng, jnp.ones((args.batch_size, in_features)))
    params_shape = jax.tree_util.tree_map(jnp.shape, unfreeze(params))
    print("Initialized parameter shapes:\n", params_shape)

    # optimizer
    optim = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr)
    )

    # train state
    state = train_state.TrainState.create(apply_fn=wcrbf.apply, params=params, tx=optim)

    # dynamic params for integration
    dyn_params = jnp.array(
        [
            args.mu,
            1.0489,
            0.04712,
            0.15875,
            0.17145,
            args.cs,
            args.cs,
            0.074,
            0.1,
            3.2,
            9.51,
            0.4189,
            7.0,
        ]
    )

    # train one step, with one step integration
    @jax.jit
    def train_step_oneint(state, x, y):
        initial_state = jnp.zeros((x.shape[0], 7))
        # v
        initial_state = initial_state.at[:, 3].set(x[:, 0])
        # beta
        initial_state = initial_state.at[:, 6].set(x[:, 5])
        # angv
        initial_state = initial_state.at[:, 5].set(x[:, 6])

        def loss_fn(params):
            y_predictions = wcrbf.apply(params, x)
            # x order, goal state
            # [v_c, x_g, y_g, t_g, v_g, beta, angv]
            x_pred_u = jnp.hstack((initial_state, y_predictions))
            x_u = jnp.hstack((initial_state, y))

            actual_integrated_states = dynamic_st_onestep_aux(x_u, dyn_params)
            predicted_integrated_states = dynamic_st_onestep_aux(x_pred_u, dyn_params)

            # loss = (
            #     jnp.abs(y_predictions - y).mean()
            #     + jnp.abs(
            #         predicted_integrated_states[:, [0, 1, 3, 4]]
            #         - actual_integrated_states[:, [0, 1, 3, 4]]
            #     ).mean()
            # )

            loss = (
                optax.l2_loss(predictions=y_predictions, targets=y).mean()
                + optax.l2_loss(
                    predictions=predicted_integrated_states[:, [0, 1, 3, 4]],
                    targets=actual_integrated_states[:, [0, 1, 3, 4]],
                ).mean()
            )

            # jax.debug.print("loss {l}", l=loss)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss_, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss_

    # train one step, with full five step integration
    @jax.jit
    def train_step_fullint(state, x, y):

        def loss_fn(params):
            DT = 0.1
            WB = 0.33
            MAX_SPEED = 7.0
            MIN_SPEED = 0.0
            MAX_STEER = 0.4189

            y_predictions = wcrbf.apply(params, x)

            batch_size = x.shape[0]
            x_ = jnp.zeros(batch_size)
            y_ = jnp.zeros(batch_size)
            delta = jnp.zeros(batch_size)
            v = jnp.clip(x[:, 0], a_min=MIN_SPEED, a_max=MAX_SPEED)
            yaw = jnp.zeros(batch_size)

            x_actual = x_
            y_actual = y_
            delta_actual = delta
            v_actual = v
            yaw_actual = yaw
            first_states_actual = None
            final_states_actual = None
            for i in range(5):
                # for i in range(1):
                a = y[:, i]
                delta_v = y[:, i + 5]
                x_actual = x_actual + v_actual * jnp.cos(yaw_actual) * DT
                y_actual = y_actual + v_actual * jnp.sin(yaw_actual) * DT
                delta_actual = delta_actual + delta_v * DT
                delta_actual = jnp.clip(delta_actual, a_min=-MAX_STEER, a_max=MAX_STEER)
                v_actual = v_actual + a * DT
                v_actual = jnp.clip(v_actual, a_min=MIN_SPEED, a_max=MAX_SPEED)
                yaw_actual = yaw_actual + (v_actual / WB) * jnp.tan(delta_actual) * DT
                if i == 0:
                    first_states_actual = jnp.vstack(
                        [x_actual, y_actual, delta_actual, v_actual, yaw_actual]
                    ).T
                if i == 4:
                    final_states_actual = jnp.vstack(
                        [x_actual, y_actual, delta_actual, v_actual, yaw_actual]
                    ).T

            x_pred = x_
            y_pred = y_
            delta_pred = delta
            v_pred = v
            yaw_pred = yaw
            first_states_pred = None
            final_states_pred = None
            for i in range(5):
                # for i in range(1):
                a = y_predictions[:, i]
                delta_v = y_predictions[:, i + 5]
                x_pred = x_pred + v_pred * jnp.cos(yaw_pred) * DT
                y_pred = y_pred + v_pred * jnp.sin(yaw_pred) * DT
                delta_pred = delta_pred + delta_v * DT
                delta_pred = jnp.clip(delta_pred, a_min=-MAX_STEER, a_max=MAX_STEER)
                v_pred = v_pred + a * DT
                v_pred = jnp.clip(v_pred, a_min=MIN_SPEED, a_max=MAX_SPEED)
                yaw_pred = yaw_pred + (v_pred / WB) * jnp.tan(delta_pred) * DT
                if i == 0:
                    first_states_pred = jnp.vstack(
                        [x_pred, y_pred, delta_pred, v_pred, yaw_pred]
                    ).T
                if i == 4:
                    final_states_pred = jnp.vstack(
                        [x_pred, y_pred, delta_pred, v_pred, yaw_pred]
                    ).T

            # loss = (
            #     optax.l2_loss(
            #         predictions=first_states_pred, targets=first_states_actual
            #     ).mean()
            #     + optax.l2_loss(
            #         predictions=final_states_pred, targets=final_states_actual
            #     ).mean()
            #     + optax.l2_loss(predictions=y_predictions, targets=y).mean()
            # )

            loss = (
                jnp.abs(y_predictions[:, [0, 5]] - y[:, [0, 5]]).mean()
                + jnp.abs(first_states_pred - first_states_pred).mean()
                + jnp.abs(final_states_pred - final_states_actual).mean()
            )

            # loss = (
            #     jnp.abs(y_predictions[:, [0, 5]] - y[:, [0, 5]]).mean()
            #     + jnp.abs(first_states_pred - first_states_pred).mean()
            #     + jnp.abs(final_states_pred - final_states_actual).mean()
            # )

            # loss = (
            #     optax.l2_loss(
            #         predictions=predicted_integrated_states[:, 0, :],
            #         targets=actual_integrated_states[:, 0, :],
            #     ).mean()
            #     + optax.l2_loss(
            #         predictions=predicted_integrated_states[:, -1, :],
            #         targets=actual_integrated_states[:, -1, :],
            #     ).mean()
            #     + optax.l2_loss(predictions=y_predictions, targets=y).mean()
            # )
            # jax.debug.print("predicted state any isnan {x}", x=jnp.any(jnp.isnan(predicted_integrated_states)))
            # jax.debug.print("actual state any isnan {y}", y=jnp.any(jnp.isnan(actual_integrated_states)))
            # jax.debug.print("loss: {l}", l=loss)

            # loss = optax.l2_loss(predictions=y_predictions, targets=y).mean()
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss_, grads = grad_fn(state.params)
        # jax.debug.print("grad norm {g}", g=jax.tree.map(jnp.linalg.norm, grads))
        # jax.debug.print("loss {l}", l=loss_)
        state = state.apply_gradients(grads=grads)
        return state, loss_

    # config logging
    yaml_dir = "./configs/" + args.run_name + ".yaml"
    CKPT_DIR = "ckpts/" + args.run_name
    flax.config.update("flax_use_orbax_checkpointing", False)
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    # config logging
    config_dict = {
        "in_features": in_features,
        "out_features": out_features,
        "num_kernels": args.num_k,
        "basis_func": basis_function.__name__,
        "num_regions": num_regions,
        "lower_bounds": [[float(l) for l in ll] for ll in lower_bounds],
        "upper_bounds": [[float(u) for u in uu] for uu in upper_bounds],
        "dimension_ranges": dimension_ranges,
        "activation_idx": activation_idx,
        "delta": delta,
        "epochs": args.train_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "mu": args.mu,
        "cs": args.cs,
    }
    with open(yaml_dir, "w+") as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)

    # logging
    wandb.init(project="irbfn", config=config_dict, tags=args.run_tags)

    def train_epoch(train_state, train_x, train_y, bs, epoch, epoch_rng):
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
            if args.only_onestep:
                train_state, batch_loss = train_step_oneint(
                    train_state, batch_x, batch_y
                )
            else:
                train_state, batch_loss = train_step_fullint(
                    train_state, batch_x, batch_y
                )
            batch_losses.append(batch_loss)
            wandb.log(
                {"train_loss_batch": jax.device_get(batch_loss)},
                step=b + (epoch * len(perms)),
            )

        batch_losses_np = jax.device_get(batch_losses)

        wandb.log(
            {"train_loss": np.mean(batch_losses_np)}, step=b + (epoch * len(perms))
        )

        return train_state

    # training
    for e in tqdm(range(args.train_epochs)):
        rng, perm_rng = jax.random.split(rng)
        state = train_epoch(
            state, flattened_input, flattened_output, args.batch_size, e, perm_rng
        )

        if e % 100 == 0:
            checkpoints.save_checkpoint(
                ckpt_dir=CKPT_DIR, target=state, step=e, keep=100
            )

    checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=e, keep=100)


if __name__ == "__main__":
    args = dnmpc_train_args()
    main(args)

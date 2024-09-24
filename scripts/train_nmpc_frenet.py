import os
import yaml
from datetime import datetime

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex

jax.config.update("jax_debug_nans", True)

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
from irbfn_mpc.model import WCRBFNet, DeeperWCRBFNet, MLP, ClusterWCRBFNet
from irbfn_mpc.arg_utils import dnmpc_frenet_train_args
from irbfn_mpc.dynamics import integrate_frenet_mult, dynamic_frenet_onestep_aux


def main(args):
    if args.use_float64:
        jax.config.update("jax_enable_x64", True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # if not os.path.exists(args.npz_path + "_cleaned.npz"):
    # loading raw data
    print("Loading data...")
    data = np.load(args.npz_path)
    inputs, outputs = data["inputs"], data["outputs"]
    print("Removing infeasible...")
    valid_ind = np.unique(np.where(outputs != -999)[0])
    inputs = inputs[valid_ind]
    outputs = outputs[valid_ind]
    if args.use_cluster:
        constraints = data["constraints"]
        constraints = constraints[valid_ind]
    # inputs [ey, delta, vx_car, vy_car, vx_goal, wz, epsi, curv]
    ey = inputs[:, 0].flatten()
    delta = inputs[:, 1].flatten()
    vx_car = inputs[:, 2].flatten()
    vy_car = inputs[:, 3].flatten()
    vx_goal = inputs[:, 4].flatten()
    wz = inputs[:, 5].flatten()
    epsi = inputs[:, 6].flatten()
    curv = inputs[:, 7].flatten()
    accel = outputs[:, :, 0]
    deltv = outputs[:, :, 1]

    centers = None
    if args.use_centers:
        assert (
            args.centers_name is not None
        ), "Must set centers file name suffix if use centers."
        centers_data = np.load(
            args.npz_path[:-4] + args.centers_name + args.npz_path[-4:]
        )
        centers = centers_data["centers"]

    # normalize
    # accel = (outputs[:, :, 0] - args.min_accl) / (args.max_accl - args.min_accl)
    # deltv = (outputs[:, :, 1] - args.min_steerv) / (args.max_steerv - args.min_steerv)
    print("Data import completed")

    # min_accl = args.min_accl
    # accl_range = args.max_accl - args.min_accl
    # min_steerv = args.min_steerv
    # steerv_range = args.max_steerv - args.min_steerv

    if args.mirror_data:
        print("Mirroring data...")
        ey_m = np.concatenate((ey, -ey), axis=0)
        delta_m = np.concatenate((delta, delta), axis=0)
        vx_car_m = np.concatenate((vx_car, vx_car), axis=0)
        vy_car_m = np.concatenate((vy_car, vy_car), axis=0)
        vx_goal_m = np.concatenate((vx_goal, vx_goal), axis=0)
        wz_m = np.concatenate((wz, wz), axis=0)
        epsi_m = np.concatenate((epsi, -epsi), axis=0)
        curv_m = np.concatenate((curv, curv), axis=0)
        accel_m = np.concatenate((accel, accel), axis=0)
        deltv_m = np.concatenate((deltv, -deltv), axis=0)
        print("Mirroring completed")
    else:
        ey_m = ey
        delta_m = delta
        vx_car_m = vx_car
        vy_car_m = vy_car
        vx_goal_m = vx_goal
        wz_m = wz
        epsi_m = epsi
        curv_m = curv
        accel_m = accel
        deltv_m = deltv

    print("Generating bounds...")

    ey_bounds = np.sort(np.unique(ey_m))
    delta_bounds = np.sort(np.unique(delta_m))
    vx_car_bounds = np.sort(np.unique(vx_car_m))
    vy_car_bounds = np.sort(np.unique(vy_car_m))
    vx_goal_bounds = np.sort(np.unique(vx_goal_m))
    wz_bounds = np.sort(np.unique(wz_m))
    epsi_bounds = np.sort(np.unique(epsi_m))
    curv_bounds = np.sort(np.unique(curv_m))

    ey_ind = np.linspace(
        start=0, stop=len(ey_bounds) - 1, num=args.num_ey + 1, endpoint=True, dtype=int
    )
    delta_ind = np.linspace(
        start=0,
        stop=len(delta_bounds) - 1,
        num=args.num_delta + 1,
        endpoint=True,
        dtype=int,
    )
    vx_car_ind = np.linspace(
        start=0,
        stop=len(vx_car_bounds) - 1,
        num=args.num_vx_car + 1,
        endpoint=True,
        dtype=int,
    )
    vy_car_ind = np.linspace(
        start=0,
        stop=len(vy_car_bounds) - 1,
        num=args.num_vy_car + 1,
        endpoint=True,
        dtype=int,
    )
    vx_goal_ind = np.linspace(
        start=0,
        stop=len(vx_goal_bounds) - 1,
        num=args.num_vx_goal + 1,
        endpoint=True,
        dtype=int,
    )
    wz_ind = np.linspace(
        start=0,
        stop=len(wz_bounds) - 1,
        num=args.num_wz + 1,
        endpoint=True,
        dtype=int,
    )
    epsi_ind = np.linspace(
        start=0,
        stop=len(epsi_bounds) - 1,
        num=args.num_epsi + 1,
        endpoint=True,
        dtype=int,
    )
    curv_ind = np.linspace(
        start=0,
        stop=len(curv_bounds) - 1,
        num=args.num_curv + 1,
        endpoint=True,
        dtype=int,
    )

    lower_bounds = [
        list(ey_bounds[ey_ind[:-1]]),
        list(delta_bounds[delta_ind[:-1]]),
        list(vx_car_bounds[vx_car_ind[:-1]]),
        list(vy_car_bounds[vy_car_ind[:-1]]),
        list(vx_goal_bounds[vx_goal_ind[:-1]]),
        list(wz_bounds[wz_ind[:-1]]),
        list(epsi_bounds[epsi_ind[:-1]]),
        list(curv_bounds[curv_ind[:-1]]),
    ]
    upper_bounds = [
        list(ey_bounds[ey_ind[1:]]),
        list(delta_bounds[delta_ind[1:]]),
        list(vx_car_bounds[vx_car_ind[1:]]),
        list(vy_car_bounds[vy_car_ind[1:]]),
        list(vx_goal_bounds[vx_goal_ind[1:]]),
        list(wz_bounds[wz_ind[1:]]),
        list(epsi_bounds[epsi_ind[1:]]),
        list(curv_bounds[curv_ind[1:]]),
    ]

    print("Bounds defined")

    print("Generating input and output...")
    flattened_input = np.vstack(
        [ey_m, delta_m, vx_car_m, vy_car_m, vx_goal_m, wz_m, epsi_m, curv_m]
    ).T
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
        args.num_ey
        * args.num_delta
        * args.num_vx_car
        * args.num_vy_car
        * args.num_vx_goal
        * args.num_wz
        * args.num_epsi
        * args.num_curv
    )

    if args.use_cluster:
        # leave 1 region of networks for outside top k
        num_regions = args.num_clusters + 1
        # get clusterid one hots
        print("Loading cluster int ids")
        c_data = np.load(
            args.npz_path[:-4] + f"_{args.num_clusters}_cluster_ids" + args.npz_path[-4:]
        )
        cluster_int_ids = c_data["cluster_int_ids"][valid_ind]
        # convert to one hot
        flattened_clusterid = np.zeros((len(cluster_int_ids), num_regions), dtype=int)
        flattened_clusterid[np.arange(len(cluster_int_ids)), cluster_int_ids] = 1

    activation_idx = np.arange(in_features).tolist()
    delta = [15.0, 10.0, 100.0, 100.0, 100.0, 10.0, 10.0, 10.0]
    basis_function = eval(args.basis_function)

    # rng
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    # model init
    if args.deeper:
        wcrbf = DeeperWCRBFNet(
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
    elif args.mlp:
        wcrbf = MLP(
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
    elif args.use_cluster:
        wcrbf = ClusterWCRBFNet(
            in_features=in_features,
            out_features=out_features,
            num_kernels=args.num_k,
            basis_func=basis_function,
            num_regions=num_regions,
        )
    else:
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
            centers=centers,
            fixed_centers=args.fixed_centers,
            fixed_width=args.fixed_width,
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
        # input states x [ey, delta, vx_car, vy_car, vx_goal, wz, epsi, curv]
        # output from dynamics [ey, delta, vx_car, vy_car, wz, epsi]
        initial_state = x[:, [0, 1, 2, 3, 5, 6, 7]]

        def loss_fn(params):
            y_predictions = wcrbf.apply(params, x)
            pred_loss = jnp.abs(y_predictions - y).mean()
            # pred_loss = optax.l2_loss(predictions=y_predictions, targets=y).mean()
            # # unnorm
            # y_predictions = y_predictions.at[:, 0].multiply(accl_range)
            # y_predictions = y_predictions.at[:, 0].add(min_accl)
            # y_predictions = y_predictions.at[:, 1].multiply(steerv_range)
            # y_predictions = y_predictions.at[:, 1].add(min_steerv)
            # [ey, delta, vx_car, vy_car, vx_goal, wz, epsi, curv]
            x_pred_u = jnp.hstack((initial_state, y_predictions))
            x_u = jnp.hstack((initial_state, y))

            actual_integrated_states = dynamic_frenet_onestep_aux(x_u, dyn_params)
            predicted_integrated_states = dynamic_frenet_onestep_aux(
                x_pred_u, dyn_params
            )

            int_loss = jnp.abs(
                predicted_integrated_states - actual_integrated_states
            ).mean()

            # int_loss = optax.l2_loss(
            #     predictions=predicted_integrated_states,
            #     targets=actual_integrated_states
            # ).mean()

            loss = pred_loss + 100.0 * int_loss

            # loss = pred_loss
            # int_loss = 0.0

            # loss = (
            #     optax.l2_loss(predictions=y_predictions, targets=y).mean()
            #     + optax.l2_loss(
            #         predictions=predicted_integrated_states[:, [0, 1, 3, 4]],
            #         targets=actual_integrated_states[:, [0, 1, 3, 4]],
            #     ).mean()
            # )

            # jax.debug.print("loss {l}", l=loss)
            return loss, (pred_loss, 100.0 * int_loss)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_, (pred_loss_, int_loss_)), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss_, pred_loss_, int_loss_

    # train one step, with full five step integration
    @jax.jit
    def train_step_fullint(state, x, y):
        # input states x [ey, delta, vx_car, vy_car, vx_goal, wz, epsi, curv]
        # output from dynamics [ey, delta, vx_car, vy_car, wz, epsi]
        initial_state = x[:, [0, 0, 1, 2, 3, 5, 6, 7]]

        def loss_fn(params):
            y_predictions = wcrbf.apply(params, x)
            pred_loss = jnp.abs(y_predictions - y).mean()
            # pred_loss = optax.huber_loss(y_predictions, y).mean()

            x_pred_u = jnp.hstack((initial_state, y_predictions))
            x_u = jnp.hstack((initial_state, y))

            actual_states = integrate_frenet_mult(x_u, dyn_params)
            pred_states = integrate_frenet_mult(x_pred_u, dyn_params)
            int_loss = jnp.abs(pred_states - actual_states).mean()
            # int_loss = optax.huber_loss(pred_states[:, [0, 5], :], actual_states[:, [0, 5], :]).mean()

            loss = pred_loss + int_loss
            return loss, (pred_loss, int_loss)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_, (pred_loss_, int_loss_)), grads = grad_fn(state.params)
        # jax.debug.print("grad norm {g}", g=jax.tree.map(jnp.linalg.norm, grads))
        # jax.debug.print("loss {l}", l=loss_)
        state = state.apply_gradients(grads=grads)
        return state, loss_, pred_loss_, int_loss_

    # train one step, with full five step integration
    @jax.jit
    def train_step_fullint_withcluster(state, x, y, cluster_ids):
        # input states x [ey, delta, vx_car, vy_car, vx_goal, wz, epsi, curv]
        # output from dynamics [ey, delta, vx_car, vy_car, wz, epsi]
        initial_state = x[:, [0, 0, 1, 2, 3, 5, 6, 7]]

        def loss_fn(params):
            y_predictions, logits = wcrbf.apply(params, x)
            cluster_loss = optax.softmax_cross_entropy(logits, cluster_ids).mean()

            pred_loss = jnp.abs(y_predictions - y).mean()
            # pred_loss = optax.huber_loss(y_predictions, y).mean()

            x_pred_u = jnp.hstack((initial_state, y_predictions))
            x_u = jnp.hstack((initial_state, y))

            actual_states = integrate_frenet_mult(x_u, dyn_params)
            pred_states = integrate_frenet_mult(x_pred_u, dyn_params)
            int_loss = jnp.abs(pred_states - actual_states).mean()
            # int_loss = optax.huber_loss(pred_states[:, [0, 5], :], actual_states[:, [0, 5], :]).mean()

            loss = pred_loss + int_loss + cluster_loss
            return loss, (pred_loss, int_loss, cluster_loss)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_, (pred_loss_, int_loss_, cluster_loss_)), grads = grad_fn(state.params)
        # jax.debug.print("grad norm {g}", g=jax.tree.map(jnp.linalg.norm, grads))
        # jax.debug.print("loss {l}", l=loss_)
        state = state.apply_gradients(grads=grads)
        return state, loss_, pred_loss_, int_loss_, cluster_loss_

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

    def train_epoch(
        train_state, train_x, train_y, bs, epoch, epoch_rng, train_cluster_id=None
    ):
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
            if train_cluster_id is not None:
                batch_cluster_id = train_cluster_id[perm, :]
            if args.only_onestep:
                train_state, batch_loss, pred_loss, int_loss = train_step_oneint(
                    train_state, batch_x, batch_y
                )
            elif args.use_cluster:
                train_state, batch_loss, pred_loss, int_loss, cluster_loss = (
                    train_step_fullint_withcluster(
                        train_state,
                        batch_x,
                        batch_y,
                        batch_cluster_id,
                    )
                )
            else:
                train_state, batch_loss, pred_loss, int_loss = train_step_fullint(
                    train_state, batch_x, batch_y
                )
            batch_losses.append(batch_loss)
            wandb.log(
                {
                    "train_loss_batch": jax.device_get(batch_loss),
                    "pred_loss_batch": jax.device_get(pred_loss),
                    "int_loss_batch": jax.device_get(int_loss),
                    "cluster_loss_batch": (
                        jax.device_get(cluster_loss) if args.use_cluster else None
                    ),
                },
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
        if args.use_cluster:
            state = train_epoch(
                state,
                flattened_input,
                flattened_output,
                args.batch_size,
                e,
                perm_rng,
                flattened_clusterid,
            )
        else:
            state = train_epoch(
                state, flattened_input, flattened_output, args.batch_size, e, perm_rng
            )

        if e % 100 == 0:
            checkpoints.save_checkpoint(
                ckpt_dir=CKPT_DIR, target=state, step=e, keep=100
            )

    checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=e, keep=100)


if __name__ == "__main__":
    args = dnmpc_frenet_train_args()
    main(args)

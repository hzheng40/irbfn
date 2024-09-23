import argparse


def dnmpc_table_gen_args():
    parser = argparse.ArgumentParser()
    # gridding
    parser.add_argument("--v_car_min", type=float, default=0.0)
    parser.add_argument("--v_car_max", type=float, default=7.0)
    parser.add_argument("--dv_car", type=float, default=1.0)
    parser.add_argument("--x_goal_min", type=float, default=0.0)
    parser.add_argument("--x_goal_max", type=float, default=3.5)
    parser.add_argument("--dx_goal", type=float, default=0.2)
    parser.add_argument("--y_goal_min", type=float, default=0.0)
    parser.add_argument("--y_goal_max", type=float, default=3.5)
    parser.add_argument("--dy_goal", type=float, default=0.2)
    parser.add_argument("--t_goal_min", type=float, default=-3.1)
    parser.add_argument("--t_goal_max", type=float, default=3.1)
    parser.add_argument("--dt_goal", type=float, default=0.1)
    parser.add_argument("--v_goal_min", type=float, default=0.0)
    parser.add_argument("--v_goal_max", type=float, default=7.0)
    parser.add_argument("--dv_goal", type=float, default=1.0)
    parser.add_argument("--beta_min", type=float, default=-0.6)
    parser.add_argument("--beta_max", type=float, default=0.6)
    parser.add_argument("--dbeta", type=float, default=0.2)
    parser.add_argument("--angv_z_min", type=float, default=-3.0)
    parser.add_argument("--angv_z_max", type=float, default=3.0)
    parser.add_argument("--dang_v", type=float, default=0.5)
    # run config
    parser.add_argument("--n_jobs", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="/data/tables/")
    # parser.add_argument("--ouput_steps", type=int, default=2)
    # mpc model config
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--cs", type=float, default=5.0)
    args = parser.parse_args()
    return args


def dnmpc_frenet_table_gen_args():
    parser = argparse.ArgumentParser()
    # gridding
    # states for frenet nmpc are [ey, delta, vx, vy, vgoal, wz, epsi, curv]
    parser.add_argument("--ey_min", type=float, default=-0.2)
    parser.add_argument("--ey_max", type=float, default=2.0)
    parser.add_argument("--num_ey", type=int, default=12)

    parser.add_argument("--delta_min", type=float, default=-0.3)
    parser.add_argument("--delta_max", type=float, default=0.3)
    parser.add_argument("--num_delta", type=int, default=7)

    parser.add_argument("--vx_car_min", type=float, default=1.0)
    parser.add_argument("--vx_car_max", type=float, default=7.0)
    parser.add_argument("--vy_car_min", type=float, default=-1.0)
    parser.add_argument("--vy_car_max", type=float, default=1.0)
    parser.add_argument("--num_vx_car", type=int, default=11)
    parser.add_argument("--num_vy_car", type=int, default=11)

    parser.add_argument("--vx_goal_min", type=float, default=3.0)
    parser.add_argument("--vx_goal_max", type=float, default=7.0)
    parser.add_argument("--num_v_goal", type=int, default=5)

    parser.add_argument("--wz_min", type=float, default=-2.6)
    parser.add_argument("--wz_max", type=float, default=2.6)
    parser.add_argument("--num_wz", type=int, default=11)

    parser.add_argument("--epsi_min", type=float, default=-1.0)
    parser.add_argument("--epsi_max", type=float, default=1.0)
    parser.add_argument("--num_epsi", type=int, default=11)

    parser.add_argument("--curv_min", type=float, default=-0.1)
    parser.add_argument("--curv_max", type=float, default=0.1)
    parser.add_argument("--num_curv", type=int, default=3)

    # run config
    parser.add_argument("--n_jobs", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="/data/tables/frenet/")
    parser.add_argument("--additional_run_name", type=str, default="")

    # mpc model config
    parser.add_argument("--mu_min", type=float, default=0.5)
    parser.add_argument("--mu_max", type=float, default=1.0)
    parser.add_argument("--d_mu", type=float, default=0.1)
    parser.add_argument("--cs", type=float, default=5.0)
    args = parser.parse_args()
    return args


def dnmpc_train_args():
    parser = argparse.ArgumentParser()
    # splits
    parser.add_argument("--num_v", type=int, default=1)
    parser.add_argument("--num_x", type=int, default=1)
    parser.add_argument("--num_y", type=int, default=2)
    parser.add_argument("--num_t", type=int, default=2)
    parser.add_argument("--num_vgoal", type=int, default=1)
    parser.add_argument("--num_beta", type=int, default=1)
    parser.add_argument("--num_angvz", type=int, default=1)
    # basis func
    parser.add_argument("--basis_function", type=str, default="gaussian")
    # data
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--mirror_data", action="store_true")
    parser.add_argument("--only_onestep", action="store_true")
    # training
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=80000)
    parser.add_argument("--num_k", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=1000)
    parser.add_argument("--use_float64", action="store_true")
    parser.add_argument("--run_name", type=str, default="dnmpc_4regions")
    parser.add_argument("--run_tags", nargs="+", type=str)

    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--cs", type=float, default=5.0)
    
    args = parser.parse_args()
    return args


def dnmpc_frenet_train_args():
    parser = argparse.ArgumentParser()
    # splits
    parser.add_argument("--num_ey", type=int, default=1)
    parser.add_argument("--num_delta", type=int, default=1)
    parser.add_argument("--num_vx_car", type=int, default=1)
    parser.add_argument("--num_vy_car", type=int, default=1)
    parser.add_argument("--num_vx_goal", type=int, default=1)
    parser.add_argument("--num_wz", type=int, default=1)
    parser.add_argument("--num_epsi", type=int, default=1)
    parser.add_argument("--num_curv", type=int, default=1)
    # basis func
    parser.add_argument("--basis_function", type=str, default="gaussian")
    parser.add_argument("--deeper", action="store_true")
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--normalize_input", action="store_true")

    # data
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--mirror_data", action="store_true")
    parser.add_argument("--only_onestep", action="store_true")
    # training
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=80000)
    parser.add_argument("--num_k", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=10000)
    parser.add_argument("--use_float64", action="store_true")
    parser.add_argument("--run_name", type=str, default="dnmpc_4regions")
    parser.add_argument("--run_tags", nargs="+", type=str)
    # normalization
    parser.add_argument("--max_accl", type=float, default=9.51)
    parser.add_argument("--min_accl", type=float, default=-9.51)
    parser.add_argument("--max_steerv", type=float, default=3.14159)
    parser.add_argument("--min_steerv", type=float, default=-3.14159)
    # warmstart/center
    parser.add_argument("--use_centers", action="store_true")
    parser.add_argument("--fixed_centers", action="store_true")
    parser.add_argument("--fixed_width", action="store_true")
    parser.add_argument("--centers_name", type=str, default="_top300mean")

    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--cs", type=float, default=5.0)
    
    args = parser.parse_args()
    return args


def dnmpc_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--j", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=10, help="number of trials per parameter combo")
    parser.add_argument("--num_mu", type=int, default=10)
    parser.add_argument("--mu_min", type=float, default=0.5)
    parser.add_argument("--mu_max", type=float, default=1.1)
    parser.add_argument("--num_cs", type=int, default=10)
    parser.add_argument("--cs_min", type=float, default=1.)
    parser.add_argument("--cs_max", type=float, default=10.)
    parser.add_argument("--out_name", type=str, default="dnmpc_eval_results")
    parser.add_argument("--noise_scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    return args


def irbfn_dnmpc_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_f", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    return args
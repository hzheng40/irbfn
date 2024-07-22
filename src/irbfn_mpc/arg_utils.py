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
    parser.add_argument("--angv_z_min", type=float, default=0.0)
    parser.add_argument("--angv_z_max", type=float, default=3.0)
    parser.add_argument("--dang_v", type=float, default=0.5)
    # run config
    parser.add_argument("--n_jobs", type=int, default=80)
    parser.add_argument("--save_path", type=str, default="/data/tables")
    # parser.add_argument("--ouput_steps", type=int, default=2)
    # mpc model config
    parser.add_argument("--mu", type=float, default=1.0489)
    # TODO: add more?
    args = parser.parse_args()
    return args
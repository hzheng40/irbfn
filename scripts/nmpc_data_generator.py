import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from irbfn_mpc.arg_utils import dnmpc_table_gen_args

from irbfn_mpc.nonlinear_dmpc import NMPCPlanner as DNMPCPlanner
from irbfn_mpc.nonlinear_dmpc import mpc_config as dmpc_config


def gen_data(args):
    print(f"Instantiating MPC solver")
    mpc_config = dmpc_config(MU=args.mu, C_SF=args.cs, C_SR=args.cs)

    def mpc_solve(chunks, worker_i):
        solutions = []
        solver = DNMPCPlanner(config=mpc_config)
        iterations = tqdm(chunks) if worker_i == 69 else chunks

        for chunk in iterations:
            v_car, x_goal, y_goal, t_goal, v_goal, beta, angv_z = chunk
            inputs = np.array([v_car, x_goal, y_goal, t_goal, v_goal, beta, angv_z])
            current_state = {
                "pose_x": 0.0,
                "pose_y": 0.0,
                "pose_theta": 0.0,
                "delta": 0.0,
                "linear_vel_x": v_car,
                "ang_vel_z": angv_z,
                "beta": beta,
            }
            goal_state = np.array([x_goal, y_goal, 0.0, v_goal, t_goal, 0.0, 0.0])
            solutions.append((inputs, solver.mpc_prob_solve(goal_state, current_state)))
        return solutions

    print("Generating input state mesh grid")
    v_car = np.arange(args.v_car_min, args.v_car_max + args.dv_car, args.dv_car)
    x_goal = np.arange(args.x_goal_min, args.x_goal_max + args.dx_goal, args.dx_goal)
    y_goal = np.arange(args.y_goal_min, args.y_goal_max + args.dy_goal, args.dy_goal)
    t_goal = np.arange(args.t_goal_min, args.t_goal_max + args.dt_goal, args.dt_goal)
    v_goal = np.arange(args.v_goal_min, args.v_goal_max + args.dv_goal, args.dv_goal)
    beta = np.arange(args.beta_min, args.beta_max, args.dbeta)
    angv_z = np.arange(args.angv_z_min, args.angv_z_max, args.dang_v)

    num_v = len(v_car)
    num_x = len(x_goal)
    num_y = len(y_goal)
    num_t = len(t_goal)
    num_v_goal = len(v_goal)
    num_beta = len(beta)
    num_angv_z = len(angv_z)
    filename = f"{num_v}v_{num_x}x_{num_y}y_{num_t}t_{num_v_goal}vgoal_{num_beta}beta_{num_angv_z}angvz_mu{args.mu}_cs{args.cs}.npz"

    v_car_m, x_goal_m, y_goal_m, t_goal_m, v_goal_m, beta_m, angv_z_m = np.meshgrid(
        v_car, x_goal, y_goal, t_goal, v_goal, beta, angv_z, indexing="ij"
    )
    v_car = v_car_m.flatten()
    x_goal = x_goal_m.flatten()
    y_goal = y_goal_m.flatten()
    t_goal = t_goal_m.flatten()
    v_goal = v_goal_m.flatten()
    beta = beta_m.flatten()
    angv_z = angv_z_m.flatten()

    print(f"Input state mesh grid generation completed: {len(v_car)} samples")
    all_chunks = np.column_stack((v_car, x_goal, y_goal, t_goal, v_goal, beta, angv_z))
    # randomize order
    np.random.shuffle(all_chunks)
    balanced_chunks = np.array_split(all_chunks, args.n_jobs, axis=0)

    print(f"Generating {filename}.")
    table_frags = Parallel(n_jobs=args.n_jobs)(
        delayed(mpc_solve)(chunks, worker_i) for worker_i, chunks in enumerate(balanced_chunks)
    )
    table = []
    for frags in table_frags:
        table += frags

    print(f"Saving {filename}.")
    table = [solution for solution in table if solution[1] is not None]
    inputs = np.array([solution[0] for solution in table])
    outputs = np.array([solution[1] for solution in table])
    outputs = np.moveaxis(outputs, 1, 2)

    np.savez(args.save_path + filename, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    args = dnmpc_table_gen_args()
    gen_data(args)

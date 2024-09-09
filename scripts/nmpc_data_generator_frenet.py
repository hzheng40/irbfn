import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from irbfn_mpc.arg_utils import dnmpc_frenet_table_gen_args

from irbfn_mpc.nonlinear_dmpc_frenet import NMPCPlanner as DNMPCPlanner
from irbfn_mpc.nonlinear_dmpc_frenet import mpc_config as dmpc_config


def gen_data(args):
    
    all_mu = np.arange(args.mu_min, args.mu_max + args.d_mu, args.d_mu)

    for mu in all_mu[::-1]:
        print(f"Generating table for mu {mu}")
        mpc_config = dmpc_config(
            MU=mu,
            C_SF=args.cs,
            C_SR=args.cs,
        )

        def mpc_solve(chunks, worker_i):
            solutions = []
            solver = DNMPCPlanner(config=mpc_config)
            iterations = tqdm(chunks) if worker_i == 12 else chunks

            for chunk in iterations:
                ey, delta, vx_car, vy_car, vx_goal, wz, epsi, curv = chunk
                inputs = np.array([ey, delta, vx_car, vy_car, vx_goal, wz, epsi, curv])
                solutions.append((inputs, solver.mpc_prob_solve_aux(inputs)))
            return solutions

        print("Generating input state mesh grid")
        ey = np.arange(args.ey_min, args.ey_max + args.d_ey, args.d_ey)
        delta = np.arange(args.delta_min, args.delta_max + args.d_delta, args.d_delta)
        vx_car = np.arange(args.vx_car_min, args.vx_car_max + args.d_vx_car, args.d_vx_car)
        vy_car = np.arange(args.vy_car_min, args.vy_car_max + args.d_vy_car, args.d_vy_car)
        vx_goal = np.arange(args.vx_goal_min, args.vx_goal_max + args.d_v_goal, args.d_v_goal)
        wz = np.arange(args.wz_min, args.wz_max + args.d_wz, args.d_wz)
        epsi = np.arange(args.epsi_min, args.epsi_max + args.d_epsi, args.d_epsi)
        curv = np.arange(args.curv_min, args.curv_max + args.d_curv, args.d_curv)

        print(f"ey: {ey}")
        print(f"delta: {delta}")
        print(f"vx_car: {vx_car}")
        print(f"vy_car: {vy_car}")
        print(f"vx_goal: {vx_goal}")
        print(f"wz: {wz}")
        print(f"epsi: {epsi}")
        print(f"curv: {curv}")

        num_ey = len(ey)
        num_delta = len(delta)
        num_vx_car = len(vx_car)
        num_vy_car = len(vy_car)
        num_vx_goal = len(vx_goal)
        num_wz = len(wz)
        num_epsi = len(epsi)
        num_curv = len(curv)
        filename = f"{num_ey}ey_{num_delta}delta_{num_vx_car}vxcar_{num_vy_car}vycar_{num_vx_goal}vxgoal_{num_wz}wz_{num_epsi}epsi_{num_curv}curv_mu{mu}_cs{args.cs}.npz"

        ey_m, delta_m, vx_car_m, vy_car_m, vx_goal_m, wz_m, epsi_m, curv_m = np.meshgrid(
            ey, delta, vx_car, vy_car, vx_goal, wz, epsi, curv, indexing="ij"
        )
        ey = ey_m.flatten()
        delta = delta_m.flatten()
        vx_car = vx_car_m.flatten()
        vy_car = vy_car_m.flatten()
        vx_goal = vx_goal_m.flatten()
        wz = wz_m.flatten()
        epsi = epsi_m.flatten()
        curv = curv_m.flatten()

        print(f"Input state mesh grid generation completed: {len(ey)} samples")
        all_chunks = np.column_stack((ey, delta, vx_car, vy_car, vx_goal, wz, epsi, curv))

        # randomize order
        # np.random.shuffle(all_chunks)
        rand_ind = np.arange(all_chunks.shape[0])
        np.random.shuffle(rand_ind)
        unrand_ind = np.argsort(rand_ind)

        all_chunks = all_chunks[rand_ind]

        balanced_chunks = np.array_split(all_chunks, args.n_jobs, axis=0)

        print(f"Generating {filename}.")
        table_frags = Parallel(n_jobs=args.n_jobs)(
            delayed(mpc_solve)(chunks, worker_i)
            for worker_i, chunks in enumerate(balanced_chunks)
        )
        table = []
        for frags in table_frags:
            table += frags

        print(f"Saving {filename}.")
        table = [solution for solution in table if solution[1] is not None]
        inputs = np.array([solution[0] for solution in table])
        outputs = np.array([solution[1] for solution in table])
        outputs = np.moveaxis(outputs, 1, 2)

        inputs_sorted = inputs[unrand_ind, :]
        outputs_sorted = outputs[unrand_ind, :]

        np.savez(args.save_path + filename, inputs=inputs, outputs=outputs)
        np.savez(
            args.save_path + filename[:-4] + "_sorted" + filename[-4:],
            inputs=inputs_sorted,
            outputs=outputs_sorted,
        )


if __name__ == "__main__":
    args = dnmpc_frenet_table_gen_args()
    gen_data(args)

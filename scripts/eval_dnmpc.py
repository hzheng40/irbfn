import numpy as np
import gymnasium as gym
import time
from joblib import Parallel, delayed
from irbfn_mpc.nonlinear_dmpc import NMPCPlanner as DNMPCPlanner
from tqdm import tqdm
from irbfn_mpc.arg_utils import dnmpc_eval_args
import pickle

def run_simulation(cs, mu, n_trials, root_seed, task_i, std_dev):
    rng = np.random.default_rng([root_seed, task_i])

    trajectories = []
    params_dict = {
        "mu": mu,
        "C_Sf": cs,
        "C_Sr": cs,
    }
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg_blank",
            "num_agents": 1,
            "control_input": ["accl", "steering_speed"],
            "observation_config": {"type": "dynamic_state"},
            "params": params_dict,
        },
        render_mode="rgb_array",
    )
    track = env.unwrapped.track
    planner = DNMPCPlanner(track=track, debug=False)
    path_length = track.centerline.ss[-1]
    planner.config.dlk = (
        track.centerline.ss[1] - track.centerline.ss[0]
    )

    trial_done = 0
    trial_fail = False

    while trial_done < n_trials:
        # reset environment
        poses = np.array(
            [
                [
                    track.centerline.xs[10] + 0.01,
                    track.centerline.ys[10] + 0.01,
                    track.centerline.yaws[10],
                ]
            ]
        )
        noise = rng.normal(loc=0.0, scale=std_dev, size=poses.shape)
        noise[0, -1] = 0.0
        obs, info = env.reset(options={"poses": poses + noise})
        # obs, info = env.reset(options={"poses": poses})
        theta = obs["agent_0"]["pose_theta"]
        current_x = obs["agent_0"]["pose_x"]
        current_y = obs["agent_0"]["pose_y"]
        current_s, current_ey, current_ephi = track.cartesian_to_frenet(
            current_x, current_y, theta
        )
        previous_s = current_s

        done = False
        laptime = 0.0
        start = time.time()
        traj = []
        while not done:
            if (time.time() - start) > 500:
                break
            try:
                accl, steerv = planner.plan(obs["agent_0"])
            except AttributeError as e:
                trial_fail = True
                break
            obs, timestep, terminated, truncated, infos = env.step(
                np.array([[steerv[0], accl[0]]])
            )
            done = terminated or truncated
            laptime += timestep

            traj.append(
                [current_x, current_y, theta, current_s, current_ey, current_ephi]
            )

            theta = obs["agent_0"]["pose_theta"]
            current_x = obs["agent_0"]["pose_x"]
            current_y = obs["agent_0"]["pose_y"]
            current_s, current_ey, current_ephi = (
                track.cartesian_to_frenet(current_x, current_y, theta)
            )

            if np.abs(current_s - previous_s) > 0.99 * path_length and laptime > 20.0:
                break

            previous_s = current_s

        if not trial_fail:
            trajectories.append(traj)
            trial_done += 1
        else:
            trial_fail = False

    return trajectories


def eval_dnmpc(args):
    mu = np.linspace(args.mu_min, args.mu_max, args.num_mu)
    cs = np.linspace(args.cs_min, args.cs_max, args.num_cs)

    mu_m, cs_m = np.meshgrid(mu, cs, indexing="ij")
    all_combo = np.column_stack((cs_m.flatten(), mu_m.flatten()))

    with open(args.out_name + "_inputs.pkl", "wb") as input_file:
        pickle.dump(all_combo, input_file)
        input_file.close()

    results = Parallel(n_jobs=args.j)(
        delayed(run_simulation)(
            c[0],
            c[1],
            args.num_trials,
            args.seed,
            task_i,
            std_dev=args.noise_scale,
        )
        for task_i, c in enumerate(tqdm(all_combo))
    )
    with open(args.out_name + ".pkl", "wb") as file:
        pickle.dump(results, file)
        file.close()


if __name__ == "__main__":
    args = dnmpc_eval_args()
    eval_dnmpc(args)

    # directory = "results"
    # # Check if the directory exists
    # if not os.path.exists(directory):
    #     # Create the directory
    #     os.makedirs(directory)
    #     print(f"Directory '{directory}' created.")
    # else:
    #     print(f"Directory '{directory}' already exists.")
    # # Run the simulation in parallel
    # dmpc()
    # # Save the files in results directory to a new txt file called results.txt
    # # Define the directory containing the result files and the output file
    # results_folder = directory
    # output_file = "results.txt"
    # with open(output_file, "w") as outfile:
    #     outfile.write(
    #         "C_Sf,C_Sr,h,mu,laptime,s,avg_ey,std_ey,avg_ephi,std_ephi,s_score,ephi_collapse\n"
    #     )
    # # Open the output file in append mode
    # with open(output_file, "a") as outfile:
    #     # Iterate through each file in the results folder
    #     for filename in os.listdir(results_folder):
    #         # Create the full path to the file
    #         file_path = os.path.join(results_folder, filename)
    #         # Ensure it's a file before trying to read it
    #         if os.path.isfile(file_path):
    #             # Open and read the content of the file
    #             with open(file_path, "r") as infile:
    #                 # Write the content to the output file
    #                 outfile.write(infile.read())

    # print(f"All files have been combined into {output_file}")

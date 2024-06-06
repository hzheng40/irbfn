import gymnasium as gym
from irbfn_mpc.nonlinear_mpc import NMPCPlanner
import numpy as np


def main():
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "observation_config": {"type": "kinematic_state"},
        },
        render_mode="rgb_array",
    )
    env = gym.wrappers.RecordVideo(env, "video_nmpc")
    track = env.unwrapped.track
    clx = track.centerline.xs
    cly = track.centerline.ys
    clt = track.centerline.ts

    mpc = NMPCPlanner()
    obs, info = env.reset()
    done = False

    while not done:
        agent_obs = obs[obs.keys()[0]]
        current_state = {
            k: agent_obs[k]
            for k in ["pose_x", "pose_y", "pose_theta", "delta", "linear_vel_x"]
        }
        goal_state = np.array([x_goal, y_goal, 0.0, v_goal, t_goal])
        speeds, steers = mpc.plan(current_state, goal_state)
        action = env.action_space.sample()
        action[0] = [steers[0], speeds[0]]

        obs, step_reward, done, truncated, info = env.step(action)


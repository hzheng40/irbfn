import numpy as np
import gymnasium as gym
from irbfn_mpc.nonlinear_dmpc_frenet import NMPCPlanner

if __name__ == "__main__":
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "goggle",
            "observation_config": {"type": "frenet_dynamic_state"},
            "num_agents": 1,
            "control_input": "accl",
        },
        render_mode="human_fast",
    )
    track = env.unwrapped.track
    mpc = NMPCPlanner(track=track)

    env.unwrapped.add_render_callback(mpc.render_waypoints)
    env.unwrapped.add_render_callback(mpc.render_mpc_sol)
    env.unwrapped.add_render_callback(mpc.render_local_plan)

    poses = np.array(
        [
            [
                env.unwrapped.track.raceline.xs[0],
                env.unwrapped.track.raceline.ys[0],
                env.unwrapped.track.raceline.yaws[0],
            ]
        ]
    )
    obs, info = env.reset(options={"poses": poses})
    done = False

    step = 0

    while not done:
        current_state = obs["agent_0"]
        
        if step < 10:
            accl = 3.0
            steer_vel = 0.0
        else:
            accl, steer_vel = mpc.plan(current_state)
            # print(f"accl {accl}, steer_vel {steer_vel}")

        action = env.action_space.sample()
        action[0] = [steer_vel, accl]
        # print(f"current state {current_state}")
        # print(f"goal state {goal_state}")
        # print(f"taking action {action[0]}")

        obs, step_reward, done, truncated, info = env.step(action)
        step += 1
        env.render()

    env.close()
import os
import time

import gymnasium as gym
import numpy as np
from numba import njit
from pyglet.gl import GL_POINTS

import argparse
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax.training import train_state, checkpoints
from flax_rbf import *
from model import WCRBFNet


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


"""
Planner Helpers
"""


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(
    point, radius, trajectory, t=0.0, wrap=False
):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start).astype(
            np.float32
        )  # NOTE: specify type or numba complains

        a = np.dot(V, V)
        b = np.float32(2.0) * np.dot(
            V, start - point
        )  # NOTE: specify type or numba complains
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - np.float32(2.0) * np.dot(start, point)
            - radius * radius
        )
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = (end - start).astype(np.float32)

            a = np.dot(V, V)
            b = np.float32(2.0) * np.dot(
                V, start - point
            )  # NOTE: specify type or numba complains
            c = (
                np.dot(start, start)
                + np.dot(point, point)
                - np.float32(2.0) * np.dot(start, point)
                - radius * radius
            )
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)], dtype=np.float32),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


@jax.jit
def pred_step(state, x):
    output = state.apply_fn(state.params, x)
    return output.flatten()


class IRBFNPlanner:
    """
    Motion planner using IRBFN
    """
    def __init__(self, track):
        self.waypoints = np.stack(
            [track.centerline.xs, track.centerline.ys, track.centerline.vxs, track.centerline.yaws]
        ).T
        self.max_reacquire = 20.0
        self.drawn_waypoints = []

        self.speed = 0.0

        flax.config.update('flax_use_orbax_checkpointing', False)

        config_f = 'configs/big_mpc_20231207_083352.yaml'
        ckpt = 'ckpts/big_mpc_20231207_083352/checkpoint_0'
        with open(config_f, 'r') as f:
            config_dict = yaml.safe_load(f)
        conf = argparse.Namespace(**config_dict)

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

        rng=jax.random.PRNGKey(conf.seed)
        rng, init_rng = jax.random.split(rng)
        params = wcrbf.init(init_rng, jnp.ones((1, conf.in_features)))
        optim = optax.adam(conf.lr)
        state = train_state.TrainState.create(apply_fn=wcrbf.apply, params=params, tx=optim)
        restored_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt, target=state)
        self.state = restored_state

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        # NOTE: specify type or numba complains
        self.waypoints = np.loadtxt(
            conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip
        ).astype(np.float32)

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        points = self.waypoints[:, :2]

        scaled_points = 50.0 * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(
                    1,
                    GL_POINTS,
                    None,
                    ("v3f/stream", [scaled_points[i, 0], scaled_points[i, 1], 0.0]),
                    ("c3B/stream", [183, 193, 222]),
                )
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [
                    scaled_points[i, 0],
                    scaled_points[i, 1],
                    0.0,
                ]

    def _get_current_waypoint(self, waypoints, lookahead_distance, position):
        """
        gets the current waypoint to follow
        """
        wpts = waypoints[:, :2]
        lookahead_distance = np.float32(lookahead_distance)
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            t1 = np.float32(i + t)
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, t1, wrap=True
            )
            if i2 is None:
                return None
            current_waypoint = np.empty((4,), dtype=np.float32)
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, -2]
            # heading
            current_waypoint[3] = waypoints[i2, -1]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            # NOTE: specify type or numba complains
            return waypoints[i, :]
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        *_, i = nearest_point_on_trajectory(position, self.waypoints[:, :2])
        start = self.waypoints[i, :]

        lookahead_distance = min(4.0, max(0.5, self.speed))
        goal = self._get_current_waypoint(self.waypoints, lookahead_distance, position)

        if goal is None:
            return 3.0, 0.0
        
        xs, ys, vs, ts = start
        xg, yg, vg, tg = goal

        rotation = np.array([
            [np.cos(pose_theta), -np.sin(pose_theta)],
            [np.sin(pose_theta),  np.cos(pose_theta)]
        ]).T
        _, ysr = (rotation @ np.array([[xs - pose_x], [ys - pose_y]])).flatten()
        xgr, ygr = (rotation @ np.array([[xg - pose_x], [yg - pose_y]])).flatten()

        if ts < 0.0:
            ts += 2 * np.pi
        if tg < 0.0:
            tg += 2 * np.pi
        if pose_theta < 0.0:
            pose_theta += 2 * np.pi
        
        tsr = ts - pose_theta
        tgr = tg - pose_theta
        if tsr > np.pi:
            tsr -= 2 * np.pi
        if tgr > np.pi:
            tgr -= 2 * np.pi
        if tsr < -np.pi:
            tsr += 2 * np.pi
        if tgr < -np.pi:
            tgr += 2 * np.pi

        ysr = min(0.15, max(-0.15, ysr))

        irbfn_input = np.array([ysr, tsr, 2.0, xgr, ygr, tgr, 2.0]).reshape(1, 7)
        irbfn_output = pred_step(self.state, irbfn_input)
        speed, steering_angle = irbfn_output.flatten()
        self.speed = vgain * speed

        print('lookahead distance:', lookahead_distance)
        print(irbfn_input.round(3))
        print('speed:', self.speed)
        print('steer:', 180 * steering_angle / np.pi)

        return self.speed, steering_angle


def main():
    """
    main entry point
    """

    work = {
        "mass": 3.463388126201571,
        "lf": 0.15597534362552312,
        "tlad": 0.82461887897713965,
        "vgain": 1,
    }  # 0.90338203837889}

    env = gym.make(
        "f110_gym:f110-v0",
        config={
            "map": "Oschersleben",
            "num_agents": 1,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": "speed",
            "model": "st",
            "observation_config": {"type": "kinematic_state"},
            "params": {"mu": 1.0},
        },
        render_mode="human",
    )

    planner = IRBFNPlanner(track=env.track)

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_waypoints(env_renderer)

    env.add_render_callback(render_callback)

    poses = np.array(
        [
            [
                env.track.centerline.xs[0],
                env.track.centerline.ys[0],
                env.track.centerline.yaws[0],
            ]
        ]
    )
    obs, info = env.reset(options={"poses": poses})
    done = False
    env.render()

    laptime = 0.0
    start = time.time()

    while not done:
        agent_id = env.agent_ids[0]
        # print('x',obs[agent_id]["pose_x"])
        # print('y', obs[agent_id]["pose_y"])
        # print('t', obs[agent_id]["pose_theta"])
        speed, steer = planner.plan(
            obs[agent_id]["pose_x"],
            obs[agent_id]["pose_y"],
            obs[agent_id]["pose_theta"],
            work["vgain"],
        )
        obs, step_reward, done, truncated, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render()

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()

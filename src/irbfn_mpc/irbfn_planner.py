import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state, checkpoints
from flax_rbf.flax_rbf import *
from irbfn_mpc.model import WCRBFNet
from irbfn_mpc.planner_utils import intersect_point, nearest_point
from irbfn_mpc.dynamics import integrate_st_mult, dynamic_st_onestep_aux
from irbfn_mpc.bandits import EXP3

import numpy as np
import yaml
import argparse
from f1tenth_gym.envs.track import Track
from typing import List


@jax.jit
def pred_step(state, x):
    y = state.apply_fn(state.params, x)
    return y


class IRBFNPlanner:
    """_summary_"""

    def __init__(self, config: str, ckpt: str, track: Track = None, mirror: bool = False, sv_ind: int = 2):
        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)
        conf = argparse.Namespace(**config_dict)

        self.dyn_params = jnp.array(
            [conf.mu, 1.0489, 0.04712, 0.15875, 0.17145, conf.cs, conf.cs, 0.074, 0.1, 3.2, 9.51, 0.4189, 7.0]
        )

        self.wcrbf = WCRBFNet(
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

        rng = jax.random.PRNGKey(conf.seed)
        self.rng, init_rng = jax.random.split(rng)
        params = self.wcrbf.init(init_rng, jnp.ones((1, conf.in_features)))
        optim = optax.chain(optax.clip_by_global_norm(0.1), optax.adam(conf.lr))
        state = train_state.TrainState.create(
            apply_fn=self.wcrbf.apply, params=params, tx=optim
        )
        self.restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt, target=state
        )

        if track is not None:
            self.waypoints = [
                track.raceline.xs,
                track.raceline.ys,
                np.zeros_like(track.raceline.xs),
                track.raceline.vxs,
                track.raceline.yaws,
                np.zeros_like(track.raceline.xs),
                np.zeros_like(track.raceline.xs),
            ]
            self.ds = track.raceline.ss[1] - track.raceline.ss[0]
        else:
            self.waypoints = None

        self.ref_point = None
        self.pred_x = None

        # if control needs to be mirrored
        self.mirror = mirror
        self.sv_ind = sv_ind

    def _get_current_waypoint(self, lookahead_distance, position):
        """
        Finds the current waypoint on the look ahead circle intersection

        Args:
            lookahead_distance (float): lookahead distance to find next point to track
            position (numpy.ndarray (2, )): current position of the vehicle (x, y)

        Returns:
            current_waypoint (numpy.ndarray (3, )): selected waypoint (x, y, velocity), None if no point is found
        """
        # if lookahead_distance <= self.ds:
        #     lookahead_distance += self.ds
        lookahead_distance = max(lookahead_distance, self.ds + 0.1)
        waypoints = np.array(self.waypoints).T
        nearest_p, nearest_dist, t, i = nearest_point(position, waypoints[:, 0:2])
        if nearest_dist < lookahead_distance:
            self.lookahead_point, self.current_index, t2 = intersect_point(
                position,
                lookahead_distance,
                waypoints[:, 0:2],
                np.float32(i + t),
                wrap=True,
            )
            if self.current_index is None:
                return None
            current_waypoint = waypoints[self.current_index, :]
            current_waypoint[3] = waypoints[i, 3]
            return current_waypoint
        # elif nearest_dist < 200:
        #     return waypoints[i, :]
        else:
            return waypoints[i, :]

    def plan(self, current_state):
        """_summary_

        Parameters
        ----------
        current_state : _type_
            _description_
        """
        if self.waypoints is None:
            raise ValueError(
                "Please set waypoints to track during planner instantiation or when calling plan(), use mpc_prob_solve() if only using a goal state."
            )

        # current state
        x = current_state["pose_x"]
        y = current_state["pose_y"]
        delta = current_state["delta"]
        v = current_state["linear_vel_x"]
        theta = current_state["pose_theta"]
        beta = current_state["beta"]
        angv = current_state["ang_vel_z"]

        # calculate lookahead point based on current v
        v_lookahead = max(v, 0.1)
        la_d = v_lookahead * (5 * 0.1)
        # Ref point is the goal_state, with the velocity from the closest point
        goal_state = self._get_current_waypoint(la_d, np.array([x, y]))
        self.ref_point = goal_state.copy()

        # closest_point = self._get_current_waypoint(
        #     0.0, np.array([current_state["pose_x"], current_state["pose_y"]])
        # )
        # self.ref_point[3] = closest_point[3]

        rot = np.array(
            [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]]
        )
        goal_local = np.dot(rot, (self.ref_point[:2] - np.array([x, y])))
        goal_theta = self.ref_point[2] - theta
        
        # input: [v, x_g, y_g, t_g, v_g, beta, angv]
        goal_needs_mirror = goal_local[1] < 0
        rbf_in = jnp.array(
            [
                [
                    v,
                    goal_local[0],
                    -goal_local[1] if goal_needs_mirror else goal_local[1],
                    -goal_theta % np.pi if goal_needs_mirror else goal_theta % np.pi,
                    self.ref_point[3],
                    beta,
                    angv,
                ]
            ]
        )
        # print(f"original local goal {goal_local}")
        # print(f"needs mirror {goal_needs_mirror}")
        # print(f"after mirror {rbf_in[0, 1], rbf_in[0, 2], rbf_in[0, 3]}")
        pred_u = pred_step(self.restored_state, rbf_in)
        # mirror inputs if only half dataset
        if self.mirror and goal_needs_mirror:
            pred_u = pred_u.at[0, self.sv_ind:].set(-pred_u[0, self.sv_ind:])
        states = jnp.array([[x, y, delta, v, theta, angv, beta]])
        x_and_pred_u = jnp.hstack((states, pred_u))
        if self.sv_ind == 5:
            self.pred_x = integrate_st_mult(x_and_pred_u, self.dyn_params)
        elif self.sv_ind ==- 2:
            self.pred_x = dynamic_st_onestep_aux(x_and_pred_u, self.dyn_params)

        # print(f"cv: {v}, gx: {rbf_in[0, 1]}, gy: {rbf_in[0, 2]}, gt: {rbf_in[0, 3]}, gv: {rbf_in[0, 4]}, beta: {rbf_in[0, 5]}, angv: {rbf_in[0, 6]}")

        return pred_u[0, 0], pred_u[0, self.sv_ind]

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        points = np.array(self.waypoints).T[:, :2]
        e.render_closed_lines(points, color=(128, 0, 0), size=1)

    def render_goal_state(self, e):
        """
        update goal state being drawn by EnvRenderer
        """
        if self.ref_point is not None:
            points = self.ref_point[:2][None]
            e.render_points(points, color=(0, 128, 0), size=3)

    def render_planner_sol(self, e):
        """
        Callback to render the lookahead point.
        """
        if self.pred_x is not None:
            for traj in self.pred_x:
                e.render_lines(np.array(traj[:, 0:2]), color=(0, 0, 128), size=2)


class AdaptiveIRBFNPlanner:
    def __init__(self, config_list: List[str], ckpt_list: List[str], track: Track = None, mirror: bool = False, sv_ind: int = 2):
        # number of arms
        self.num_arms = len(config_list)
        assert len(config_list) == len(ckpt_list), f"Number of configs and ckpts must match. Got {len(config_list)} configs and {len(ckpt_list)} ckpts."

        self.conf_list = []
        self.dyn_params_list = []
        self.network_list = []
        self.restored_state_list = []

        for (config, ckpt) in zip(config_list, ckpt_list):
            with open(config, "r") as f:
                config_dict = yaml.safe_load(f)
            conf = argparse.Namespace(**config_dict)
            self.conf_list.append(conf)

            dyn_params = jnp.array(
                [conf.mu, 1.0489, 0.04712, 0.15875, 0.17145, conf.cs, conf.cs, 0.074, 0.1, 3.2, 9.51, 0.4189, 7.0]
            )
            self.dyn_params_list.append(dyn_params)

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
            self.network_list.append(wcrbf)

            rng = jax.random.PRNGKey(conf.seed)
            self.rng, init_rng = jax.random.split(rng)
            params = wcrbf.init(init_rng, jnp.ones((1, conf.in_features)))
            optim = optax.chain(optax.clip_by_global_norm(0.1), optax.adam(conf.lr))
            state = train_state.TrainState.create(
                apply_fn=wcrbf.apply, params=params, tx=optim
            )
            self.restored_state = checkpoints.restore_checkpoint(
                ckpt_dir=ckpt, target=state
            )

        if track is not None:
            self.waypoints = [
                track.centerline.xs,
                track.centerline.ys,
                np.zeros_like(track.centerline.xs),
                track.centerline.vxs,
                track.centerline.yaws,
                np.zeros_like(track.centerline.xs),
                np.zeros_like(track.centerline.xs),
            ]
            self.ds = track.centerline.ss[1] - track.centerline.ss[0]
        else:
            self.waypoints = None

        self.ref_point = None
        self.pred_x = None

        # if control needs to be mirrored
        self.mirror = mirror
        self.sv_ind = sv_ind

    def _get_current_waypoint(self, lookahead_distance, position):
        """
        Finds the current waypoint on the look ahead circle intersection

        Args:
            lookahead_distance (float): lookahead distance to find next point to track
            position (numpy.ndarray (2, )): current position of the vehicle (x, y)

        Returns:
            current_waypoint (numpy.ndarray (3, )): selected waypoint (x, y, velocity), None if no point is found
        """
        # if lookahead_distance <= self.ds:
        #     lookahead_distance += self.ds
        lookahead_distance = max(lookahead_distance, 1.5)
        waypoints = np.array(self.waypoints).T
        nearest_p, nearest_dist, t, i = nearest_point(position, waypoints[:, 0:2])
        if nearest_dist < lookahead_distance:
            self.lookahead_point, self.current_index, t2 = intersect_point(
                position,
                lookahead_distance,
                waypoints[:, 0:2],
                np.float32(i + t),
                wrap=True,
            )
            if self.current_index is None:
                return None
            current_waypoint = waypoints[self.current_index, :]
            current_waypoint[3] = waypoints[i, 3]
            return current_waypoint
        # elif nearest_dist < 200:
        #     return waypoints[i, :]
        else:
            return waypoints[i, :]

    def plan(self, current_state):
        """_summary_

        Parameters
        ----------
        current_state : _type_
            _description_
        """
        if self.waypoints is None:
            raise ValueError(
                "Please set waypoints to track during planner instantiation or when calling plan(), use mpc_prob_solve() if only using a goal state."
            )

        # current state
        x = current_state["pose_x"]
        y = current_state["pose_y"]
        delta = current_state["delta"]
        v = current_state["linear_vel_x"]
        theta = current_state["pose_theta"]
        beta = current_state["beta"]
        angv = current_state["ang_vel_z"]

        # calculate lookahead point based on current v
        v_lookahead = max(v, 0.1)
        la_d = v_lookahead * (5 * 0.1)
        # Ref point is the goal_state, with the velocity from the closest point
        goal_state = self._get_current_waypoint(la_d, np.array([x, y]))
        self.ref_point = goal_state.copy()

        # closest_point = self._get_current_waypoint(
        #     0.0, np.array([current_state["pose_x"], current_state["pose_y"]])
        # )
        # self.ref_point[3] = closest_point[3]

        rot = np.array(
            [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]]
        )
        goal_local = np.dot(rot, (self.ref_point[:2] - np.array([x, y])))
        goal_theta = self.ref_point[2] - theta
        
        # input: [v, x_g, y_g, t_g, v_g, beta, angv]
        goal_needs_mirror = goal_local[1] < 0
        rbf_in = jnp.array(
            [
                [
                    v,
                    goal_local[0],
                    -goal_local[1] if goal_needs_mirror else goal_local[1],
                    -goal_theta % np.pi if goal_needs_mirror else goal_theta % np.pi,
                    self.ref_point[3],
                    beta,
                    angv,
                ]
            ]
        )
        # print(f"original local goal {goal_local}")
        # print(f"needs mirror {goal_needs_mirror}")
        # print(f"after mirror {rbf_in[0, 1], rbf_in[0, 2], rbf_in[0, 3]}")
        pred_u = pred_step(self.restored_state, rbf_in)
        # mirror inputs if only half dataset
        if self.mirror and goal_needs_mirror:
            pred_u = pred_u.at[0, self.sv_ind:].set(-pred_u[0, self.sv_ind:])
        states = jnp.array([[x, y, delta, v, theta, angv, beta]])
        x_and_pred_u = jnp.hstack((states, pred_u))
        if self.sv_ind == 5:
            self.pred_x = integrate_st_mult(x_and_pred_u, self.dyn_params)
        elif self.sv_ind ==- 2:
            self.pred_x = dynamic_st_onestep_aux(x_and_pred_u, self.dyn_params)

        return pred_u[0, 0], pred_u[0, self.sv_ind]

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        points = np.array(self.waypoints).T[:, :2]
        e.render_closed_lines(points, color=(128, 0, 0), size=1)

    def render_goal_state(self, e):
        """
        update goal state being drawn by EnvRenderer
        """
        if self.ref_point is not None:
            points = self.ref_point[:2][None]
            e.render_points(points, color=(0, 128, 0), size=3)

    def render_planner_sol(self, e):
        """
        Callback to render the lookahead point.
        """
        if self.pred_x is not None:
            for traj in self.pred_x:
                e.render_lines(np.array(traj[:, 0:2]), color=(0, 0, 128), size=2)
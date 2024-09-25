import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import jax
import jax.numpy as jnp

from irbfn_mpc.planner_utils import intersect_point, nearest_point
from irbfn_mpc.bandits import EXP3
from irbfn_mpc.dynamics import integrate_st_mult, integrate_frenet_mult
from irbfn_mpc.nonlinear_dmpc_frenet import mpc_config

import numpy as np
from typing import Sequence
from f1tenth_gym.envs.track import Track

import scipy.spatial as ss


@jax.jit
def get_closest_ind(input, all_inputs):
    diff = all_inputs - input
    ind = jnp.argmin(jnp.linalg.norm(diff, axis=1))
    return ind


class ExplicitPlanner:
    def __init__(
        self,
        npz_path: str = "/data/tables/8v_19x_19y_64t_8vgoal_6beta_12angvz_mu1.0_cs5.0_sorted.npz",
        track: Track = None,
        mirror: bool = False,
    ):
        data = np.load(npz_path)
        inputs, outputs = data["inputs"], data["outputs"]

        # TODO: these are hardcoded for now
        # self.inputs = inputs.reshape((5, 19, 24, 18, 5, 6, 12, -1))
        self.input_keys = []
        for ind in range(inputs.shape[1]):
            self.input_keys.append(np.unique(inputs[:, ind]))
        self.outputs = outputs.reshape((8, 19, 19, 64, 8, 6, 12, -1))

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

        # if control needs to be mirrored
        self.mirror = mirror
        self.pred_x = None

        self.dyn_params = np.array(
            [
                1.0,
                1.0489,
                0.04712,
                0.15875,
                0.17145,
                5.0,
                5.0,
                0.074,
                0.1,
                3.2,
                9.51,
                0.4189,
                7.0,
            ]
        )

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
        lookahead_distance = max(lookahead_distance, self.ds + 0.05)
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

        rot = np.array(
            [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]]
        )
        goal_local = np.dot(rot, (self.ref_point[:2] - np.array([x, y])))
        goal_theta = self.ref_point[2] - theta

        # input: [v, x_g, y_g, t_g, v_g, beta, angv]
        goal_needs_mirror = goal_local[1] < 0

        lookup = [
            v,
            goal_local[0],
            -goal_local[1] if goal_needs_mirror else goal_local[1],
            -goal_theta % np.pi if goal_needs_mirror else goal_theta % np.pi,
            self.ref_point[3],
            beta,
            angv,
        ]

        closest_ind = []
        for val_ind, val in enumerate(lookup):
            closest_ind.append(
                min(
                    self.outputs.shape[val_ind] - 1,
                    np.searchsorted(self.input_keys[val_ind], val, side="right"),
                )
            )

        # closest_ind = get_closest_ind(lookup, self.inputs)
        pred_u = self.outputs[*closest_ind].flatten()
        # mirror inputs if only half dataset
        if self.mirror and goal_needs_mirror:
            pred_u[5:] = -pred_u[5:]

        states = np.array([x, y, delta, v, theta, angv, beta])
        x_and_pred_u = np.hstack((states[None,], pred_u[None,]))
        self.pred_x = integrate_st_mult(x_and_pred_u, self.dyn_params)

        return pred_u[0], pred_u[5]

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


class ExplicitFrenetPlanner:
    def __init__(
        self,
        npz_path: str = "/data/tables/frenet/constraints_12ey_7delta_11vxcar_11vycar_5vxgoal_11wz_13epsi_3curv_mu1.0999999999999999_cs5.0_wider_angles_sorted.npz",
        track: Track = None,
    ):
        data = np.load(npz_path)
        inputs, outputs = data["inputs"], data["outputs"]
        # build kd tree
        self.kdtree = ss.KDTree(inputs)

        # TODO: these are hardcoded for now
        # self.inputs = inputs.reshape((5, 19, 24, 18, 5, 6, 12, -1))
        self.input_keys = []
        for ind in range(inputs.shape[1]):
            self.input_keys.append(np.unique(inputs[:, ind]))
        self.outputs_flat = outputs.copy()
        self.inputs_flat = inputs.copy()
        self.outputs = outputs.reshape((12, 7, 11, 11, 5, 11, 13, 3, 5, -1))
        self.inputs = inputs.reshape((12, 7, 11, 11, 5, 11, 13, 3, -1))

        if track is not None:
            self.waypoints = [
                track.raceline.xs.copy(),
                track.raceline.ys.copy(),
                track.raceline.yaws.copy(),
                track.raceline.vxs.copy(),
                track.raceline.ks.copy(),
            ]
            self.track = track
        else:
            self.waypoints = None

        self.oa = None
        self.odelta_v = None
        self.ox = None
        self.oy = None
        self.ref_path = None

        self.config = mpc_config()

        self.drawn_waypoints = []
        self.mpc_render = None
        self.goal_state_render = None
        self.waypoint_render = None

        self.dyn_params = np.array(
            [
                1.0,
                1.0489,
                0.04712,
                0.15875,
                0.17145,
                5.0,
                5.0,
                0.074,
                0.1,
                3.2,
                9.51,
                0.4189,
                7.0,
            ]
        )
        self.lookup_keys = ["ey", "delta", "vx_car", "vy_car", "vx_goal", "wz", "epsi", "curv"]

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp, ckap):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK + 1, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(
            np.array([state["pose_x"], state["pose_y"]]), np.array([cx, cy]).T
        )

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]

        ref_traj[3, 0] = sp[ind]
        ref_traj[4, 0] = cyaw[ind]

        ref_traj[5, :] = ckap[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state["linear_vel_x"]) * self.config.DTK
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[3, :] = sp[ind_list]
        cyaw[cyaw - state["pose_theta"] > 4.5] = np.abs(
            cyaw[cyaw - state["pose_theta"] > 4.5] - (2 * np.pi)
        )
        cyaw[cyaw - state["pose_theta"] < -4.5] = np.abs(
            cyaw[cyaw - state["pose_theta"] < -4.5] + (2 * np.pi)
        )
        ref_traj[4, :] = cyaw[ind_list]

        return ref_traj

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

        self.ref_path = self.calc_ref_trajectory(
            current_state,
            self.waypoints[0],
            self.waypoints[1],
            self.waypoints[2],
            self.waypoints[3],
            self.waypoints[4],
        )

        s, ey, epsi = self.track.cartesian_to_frenet(
            current_state["pose_x"],
            current_state["pose_y"],
            current_state["pose_theta"],
        )

        # input: [ey, delta, vx_car, vy_car, vx_goal, wz, epsi, curv]
        goal_needs_mirror = ey < -0.05

        lookup = np.array(
            [
                -ey if goal_needs_mirror else ey,
                current_state["delta"],
                current_state["linear_vel_x"],
                -current_state["linear_vel_y"] if goal_needs_mirror else current_state["linear_vel_y"],
                self.ref_path[3][-1],
                -current_state["ang_vel_z"] if goal_needs_mirror else current_state["ang_vel_z"],
                -epsi if goal_needs_mirror else epsi,
                self.ref_path[5][0],
            ]
        )

        for (l, keys) in zip(lookup, self.lookup_keys):
            print(f"{keys} lookup: {l}")

        # closest_ind = []
        # for val_ind, val in enumerate(lookup):
        #     closest_ind.append(
        #         min(
        #             self.outputs.shape[val_ind] - 1,
        #             np.searchsorted(self.input_keys[val_ind], val, side="left"),
        #         )
        #     )

        # closest_ind = get_closest_ind(lookup, self.inputs)
        distance, closest_ind = self.kdtree.query(lookup)
        print(f"kdtree lookup distance {distance}")
        print(f"kdtree lookup inds {closest_ind}")
        pred_u = self.outputs_flat[closest_ind]
        print(f"looked up input: {self.inputs_flat[closest_ind]}")
        print(f"accl seq:   {pred_u[:, 0]}")
        print(f"steerv seq: {pred_u[:, 1]}")
        if np.any(pred_u == -999):
            print("invlaid output")
            return 0.0, 0.0
        # mirror inputs if only half dataset
        if goal_needs_mirror:
            pred_u[:, 1] = -pred_u[:, 1]

        states = np.array(
            [
                s,
                ey,
                current_state["delta"],
                current_state["linear_vel_x"],
                current_state["linear_vel_y"],
                current_state["ang_vel_z"],
                epsi,
                self.ref_path[5][0],
            ]
        )
        x_and_pred_u = np.hstack((states[None,], pred_u.flatten("F")[None,]))
        pred_x = integrate_frenet_mult(x_and_pred_u, self.dyn_params)
        self.ox = np.array(pred_x[0, :, 0])
        self.oy = np.array(pred_x[0, :, 1])
        for i, (s, ey) in enumerate(zip(self.ox, self.oy)):
            curr_x, curr_y, _ = self.track.frenet_to_cartesian(s, ey, 0.0)
            self.ox[i] = curr_x
            self.oy[i] = curr_y

        return pred_u[0, 0], pred_u[0, 1]

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        points = np.array(self.waypoints).T[:, :2]
        if self.waypoint_render is None:
            self.waypoint_render = e.render_closed_lines(
                points, color=(128, 0, 0), size=1
            )
        else:
            self.waypoint_render.setData(points)

    def render_local_plan(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        if self.ref_path is not None:
            points = self.ref_path[:2].T
            e.render_lines(points, color=(0, 128, 0), size=2)

    def render_mpc_sol(self, e):
        """
        Callback to render the lookahead point.
        """
        if self.ox is not None and self.oy is not None:
            points = np.array([self.ox, self.oy]).T
            if self.mpc_render is None:
                self.mpc_render = e.render_lines(points, color=(0, 0, 128), size=2)
            else:
                self.mpc_render.setData(points)


class AdaptiveExplicitPlanner:
    def __init__(
        self, npz_path_list: Sequence[str], track: Track = None, mirror: bool = False
    ):
        self.inputs_list = []
        self.outputs_list = []

        for npz_path in npz_path_list:
            data = np.load(npz_path)
            inputs, outputs = data["inputs"], data["outputs"]
            self.inputs_list.append(inputs)
            self.outputs_list.append(outputs)

        self.exp3 = EXP3(n=len(npz_path_list), gamma=0.8)

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

        # if control needs to be mirrored
        self.mirror = mirror
        self.pred_x = None

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
        lookahead_distance = max(lookahead_distance, 0.5)
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

        rot = np.array(
            [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]]
        )
        goal_local = np.dot(rot, (self.ref_point[:2] - np.array([x, y])))
        goal_theta = self.ref_point[2] - theta

        # input: [v, x_g, y_g, t_g, v_g, beta, angv]
        goal_needs_mirror = goal_local[1] < 0

        lookup = jnp.array(
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

        closest_ind = get_closest_ind(lookup, self.inputs)
        pred_u = self.outputs[closest_ind]

        # mirror inputs if only half dataset
        if self.mirror and goal_needs_mirror:
            pred_u = pred_u.at[0, 5:].set(-pred_u[0, 5:])

        states = jnp.array([[x, y, delta, v, theta, angv, beta]])
        x_and_pred_u = jnp.hstack((states, pred_u))
        # if self.sv_ind == 5:
        self.pred_x = integrate_st_mult(x_and_pred_u, self.dyn_params)

        return pred_u[0, 0], pred_u[0, 5]

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

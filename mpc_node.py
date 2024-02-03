#!/usr/bin/env python3
import os
import math
from dataclasses import dataclass, field

import cvxpy
import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 8  # finite time horizon length kinematic

    # ---------------------------------------------------
    Rk: npt.NDArray = field(
        default_factory=lambda: np.diag([0.01, 5.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: npt.NDArray = field(
        default_factory=lambda: np.diag([0.05, 50.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: npt.NDArray = field(
        default_factory=lambda: np.diag([5.0, 5.0, 10.0, 1.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, v, yaw]
    Qfk: npt.NDArray = field(
        default_factory=lambda: np.diag([15.0, 15.0, 10.0, 1.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, v, yaw]
    # ---------------------------------------------------

    DTK: float = 0.05  # time step [s] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 10.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 10.0  # maximum acceleration [m/ss]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0


class MPC():
    """
    Implement Kinematic MPC on the car
    """
    def __init__(self):
        self.config = mpc_config()
        self.odelta_v = None
        self.oa = None

        # variables tracked by odometry
        self.velocity = 0.0
        self.steering_angle = 0.0

        # initialize MPC problem
        self.mpc_prob_init()

    def get_controls(self, ref_path, path_predict, v):
        # x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]
        x0 = [0.0, 0.0, v, 0.0]

        (
            self.oa,
            self.odelta_v,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, path_predict, x0)

        if ox is None or oy is None:
            return

        steer_output = self.odelta_v[0]
        speed_output = x0[2] + self.oa[0] * self.config.DTK

        return speed_output, steer_output

    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable((self.config.NXK, self.config.TK + 1))
        # Control Input vector
        self.uk = cvxpy.Variable((self.config.NU, self.config.TK))
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # reference trajectory speed maximum
        self.ref_traj_v_max = cvxpy.Parameter((self.config.TK + 1,))
        self.ref_traj_v_max.value = np.zeros((self.config.TK + 1,))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # --------------------------------------------------------
        # Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective = cvxpy.quad_form(cvxpy.vec(self.uk), R_block)

        # Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)

        # Objective part 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(
            cvxpy.vec(self.uk[:, 1:] - self.uk[:, :-1]), Rd_block
        )
        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(path_predict[2, t], path_predict[3, t], 0.0)
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------

        # get the needed dimensions from the state and input vectors
        velocity = self.xk[2, :]
        acceleration = self.uk[0, :]
        steering_angle = self.uk[1, :]

        # Constraint part 1:
        #     Add dynamics constraints to the optimization problem
        #     This constraint should be based on a few variables:
        #     self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_
        constraints.append(
            cvxpy.vec(self.xk[:, 1:])
            == self.Ak_ @ cvxpy.vec(self.xk[:, :-1])
            + self.Bk_ @ cvxpy.vec(self.uk)
            + self.Ck_
        )

        # Constraint part 2:
        #     Add constraints on steering, change in steering angle
        #     cannot exceed steering angle speed limit. Should be based on:
        #     self.uk, self.config.MAX_DSTEER, self.config.DTK
        steering_angle_change = steering_angle[1:] - steering_angle[:-1]
        constraints.append(
            cvxpy.norm_inf(steering_angle_change)
            <= self.config.MAX_DSTEER * self.config.DTK  # type: ignore
        )

        # Constraint part 3:
        #     Add constraints on upper and lower bounds of states and inputs
        #     and initial state constraint, should be based on:
        #     self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #     self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        constraints.append(cvxpy.vec(self.xk[:, 0]) == self.x0k)
        constraints.append(cvxpy.max(steering_angle) <= self.config.MAX_STEER)  # type: ignore
        constraints.append(cvxpy.min(steering_angle) >= self.config.MIN_STEER)  # type: ignore
        constraints.append(cvxpy.norm_inf(acceleration) <= self.config.MAX_ACCEL)  # type: ignore

        constraints.append(cvxpy.max(velocity) <= self.config.MAX_SPEED)  # type: ignore
        constraints.append(cvxpy.min(velocity) >= self.config.MIN_SPEED)  # type: ignore
        # constraints.append(velocity <= self.ref_traj_v_max)  # type: ignore

        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0
        self.ref_traj_v_max.value = ref_traj[2]

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(path_predict[2, t], path_predict[3, t], 0.0)
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        try:
            self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

            if (
                self.MPC_prob.status == cvxpy.OPTIMAL
                or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
            ):
                ox = np.array(self.xk.value[0, :]).flatten()
                oy = np.array(self.xk.value[1, :]).flatten()
                ov = np.array(self.xk.value[2, :]).flatten()
                oyaw = np.array(self.xk.value[3, :]).flatten()
                oa = np.array(self.uk.value[0, :]).flatten()
                odelta = np.array(self.uk.value[1, :]).flatten()

            else:
                oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

            return oa, odelta, ox, oy, oyaw, ov

        except cvxpy.error.SolverError:
            return None, None, None, None, None, None

    def linear_mpc_control(self, ref_path, path_predict, x0):
        """
        MPC control with updating operational point iteratively
        :param ref_path: reference trajectory in T steps
        :param path_predict: predicted vehicle motion for T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict


def solve_mpc(start, goal, v_start, v_goal, path_predict, ref_path):
    output = []
    car_mpc = MPC()
    for v_start_i in v_start:
        for v_goal_i in v_goal:
            reference_path = np.insert(ref_path, 2, v_goal_i, axis=1).T
            predicted_path = np.insert(path_predict, 2, v_start_i, axis=1).T

            solution = car_mpc.get_controls(reference_path, predicted_path, v_start_i)
            if solution is None:
                continue

            speed, steer = solution

            output.append([start[1], start[2], v_start_i, goal[0], goal[1], goal[2], v_goal_i, speed, steer])

    return np.array(output)


def main(args=None):
    jobs = 45

    ref_data = np.load('reference_trajectory_table_lite.npz')
    goals = ref_data['goals']
    ref_states = ref_data['ref_states']
    goal_lut = {tuple(goal): ref_state for goal, ref_state in tqdm(zip(goals, ref_states), total=len(goals))}

    raw_data = np.load('raw_trajectory_table_lite.npz')
    states = raw_data['states']
    x_start, y_start, t_start = raw_data['x_start'], raw_data['y_start'], raw_data['t_start']
    x_goal, y_goal, t_goal = raw_data['x_goal'], raw_data['y_goal'], raw_data['t_goal']
    v_start, v_goal = raw_data['v_start'], raw_data['v_goal']

    x_startm, y_startm, t_startm, x_goalm, y_goalm, t_goalm, = np.meshgrid(
        x_start, y_start, t_start, x_goal, y_goal, t_goal, indexing='ij'
    )

    start = np.stack((x_startm, y_startm, t_startm), axis=-1)
    start_flat = start.reshape((-1, 3))  # columns are x, y, theta

    goal = np.stack((x_goalm, y_goalm, t_goalm), axis=-1)
    goal_flat = goal.reshape((-1, 3))  # columns are x, y, theta

    # 7 inputs: y_start, t_start, v_start, x_goal, y_goal, t_goal, v_goal
    # 2 outputs: speed, steer
    limit = 200000
    start = 1400000
    end = 1600000
    max_end = len(states)
    while end < max_end:
        start += limit
        end += limit
        end = min(end, max_end)
        print('Current Start:', start)
        print('Current End', end)
        table = Parallel(n_jobs=jobs)(
            delayed(solve_mpc)(s_i, g_i, v_start, v_goal, goal_lut[tuple(g_i)][:, :3], r_i[:, :3])
            for s_i, g_i, r_i in tqdm(zip(start_flat[start:end], goal_flat[start:end], states[start:end]), total=len(states[start:end]))
        )
        table = np.concatenate(table)

        np.savez(
            f'explicit_mpc_table_{start}_{end}.npz',
            table=table
        )


if __name__ == "__main__":
    main()

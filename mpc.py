# MIT License

# Copyright (c) 2023 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author: Hongrui Zheng
# Last Modified: 08/23/2023
# MPC trajectory tracker for 2D kinematic bicycle in JAX

import os
from functools import partial

import jax
import jax.numpy as jnp
from trajax import integrators
from trajax.experimental.sqp import shootsqp, util
from utils import nearest_point_on_trajectory

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_ENABLE_X64"] = "true"


class JaxMPC:
    # constants
    # wheelbase
    wb = 0.33
    # sampling time
    dt = 0.05
    # state space size, control input space size, horizon
    n, m, T = (4, 2, 8)

    # indices of state corresponding to S1 sphere constraints
    s1_indices = (2,)
    state_wrap = util.get_s1_wrapper(s1_indices)

    # control bounds
    control_bounds = (jnp.array([-jnp.pi / 3.0, -6.0]), jnp.array([jnp.pi / 3.0, 6.0]))

    # cost function constants
    # control effort
    R = jnp.diag(jnp.array([0.01, 5.0]))
    # control smoothness
    Rd = jnp.diag(jnp.array([0.05, 50.0]))
    # tracking cost
    Q = jnp.diag(jnp.array([5.0, 5.0, 10.0, 1.0]))
    # terminal cost
    Q_T = jnp.diag(jnp.array([15.0, 15.0, 10.0, 1.0]))

    def __init__(self, track) -> None:
        self.waypoints = jnp.stack(
            [track.raceline.xs, track.raceline.ys, track.raceline.vxs]
        ).T

    # kinematic bicycle model
    # x = [x, y, theta, v]
    # u = [kappa, accl]
    # kappa = tan(steer)/wb ~= steer/wb
    @staticmethod
    def car_ode(x, u, t):
        return jnp.array(
            [x[3] * jnp.sin(x[2]), x[3] * jnp.cos(x[2]), x[3] * u[0], u[1]]
        )

    # cost function
    # TODO: add intermediate costs
    @partial(jax.jit, static_argnums=(0,))
    def cost(self, x, u, t, goal):
        stage_cost = self.dt * jnp.vdot(u, self.R @ u)
        delta = self.state_wrap(x - goal)
        term_cost = jnp.vdot(delta, self.Q_T @ delta)
        return jnp.where(t == self.T, term_cost, stage_cost)

    # state constraint
    # NOTE: don't have to worry about obs here so always >0
    @partial(jax.jit, static_argnums=(0,))
    def state_constraint(self, x, t):
        return 1.0

    def prob_init(self):
        dynamics = integrators.euler(self.car_ode, dt=self.dt)

        # define solver
        solver_options = dict(
            method=shootsqp.SQP_METHOD.SENS,
            ddp_options={"ddp_gamma": 1e-4},
            hess="full",
            verbost=True,
            max_iter=100,
            ls_eta=0.49,
            ls_beta=0.8,
            primal_tol=1e-3,
            dual_tol=1e-3,
            stall_check="abs",
            debug=False,
        )
        self.solver = shootsqp.ShootSQP(
            self.n,
            self.m,
            self.T,
            dynamics,
            self.cost,
            self.control_bounds,
            self.state_constraint,
            s1_ind=self.s1_indices,
            **solver_options
        )

    def set_init_state(self, x0, u0, X0):
        self.solver.opt.proj_init = True
        self.solver.opt.max_iter = 1
        _ = self.solver.solve(x0, u0, X0)

    def prob_solve(self, state, u, last_traj):
        if not self.solver.opt.max_iter == 100:
            self.solver.opt.max_iter = 100
        soln = self.solver.solve(state, u, last_traj)
        U, X = soln.primals
        return U, X
    
    def ref_traj(self, x, y, theta):
        _, _, _, nearest_idx = nearest_point_on_trajectory(jnp.array([x, y]), self.waypoints)

    def plan(self, x, y, theta):
        pass

if __name__ == "__main__":
    import gymnasium as gym
    import numpy as np

    env = gym.make("f110_gym:f110-v0", config={"map": "Spielberg"})
    tracker = JaxMPC(env.track)

    poses = np.array(
        [
            [
                env.track.raceline.xs[0],
                env.track.raceline.ys[0],
                env.track.raceline.yaws[0],
            ]
        ]
    )
    obs, info = env.reset(options={"poses": poses})
    done = False
    agent_id = env.agent_ids[0]

    while not done:
        speed, steer = tracker.plan(
            obs[agent_id]["pose_x"],
            obs[agent_id]["pose_y"],
            obs[agent_id]["pose_theta"],
        )
        obs, step_reward, done, truncated, info = env.step(np.array([[steer, speed]]))
        env.render()

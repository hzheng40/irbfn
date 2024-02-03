import os
import numpy as np
import pyclothoids
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import integrate_path_mult


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


y_start_min = 0.0
y_start_max = 0.5
dy_start = 0.1

t_start_min = -1.0
t_start_max = 1.0
dt_start = 0.04

v_start_min = 0.1
v_start_max = 10.1
dv_start = 1.0


x_goal_min = 0.01
x_goal_max = 4.51
dx_goal = 0.1

y_goal_min = -4.2
y_goal_max = 4.2
dy_goal = 0.1

t_goal_min = -2.0
t_goal_max = 2.0
dt_goal = 0.04

v_goal_min = 0.1
v_goal_max = 10.1
dv_goal = 1.0


x_start = np.array([0])
y_start = np.arange(y_start_min, y_start_max + dy_start, dy_start)
t_start = np.arange(t_start_min, t_start_max + dt_start, dt_start)
v_start = np.arange(v_start_min, v_start_max + dv_start, dv_start)


x_goal = np.arange(x_goal_min, x_goal_max + dx_goal, dx_goal)
y_goal = np.arange(y_goal_min, y_goal_max + dy_goal, dy_goal)
t_goal = np.arange(t_goal_min, t_goal_max + dt_goal, dt_goal)
v_goal = np.arange(v_goal_min, v_goal_max + dv_goal, dv_goal)


goalm = np.meshgrid(x_goal, y_goal, t_goal, indexing='ij')
goals = np.stack(goalm, axis=-1).reshape((-1, 3))


x_startm, y_startm, t_startm, x_goalm, y_goalm, t_goalm, = np.meshgrid(
    x_start, y_start, t_start, x_goal, y_goal, t_goal, indexing='ij'
)

start = np.stack((x_startm, y_startm, t_startm), axis=-1)
start_flat = start.reshape((-1, 3))  # columns are x, y, theta

goal = np.stack((x_goalm, y_goalm, t_goalm), axis=-1)
goal_flat = goal.reshape((-1, 3))  # columns are x, y, theta


def gen_traj(start, goal):
    clothoid = pyclothoids.Clothoid.G1Hermite(start[0], start[1], start[2], goal[0], goal[1], goal[2])
    k0 = clothoid.Parameters[3]
    dk = clothoid.Parameters[4]
    s = clothoid.Parameters[5]
    k1 = k0 + (1/3)*s*dk
    k2 = k0 + (2/3)*s*dk
    k3 = k0 + (3/3)*s*dk
    return [k0, k1, k2, k3, s]


def transform_traj(traj, start):
    x, y, t = start

    R = np.array([
        [np.cos(t), -np.sin(t)],
        [np.sin(t),  np.cos(t)]
    ])

    traj[:, :2] = (R @ traj[:, :2].T).T
    traj[:, 0] += x
    traj[:, 1] += y

    return traj


def main(args=None):
    jobs = 45
    begin = 0
    end = len(start_flat) // 5

    start_slice = start_flat[begin:end]
    goal_slice = goal_flat[begin:end]

    table = Parallel(n_jobs=jobs)(
        delayed(gen_traj)(start_i, goal_i) for start_i, goal_i in tqdm(zip(start_slice, goal_slice), total=len(start_slice))
    )
    table = np.array(table)

    all_states = integrate_path_mult(table)
    all_states = np.array(all_states)

    init_states = start_slice
    transformed_states = Parallel(n_jobs=jobs)(
        delayed(transform_traj)(t_i, s_i) for t_i, s_i in tqdm(zip(all_states, init_states), total=len(init_states))
    )
    all_states = np.array(transformed_states)

    np.savez(
        'raw_trajectory_table_1.npz',
        table=table,
        states=all_states,
        x_start=x_start,
        y_start=y_start,
        t_start=t_start,
        v_start=v_start,
        x_goal=x_goal,
        y_goal=y_goal,
        t_goal=t_goal,
        v_goal=v_goal,
        goals=goals,
        indices=np.array([begin, end])
    )


if __name__ == '__main__':
    main()


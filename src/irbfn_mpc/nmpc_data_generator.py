import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .nonlinear_mpc import mpc_solve


""" Input state mesh grid parameters. """
# car velocity [23]
v_car_min = 0.0
v_car_max = 7.0
dv_car =    0.3

# goal x-position [35]
x_goal_min = 0.0
x_goal_max = 3.5
dx_goal =    0.1

# goal y-position [35]
y_goal_min = 0.0
y_goal_max = 3.5
dy_goal =    0.1

# goal heading [62]
t_goal_min = -3.1
t_goal_max =  3.1
dt_goal =     0.1

# goal velocity [23]
v_goal_min = 0.0
v_goal_max = 7.0
dv_goal =    0.3

n_jobs = 45  # number of jobs


def train_data():
    # generate input state mesh grid
    print('Generating input state mesh grid')
    v_car = np.arange(v_car_min, v_car_max + dv_car, dv_car)
    x_goal = np.arange(x_goal_min, x_goal_max + dx_goal, dx_goal)
    y_goal = np.arange(y_goal_min, y_goal_max + dy_goal, dy_goal)
    t_goal = np.arange(t_goal_min, t_goal_max + dt_goal, dt_goal)
    v_goal = np.arange(v_goal_min, v_goal_max + dv_goal, dv_goal)
    v_car_m, x_goal_m, y_goal_m, t_goal_m, v_goal_m = np.meshgrid(
        v_car, x_goal, y_goal, t_goal, v_goal, indexing='ij'
    )
    v_car = v_car_m.flatten()
    x_goal = x_goal_m.flatten()
    y_goal = y_goal_m.flatten()
    t_goal = t_goal_m.flatten()
    v_goal = v_goal_m.flatten()
    print(f'Input state mesh grid generation completed: {len(v_car)} samples')

    table = Parallel(n_jobs=n_jobs)(
        delayed(mpc_solve)(vc, xg, yg, tg, vg)
        for vc, xg, yg, tg, vg in tqdm(zip(v_car, x_goal, y_goal, t_goal, v_goal), total=len(v_car))
    )
    table = [solution for solution in table if solution[1] is not None]
    inputs =  np.array([solution[0] for solution in table])
    outputs = np.array([solution[1] for solution in table])
    outputs = np.moveaxis(outputs, 1, 2)

    np.savez(f'nmpc_lookup_table.npz', inputs=inputs, outputs=outputs)


""" Test mesh grid parameters. """
# car velocity [23]
test_v_car_min = 0.7
test_v_car_max = 6.3
test_dv_car =    0.3

# goal x-position [35]
test_x_goal_min = 0.25
test_x_goal_max = 3.25
test_dx_goal =    0.1

# goal y-position [35]
test_y_goal_min = 0.25
test_y_goal_max = 3.25
test_dy_goal =    0.1

# goal heading [62]
test_t_goal_min = -2.85
test_t_goal_max =  2.85
test_dt_goal =     0.1

# goal velocity [23]
test_v_goal_min = 0.7
test_v_goal_max = 6.3
test_dv_goal =    0.3


def test_data():
    # generate input state mesh grid
    print('Generating test state mesh grid')
    v_car = np.arange(test_v_car_min, test_v_car_max + test_dv_car, test_dv_car)
    x_goal = np.arange(test_x_goal_min, test_x_goal_max + test_dx_goal, test_dx_goal)
    y_goal = np.arange(test_y_goal_min, test_y_goal_max + test_dy_goal, test_dy_goal)
    t_goal = np.arange(test_t_goal_min, test_t_goal_max + test_dt_goal, test_dt_goal)
    v_goal = np.arange(test_v_goal_min, test_v_goal_max + test_dv_goal, test_dv_goal)
    v_car_m, x_goal_m, y_goal_m, t_goal_m, v_goal_m = np.meshgrid(
        v_car, x_goal, y_goal, t_goal, v_goal, indexing='ij'
    )
    v_car = v_car_m.flatten()
    x_goal = x_goal_m.flatten()
    y_goal = y_goal_m.flatten()
    t_goal = t_goal_m.flatten()
    v_goal = v_goal_m.flatten()
    print(f'Test state mesh grid generation completed: {len(v_car)} samples')

    table = Parallel(n_jobs=n_jobs)(
        delayed(mpc_solve)(vc, xg, yg, tg, vg)
        for vc, xg, yg, tg, vg in tqdm(zip(v_car, x_goal, y_goal, t_goal, v_goal), total=len(v_car))
    )
    table = [solution for solution in table if solution[1] is not None]
    inputs =  np.array([solution[0] for solution in table])
    outputs = np.array([solution[1] for solution in table])
    outputs = np.moveaxis(outputs, 1, 2)

    np.savez(f'test_nmpc_lookup_table.npz', inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    test_data()


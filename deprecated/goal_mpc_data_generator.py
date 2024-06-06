import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from goal_mpc_node import MPC, mpc_solution_generator


## input state mesh grid parameters [>1.1 billion]
# car velocity [18]
v_car_min = -1.0
v_car_max =  8.0
dv_car =     0.5

# goal x-position [52]
x_goal_min = -1.2
x_goal_max =  4.0
dx_goal =     0.1

# goal y-position [40]
y_goal_min = 0.0
y_goal_max = 4.0
dy_goal =    0.1

# goal heading [63]
t_goal_min = -3.14
t_goal_max =  3.14
dt_goal =     0.1

# goal velocity [18]
v_goal_min = -1.0
v_goal_max =  8.0
dv_goal =     0.5


def main(args=None):
    n_jobs = 45  # number of jobs

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
    print('Input state mesh grid generation completed:', len(v_car), 'samples')

    v_car_split = np.array_split(v_car, n_jobs)
    x_goal_split = np.array_split(x_goal, n_jobs)
    y_goal_split = np.array_split(y_goal, n_jobs)
    t_goal_split = np.array_split(t_goal, n_jobs)
    v_goal_split = np.array_split(v_goal, n_jobs)

    mpc_list = [MPC() for _ in range(n_jobs)]

    table = Parallel(n_jobs=n_jobs)(
        delayed(mpc_solution_generator)(vc, xg, yg, tg, vg, mpc)
        for vc, xg, yg, tg, vg, mpc in tqdm(zip(v_car_split, x_goal_split, y_goal_split, t_goal_split, v_goal_split, mpc_list), total=n_jobs)
    )
    table = np.concatenate(table, axis=0)

    np.savez(f'goal_mpc_lookup_table_tiny_2.npz', table=table)
    print('Final shape:', table.shape)
    print('Done')


def normalize(data: str):
    # loading raw data
    print('Loading data...')
    table = np.load(data)['table']
    v_car = table[:, 0].flatten()
    x_goal = table[:, 1].flatten()
    y_goal = table[:, 2].flatten()
    t_goal = table[:, 3].flatten()
    v_goal = table[:, 4].flatten()
    speed = table[:, 5].flatten()
    steer = table[:, 6].flatten()
    print('Data import completed')
    print('Table shape:', table.shape)

    print('Mirroring data...')
    v_car_m = np.concatenate((v_car, v_car))
    x_goal_m = np.concatenate((x_goal, x_goal))
    y_goal_m = np.concatenate((y_goal, -y_goal))
    t_goal_m = np.concatenate((t_goal, -t_goal))
    v_goal_m = np.concatenate((v_goal, v_goal))
    speed_m = np.concatenate((speed, speed))
    steer_m = np.concatenate((steer, -steer))
    print('Mirroring completed')

    # get actual min and max
    print('Finding range...')
    v_c = [min(v_car_m), max(v_car_m)]
    print('Done 1/7...')
    x_g = [min(x_goal_m), max(x_goal_m)]
    print('Done 2/7...')
    y_g = [min(y_goal_m), max(y_goal_m)]
    print('Done 3/7...')
    t_g = [min(t_goal_m), max(t_goal_m)]
    print('Done 4/7...')
    v_g = [min(v_goal_m), max(v_goal_m)]
    print('Done 5/7...')
    speed_range = [min(speed_m), max(speed_m)]
    print('Done 6/7...')
    steer_range = [min(steer_m), max(steer_m)]
    print('Ranges found')
    np.save(f'{data.split(".")[0]}_ranges.npy', np.vstack((v_c, x_g, y_g, t_g, v_g, speed_range, steer_range)))
    print('Ranges saved')
    print(np.load(f'{data.split(".")[0]}_ranges.npy'))

    print('Making table...')
    table = np.vstack((v_car_m, x_goal_m, y_goal_m, t_goal_m, v_goal_m, speed_m, steer_m))
    table = np.transpose(table)
    print('Table generated')
    print('Table shape:', table.shape)

    # normalize [v_car, x_goal, y_goal, t_goal, v_goal, speed, steer]
    print('Normalizing data...')
    table[:, 0] = (table[:, 0] - v_c[0]) / (v_c[-1] - v_c[0])
    table[:, 1] = (table[:, 1] - x_g[0]) / (x_g[-1] - x_g[0])
    table[:, 2] = (table[:, 2] - y_g[0]) / (y_g[-1] - y_g[0])
    table[:, 3] = (table[:, 3] - t_g[0]) / (t_g[-1] - t_g[0])
    table[:, 4] = (table[:, 4] - v_g[0]) / (v_g[-1] - v_g[0])
    table[:, 5] = (table[:, 5] - speed_range[0]) / (speed_range[-1] - speed_range[0])
    table[:, 6] = (table[:, 6] - steer_range[0]) / (steer_range[-1] - steer_range[0])
    print('Normalization completed')

    print('Confirming normalization...')
    print('v_car min:', min(table[:, 0]), 'v_car max:', max(table[:, 0]))
    print('x_goal min:', min(table[:, 1]), 'x_goal max:', max(table[:, 1]))
    print('y_goal min:', min(table[:, 2]), 'y_goal max:', max(table[:, 2]))
    print('t_goal min:', min(table[:, 3]), 't_goal max:', max(table[:, 3]))
    print('v_goal min:', min(table[:, 4]), 'v_goal max:', max(table[:, 4]))
    print('speed min:', min(table[:, 5]), 'speed max:', max(table[:, 5]))
    print('steer min:', min(table[:, 6]), 'steer max:', max(table[:, 6]))
    
    print('Saving table...')
    np.savez(f'{data.split(".")[0]}_normalized.npz', table=table)
    print('Final shape:', table.shape)
    print('Done')


if __name__ == "__main__":
    # main()
    # normalize('goal_mpc_lookup_table_tiny_2.npz')
    ranges = np.load('goal_mpc_lookup_table_tiny_2_ranges.npy')
    print(ranges)


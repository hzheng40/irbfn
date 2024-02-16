import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from goal_mpc_node import solve_mpc


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
    print('Input state mesh grid generation completed:', len(v_car))

    # generate MPC lookup table
    # print('Importing previous checkpoint')
    # full_table = np.load('goal_mpc_lookup_table.npz')['table']
    # print('Checkpoint import completed')
    # print(len(v_car))
    # print(full_table.shape)
    # interval = 100000000
    # start = 0
    # end = interval
    # while start < len(v_car):
    #     table = Parallel(n_jobs=n_jobs)(
    #         delayed(solve_mpc)(vc_i, xg_i, yg_i, tg_i, vg_i)
    #         for vc_i, xg_i, yg_i, tg_i, vg_i in tqdm(zip(v_car[start:end], x_goal[start:end], y_goal[start:end], t_goal[start:end], v_goal[start:end]), total=end-start)
    #     )
    #     table = np.array(list(filter(lambda item: item is not None, table)))
    #     if full_table is None:
    #         full_table = table
    #     else:
    #         full_table = np.concatenate((full_table, table), axis=0)
    #     np.savez(f'goal_mpc_lookup_table_{end}.npz', table=full_table)
    #     start += interval
    #     end = min(end + interval, len(v_car))
    # np.savez('goal_mpc_lookup_table.npz', table=full_table)
    start = 0
    end = len(v_car)
    table = Parallel(n_jobs=n_jobs)(
        delayed(solve_mpc)(vc_i, xg_i, yg_i, tg_i, vg_i)
        for vc_i, xg_i, yg_i, tg_i, vg_i in tqdm(zip(v_car[start:end], x_goal[start:end], y_goal[start:end], t_goal[start:end], v_goal[start:end]), total=end-start)
    )
    table = np.array(list(filter(lambda item: item is not None, table)))
    np.savez(f'goal_mpc_lookup_table_tiny_2.npz', table=table)
    print('Final shape:', table.shape)
    print('Done')


if __name__ == "__main__":
    main()


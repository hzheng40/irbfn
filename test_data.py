import numpy as np
from tqdm import tqdm
from goal_mpc_node import solve_mpc


def check_table_index(table_name, index):
    print('Loading table...')
    full_table = np.load(table_name)['table']
    print('Table imported')
    data_point = full_table[index]
    saved = data_point[-2:]
    actual = solve_mpc(*data_point[:5])[-2:]
    print('Input:', data_point[:5])
    print('Saved Output:', saved)
    print('Actual Output:', actual)
    print('Difference:', np.abs(saved - actual))


def check_single_table_sample(table_name, data_point):
    print('Loading table...')
    table = np.load(table_name)['table']
    print('Table imported')
    v_car = table[:, 0].flatten()
    x_goal = table[:, 1].flatten()
    y_goal = table[:, 2].flatten()
    t_goal = table[:, 3].flatten()
    v_goal = table[:, 4].flatten()
    speed = table[:, 5].flatten()
    steer = table[:, 6].flatten()

    print('Mirroring data...')
    v_car_m = np.concatenate((v_car, v_car))
    x_goal_m = np.concatenate((x_goal, x_goal))
    y_goal_m = np.concatenate((y_goal, -y_goal))
    t_goal_m = np.concatenate((t_goal, -t_goal))
    v_goal_m = np.concatenate((v_goal, v_goal))
    speed_m = np.concatenate((speed, speed))
    steer_m = np.concatenate((steer, -steer))
    print('Mirroring completed')

    print('Assembling table...')
    full_table = np.vstack((v_car_m, x_goal_m, y_goal_m, t_goal_m, v_goal_m, speed_m, steer_m)).T
    print('Table assembled')

    actual = solve_mpc(*data_point)
    saved = None
    distance = np.inf
    print('Starting linear search...')
    for point in tqdm(full_table):
        curr_distance = np.linalg.norm(data_point - point[:5])
        if curr_distance < distance:
            distance = curr_distance
            saved = point
            # print('Updated Distance:', distance)
            # print('New Save Point:', saved)

    print('Input:', data_point)
    print('Actual:', actual)
    print('Saved:', saved)
    print('Difference:', np.abs(actual[-2:] - saved[-2:]))


if __name__ == "__main__":
    check_table_index('goal_mpc_lookup_table_tiny_2.npz', 50000000)
    # data_point = np.array([0.247, 0.047, -0.986, -0.356, 2.0])
    # check_single_table_sample('goal_mpc_lookup_table_tiny_2.npz', data_point)


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
    full_table = np.load(table_name)['table']
    print('Table imported')
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
    # check_table_index('goal_mpc_lookup_table_tiny.npz', 100000)
    data_point = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    check_table_index('goal_mpc_lookup_table_tiny_2.npz', 1000)


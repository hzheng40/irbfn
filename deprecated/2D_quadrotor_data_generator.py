# 2D Quadrotor MPC
import numpy as np
import cvxpy
from joblib import Parallel, delayed
from tqdm import tqdm


xlim = 15
ulim = 5


def mpc_one_step(current_state: np.ndarray):
    # Define system dynamics and cost
    A = np.array([[1,1], [0,1]])
    B = np.array([[0], [1]])
    Q = np.diag([1.0, 1.0])
    R = np.array([[1.0]])

    N = 5

    # Initial state
    x0 = current_state

    # Reference state
    xg = np.array([[0], [0]])

    # Define optimization variables
    x = cvxpy.Variable((2, N+1), name="x")
    u = cvxpy.Variable((1, N), name="u")

    # Define parameters for the initial state and reference state
    x_init = cvxpy.Parameter((2, 1), name="x_init")
    x_init.value = x0
    x_ref = cvxpy.Parameter((2, 1), name="x_ref")
    x_ref.value = xg

    objective = 0.0
    constraints = []

    # Objective 1: Influence of the control inputs: Inputs u multiplied by the penalty R
    for i in range(N):
        objective += cvxpy.quad_form(u[:, i], R)

    # Objective 2: Deviation of the vehicle from the reference trajectory weighted by Q
    for i in range(N+1):
        objective += cvxpy.quad_form(x[:, i:i+1] - x_ref, Q)

    # Add dynamics constraints to the optimization problem
    for i in range(N):
        constraints += [x[:, i+1] == A @ x[:, i] + B @ u[:, i]]

    # Add L-inf norm of the control inputs and states to the constraints
    for i in range(N):
        constraints += [
            x[:, i] <=  xlim,
            x[:, i] >= -xlim,
            u[:, i] <=  ulim,
            u[:, i] >= -ulim,
        ]


    # Add constraints on the initial state
    constraints += [x[:, 0] == cvxpy.vec(x_init)]

    problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    # Solve the optimization problem for a single step
    problem.solve(warm_start=True, solver=cvxpy.OSQP, polish=True)
    if problem.status != cvxpy.OPTIMAL:
        return None

    return u[:, 0].value[0]


def solve_mpc(x1, x2):
    u = mpc_one_step(np.array([[x1], [x2]]))
    if u is None:
        return None
    return np.array([x1, x2, u])


def main():
    n_jobs = 45  # number of jobs

    samples = 2000
    X1 = np.linspace(-xlim, xlim, num=samples)
    X2 = np.linspace(-xlim, xlim, num=samples)

    print('Generating input state mesh grid')
    X1_m, X2_m = np.meshgrid(X1, X2, indexing='ij')
    X1 = X1_m.flatten()
    X2 = X2_m.flatten()
    print('Input state mesh grid generation completed:', len(X1), 'samples')

    """Generate testing data"""
    # X1_1 = np.linspace(-5, 5, num=100)
    # X2_1 = np.linspace(-5, 5, num=100)
    # X1_2 = np.linspace(-5, -10, num=100)
    # X2_2 = np.linspace(0, 10, num=100)
    # X1_3 = np.linspace(5, 10, num=100)
    # X2_3 = np.linspace(0, -10, num=100)

    # print('Generating input state mesh grid')
    # X1_1_m, X2_1_m = np.meshgrid(X1_1, X2_1, indexing='ij')
    # X1_2_m, X2_2_m = np.meshgrid(X1_2, X2_2, indexing='ij')
    # X1_3_m, X2_3_m = np.meshgrid(X1_3, X2_3, indexing='ij')
    # X1 = np.concatenate((X1_1_m.flatten(), X1_2_m.flatten(), X1_3_m.flatten()))
    # X2 = np.concatenate((X2_1_m.flatten(), X2_2_m.flatten(), X2_3_m.flatten()))
    # print('Input state mesh grid generation completed:', len(X1), 'samples')

    table = Parallel(n_jobs=n_jobs)(
        delayed(solve_mpc)(x1_i, x2_i) for x1_i, x2_i in tqdm(zip(X1, X2), total=len(X1))
    )
    table = np.array(list(filter(lambda item: item is not None, table)))

    np.savez(f'valid_data_{samples}x{samples}.npz', table=table)
    print('Final shape:', table.shape)
    print('Done')


if __name__ == '__main__':
    main()


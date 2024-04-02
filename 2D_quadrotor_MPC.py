# 2D Quadrotor MPC
import numpy as np
import matplotlib.pyplot as plt
import cvxpy


def main():
    # Define system dynamics and cost
    A = np.array([[1,1], [0,1]])
    B = np.array([[0], [1]])
    Q = np.diag([1.0, 1.0])
    R = np.array([[1.0]])

    N = 5

    xlim = 15
    ulim = 5

    # Initial state
    x0 = np.array([[3.0], [9.0]])

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

    # Block diagonal matrices
    Q_block = np.kron(np.eye(N+1), Q)
    R_block = np.kron(np.eye(N), R)
    A_block = np.kron(np.eye(N), A)
    B_block = np.kron(np.eye(N), B)

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

    # Solve the optimization problem in a receding horizon fashion
    x_t = x0
    x_history = [x_t]
    u_history = []
    Ts = 0.1
    t_hist = [0.0]
    while np.linalg.norm(x_t - xg) > 0.1:
        x_init.value = x_t
        problem.solve(warm_start=True, solver=cvxpy.OSQP, polish=True)
        if problem.status != cvxpy.OPTIMAL:
            print("Solver did not converge!")
            break
        
        x_t = A @ x_t + (B @ u[:, 0].value).reshape(-1, 1)
        x_history.append(x_t)
        u_history.append(u[:, 0].value)
        t_hist.append(t_hist[-1] + Ts)

    x_history = np.array(x_history)
    u_history = np.array(u_history)
    t_hist = np.array(t_hist)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_history[:, 0], x_history[:, 1], "o-")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    # Add box constraint on x plot
    plt.plot([-xlim, -xlim], [-xlim, xlim], color="red")
    plt.plot([xlim, xlim], [-xlim, xlim], color="red")
    plt.plot([-xlim, xlim], [-xlim, -xlim], color="red")
    plt.plot([-xlim, xlim], [xlim, xlim], color="red", label="Constraints")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_hist[:-1], u_history, "o-")
    # Plot constraints
    plt.plot(t_hist[:-1], ulim * np.ones_like(t_hist[:-1]), color="red")
    plt.plot(t_hist[:-1], -ulim * np.ones_like(t_hist[:-1]), color="red", label="Constraints")
    plt.xlabel("Time step")
    plt.ylabel("Control input")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()


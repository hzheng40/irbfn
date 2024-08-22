import jax
import jax.numpy as jnp
import numpy as np
import chex
from functools import partial
from numba import njit

N = 9

PARAM_MAT = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [-11.0 / 2, 9.0, -9.0 / 2, 1.0],
        [9.0, -45.0 / 2, 18.0, -9.0 / 2],
        [-9.0 / 2, 27.0 / 2, -27.0 / 2, 9.0 / 2],
    ]
)


@jax.jit
def params_to_coefs(params):
    s = params[-1]
    s2 = s**2
    s3 = s**3
    coefs = jnp.matmul(PARAM_MAT, params[:-1])
    coefs = coefs.at[1].divide(s)
    coefs = coefs.at[2].divide(s2)
    coefs = coefs.at[3].divide(s3)
    return coefs


@jax.jit
def get_curvature_theta(coefs, s_cur):
    out = 0.0
    out2 = 0.0
    for i in range(coefs.shape[0]):
        temp = coefs[i] * s_cur**i
        out += temp
        out2 += temp * s_cur / (i + 1)

    return out, out2


@jax.jit
@chex.assert_max_traces(n=1)
def integrate_one_step(state_init, seq, coefs):
    kappa_k, theta_k = get_curvature_theta(coefs, seq[0])
    dx = (
        state_init[4] * (1 - 1 / seq[1])
        + (jnp.cos(theta_k) + jnp.cos(state_init[2])) / 2 / seq[1]
    )
    dy = (
        state_init[5] * (1 - 1 / seq[1])
        + (jnp.sin(theta_k) + jnp.sin(state_init[2])) / 2 / seq[1]
    )
    x = seq[0] * dx
    y = seq[0] * dy
    state_new = jnp.stack([x, y, theta_k, kappa_k, dx, dy])
    return state_new, state_new


@jax.jit
@partial(jax.vmap, in_axes=(0,))
@chex.assert_max_traces(n=1)
def integrate_path_mult(params):
    coefs = params_to_coefs(params)
    state_init = jnp.zeros(
        6,
    )
    state_init = state_init.at[3].set(coefs[0])
    sk_seq = jnp.linspace(0.0, params[-1], num=N)
    k_seq = jnp.arange(1, N + 1)
    seq = jnp.vstack((sk_seq, k_seq)).T
    last_state, all_states = jax.lax.scan(
        partial(integrate_one_step, coefs=coefs), state_init, seq
    )
    return all_states


@jax.jit
@chex.assert_max_traces(n=1)
def integrate_path(params):
    coefs = params_to_coefs(params)
    states = jnp.empty((N, 4))
    states = states.at[0].set(
        jnp.zeros(
            4,
        )
    )
    states = states.at[0, 3].set(coefs[0])
    dx = 0.0
    dy = 0.0
    x = 0.0
    y = 0.0
    ds = params[-1] / N
    theta_old = 0.0
    for k in range(1, N):
        sk = k * ds
        kappa_k, theta_k = get_curvature_theta(coefs, sk)
        dx = dx * (1 - 1 / k) + (jnp.cos(theta_k) + jnp.cos(theta_old)) / 2 / k
        dy = dy * (1 - 1 / k) + (jnp.sin(theta_k) + jnp.sin(theta_old)) / 2 / k
        x = sk * dx
        y = sk * dy
        states = states.at[k].set(jnp.stack([x, y, theta_k, kappa_k]))
        theta_old = theta_k
    return states


@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world

    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = (trajectory[1:, :] - trajectory[:-1, :]).astype(np.float32)
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        lhs = (point - trajectory[i, :]).astype(np.float32)
        dots[i] = np.dot(lhs, diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


@njit(cache=True)
def intersect_point(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = np.float32(t % 1.0)
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory).astype(np.float32)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + np.float32(1e-6)
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = np.float32(2.0) * np.dot(V, start - point)
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - np.float32(2.0) * np.dot(start, point)
            - radius * radius
        )
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (np.float32(2.0) * a)
        t2 = (-b + discriminant) / (np.float32(2.0) * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + np.float32(1e-6)
            V = end - start

            a = np.dot(V, V)
            b = np.float32(2.0) * np.dot(V, start - point)
            c = (
                np.dot(start, start)
                + np.dot(point, point)
                - np.float32(2.0) * np.dot(start, point)
                - radius * radius
            )
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (np.float32(2.0) * a)
            t2 = (-b + discriminant) / (np.float32(2.0) * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


def zero_2_2pi(angle):
    if angle > 2 * np.pi:
        return angle - 2.0 * np.pi
    if angle < 0:
        return angle + 2.0 * np.pi

    return angle


def get_rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.ascontiguousarray(np.array([[c, -s], [s, c]]))

import jax
import jax.numpy as np
import chex
from functools import partial

g = 9.81


@jax.jit
@chex.assert_max_traces(n=1)
def dynamic_st_onestep(x, seq, params):
    """_summary_

    Parameters
    ----------
    current_state : jnp.array
        [[x, y, v, theta, beta, angv]]
    pred_u : jnp.array
        [accl_0, ..., accl_4, deltav_0, ..., deltav_4]
    params : jnp.array
        [mu, m, I, lf, lr C_Sf, C_Sr, h, dt]
    """

    mu = params[0]
    m = params[1]
    I = params[2]
    lf = params[3]
    lr = params[4]
    C_Sf = params[5]
    C_Sr = params[6]
    h = params[7]
    dt = params[8]
    sv_max = params[9]
    a_max = params[10]
    s_max = params[11]
    v_max = params[12]

    X = x[0]
    Y = x[1]
    DELTA = np.clip(x[2], min=-s_max, max=s_max)
    V = np.clip(x[3], min=-v_max, max=v_max)
    PSI = x[4]
    PSI_DOT = x[5]
    BETA = x[6]

    ACCL = np.clip(seq[0], min=-a_max, max=a_max)
    STEER_VEL = np.clip(seq[1], min=-sv_max, max=sv_max)

    f = np.array(
        [
            V * np.cos(PSI + BETA),  # X_DOT
            V * np.sin(PSI + BETA),  # Y_DOT
            STEER_VEL,  # DELTA_DOT
            ACCL,  # V_DOT
            PSI_DOT,  # PSI_DOT
            ((mu * m) / (I * (lf + lr)))
            * (
                lf * C_Sf * (g * lr - ACCL * h) * DELTA
                + (lr * C_Sr * (g * lf + ACCL * h) - lf * C_Sf * (g * lr - ACCL * h))
                * BETA
                - (
                    lf * lf * C_Sf * (g * lr - ACCL * h)
                    + lr * lr * C_Sr * (g * lf + ACCL * h)
                )
                * (PSI_DOT / V)
            ),  # PSI_DOT_DOT
            (mu / (V * (lr + lf)))
            * (
                C_Sf * (g * lr - ACCL * h) * DELTA
                - (C_Sr * (g * lf + ACCL * h) + C_Sf * (g * lr - ACCL * h)) * BETA
                + (C_Sr * (g * lf + ACCL * h) * lr - C_Sf * (g * lr - ACCL * h) * lf)
                * (PSI_DOT / V)
            )
            - PSI_DOT,  # BETA_DOT
        ]
    )

    f_ks = np.array(
        [
            V * np.cos(PSI),  # X_DOT
            V * np.sin(PSI),  # Y_DOT
            STEER_VEL,  # DELTA_DOT
            ACCL,  # V_DOT
            (V / (lr + lf)) * np.tan(DELTA),  # PSI_DOT
            0.0,
            0.0,
        ]
    )

    x_new = x + jax.lax.select(V > 3.0, f, f_ks) * dt
    return x_new, x_new


@jax.jit
@partial(jax.vmap, in_axes=(0, None))
@chex.assert_max_traces(n=1)
def integrate_st_mult(x_and_pred_u, params):
    seq = x_and_pred_u[7:].reshape(5, 2, order="F")
    last_state, all_states = jax.lax.scan(partial(dynamic_st_onestep, params=params), x_and_pred_u[:7], seq)
    return all_states


@jax.jit
@partial(jax.vmap, in_axes=(0, None))
@chex.assert_max_traces(n=1)
def dynamic_st_onestep_aux(x_u, params):
    """_summary_

    Parameters
    ----------
    current_state : jnp.array
        [[x, y, v, theta, beta, angv]]
    pred_u : jnp.array
        [accl_0, ..., accl_4, deltav_0, ..., deltav_4]
    params : jnp.array
        [mu, m, I, lf, lr C_Sf, C_Sr, h, dt]
    """

    mu = params[0]
    m = params[1]
    I = params[2]
    lf = params[3]
    lr = params[4]
    C_Sf = params[5]
    C_Sr = params[6]
    h = params[7]
    dt = params[8]
    sv_max = params[9]
    a_max = params[10]
    s_max = params[11]
    v_max = params[12]

    X = x_u[0]
    Y = x_u[1]
    DELTA = np.clip(x_u[2], min=-s_max, max=s_max)
    V = np.clip(x_u[3], min=-v_max, max=v_max)
    PSI = x_u[4]
    PSI_DOT = x_u[5]
    BETA = x_u[6]

    ACCL = np.clip(x_u[7], min=-a_max, max=a_max)
    STEER_VEL = np.clip(x_u[8], min=-sv_max, max=sv_max)

    f = np.array(
        [
            V * np.cos(PSI + BETA),  # X_DOT
            V * np.sin(PSI + BETA),  # Y_DOT
            STEER_VEL,  # DELTA_DOT
            ACCL,  # V_DOT
            PSI_DOT,  # PSI_DOT
            ((mu * m) / (I * (lf + lr)))
            * (
                lf * C_Sf * (g * lr - ACCL * h) * DELTA
                + (lr * C_Sr * (g * lf + ACCL * h) - lf * C_Sf * (g * lr - ACCL * h))
                * BETA
                - (
                    lf * lf * C_Sf * (g * lr - ACCL * h)
                    + lr * lr * C_Sr * (g * lf + ACCL * h)
                )
                * (PSI_DOT / V)
            ),  # PSI_DOT_DOT
            (mu / (V * (lr + lf)))
            * (
                C_Sf * (g * lr - ACCL * h) * DELTA
                - (C_Sr * (g * lf + ACCL * h) + C_Sf * (g * lr - ACCL * h)) * BETA
                + (C_Sr * (g * lf + ACCL * h) * lr - C_Sf * (g * lr - ACCL * h) * lf)
                * (PSI_DOT / V)
            )
            - PSI_DOT,  # BETA_DOT
        ]
    )

    f_ks = np.array(
        [
            V * np.cos(PSI),  # X_DOT
            V * np.sin(PSI),  # Y_DOT
            STEER_VEL,  # DELTA_DOT
            ACCL,  # V_DOT
            (V / (lr + lf)) * np.tan(DELTA),  # PSI_DOT
            0.0,
            0.0,
        ]
    )
    
    # x_new = x_u[:7] + jax.lax.select(V > 3.0, f, f_ks) * dt
    x_new = x_u[:7] + f_ks * dt
    return x_new


@jax.jit
@chex.assert_max_traces(n=1)
def dynamic_frenet_onestep(x, seq, params):
    """_summary_

    Parameters
    ----------
    current_state : jnp.array
        [[s, ey, delta, vx, vy, wz, eyaw, curv]]
    pred_u : jnp.array
        [accl_0, ..., accl_4, deltav_0, ..., deltav_4]
    params : jnp.array
        [mu, m, I, lf, lr C_Sf, C_Sr, h, dt]
    """

    MU = params[0]
    M = params[1]
    I = params[2]
    LF = params[3]
    LR = params[4]
    C_SF = params[5]
    C_SR = params[6]
    h = params[7]
    dt = params[8]
    sv_max = params[9]
    a_max = params[10]
    s_max = params[11]
    v_max = params[12]

    DF = MU * M * 9.81 / 2.0
    DR = MU * M * 9.81 / 2.0

    BF = 1.0
    BR = 1.0

    s = x[0]
    ey = x[1]
    delta = np.clip(x[2], min=-s_max, max=s_max)
    # vx = np.clip x[3], min=-v_max, max=v_max)
    vx = x[3]
    vy = x[4]
    wz = x[5]
    epsi = x[6]
    cur = x[7]

    a = np.clip(seq[0], min=-a_max, max=a_max)
    deltv = np.clip(seq[1], min=-sv_max, max=sv_max)

    # Compute tire split angle
    alpha_f = delta - np.atan2(vy + LF * wz, vx)
    alpha_r = -np.atan2(vy - LF * wz, vx)

    # Compute lateral force at front and rear tire
    Fyf = DF * np.sin(
        C_SF * np.atan(BF * alpha_f)
    )
    Fyr = DR * np.sin(
        C_SR * np.atan(BR * alpha_r)
    )

    # [s, ey, delta, vx, vy, wz, epsi]
    deriv_x_hs = np.array(
        [((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey)),
        (vx * np.sin(epsi) + vy * np.cos(epsi)),
        deltv,
        (a - 1 / M * Fyf * np.sin(delta) + wz * vy),
        (1 / M * (Fyf * np.cos(delta) + Fyr) - wz * vx),
        (
            1
            / I
            * (LF * Fyf * np.cos(delta) - LR * Fyr)
        ),
        (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur),
        0.0,
        ])

    # [s, ey, delta, vx, vy(0.0), wz(0.0), epsi]
    deriv_x_ls = np.array(
        [(vx * np.cos(epsi)) / (1 - ey * cur),
        (vx * np.sin(epsi)),
        deltv,
        a,
        0.0,
        0.0,
        (vx * np.tan(delta)) / (LR + LF)
        - cur * ((vx * np.cos(epsi)) / (1 - cur * ey)),
        0.0,
    ])
    
    # x_new = x_u[:6] + jax.lax.select(np.sqrt(vx**2 + vy**2) > 3.0, deriv_x_hs, deriv_x_ls) * dt
    x_new = x + deriv_x_ls * dt
    return x_new, x_new


@jax.jit
@partial(jax.vmap, in_axes=(0, None))
@chex.assert_max_traces(n=1)
def integrate_frenet_mult(x_and_pred_u, params):
    seq = x_and_pred_u[8:].reshape(5, 2, order="F")
    last_state, all_states = jax.lax.scan(partial(dynamic_frenet_onestep, params=params), x_and_pred_u[:8], seq)
    return all_states


@jax.jit
@partial(jax.vmap, in_axes=(0, None))
@chex.assert_max_traces(n=1)
def dynamic_frenet_onestep_aux(x_u, params):
    """_summary_

    Parameters
    ----------
    current_state : jnp.array
        [[s, ey, delta, vx, vy, wz, eyaw, curv]]
    pred_u : jnp.array
        [accl_0, ..., accl_4, deltav_0, ..., deltav_4]
    params : jnp.array
        [mu, m, I, lf, lr C_Sf, C_Sr, h, dt]
    """

    MU = params[0]
    M = params[1]
    I = params[2]
    LF = params[3]
    LR = params[4]
    C_SF = params[5]
    C_SR = params[6]
    h = params[7]
    dt = params[8]
    sv_max = params[9]
    a_max = params[10]
    s_max = params[11]
    v_max = params[12]

    DF = MU * M * 9.81 / 2.0
    DR = MU * M * 9.81 / 2.0

    BF = 1.0
    BR = 1.0

    # s = x_u[0]
    ey = x_u[0]
    delta = np.clip(x_u[1], min=-s_max, max=s_max)
    # vx = np.clip(x_u[3], min=-v_max, max=v_max)
    vx = x_u[2]
    vy = x_u[3]
    wz = x_u[4]
    epsi = x_u[5]
    cur = x_u[6]

    a = np.clip(x_u[8], min=-a_max, max=a_max)
    deltv = np.clip(x_u[9], min=-sv_max, max=sv_max)

    # Compute tire split angle
    alpha_f = delta - np.atan2(vy + LF * wz, vx)
    alpha_r = -np.atan2(vy - LF * wz, vx)

    # Compute lateral force at front and rear tire
    Fyf = DF * np.sin(
        C_SF * np.atan(BF * alpha_f)
    )
    Fyr = DR * np.sin(
        C_SR * np.atan(BR * alpha_r)
    )

    # [s, ey, delta, vx, vy, wz, epsi]
    deriv_x_hs = np.array(
        [((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey)),
        (vx * np.sin(epsi) + vy * np.cos(epsi)),
        deltv,
        (a - 1 / M * Fyf * np.sin(delta) + wz * vy),
        (1 / M * (Fyf * np.cos(delta) + Fyr) - wz * vx),
        (
            1
            / I
            * (LF * Fyf * np.cos(delta) - LR * Fyr)
        ),
        (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur),
        ])

    # [s, ey, delta, vx, vy(0.0), wz(0.0), epsi]
    deriv_x_ls = np.array(
        [(vx * np.cos(epsi)) / (1 - ey * cur),
        (vx * np.sin(epsi)),
        deltv,
        a,
        0.0,
        0.0,
        (vx * np.tan(delta)) / (LR + LF)
        - cur * ((vx * np.cos(epsi)) / (1 - cur * ey)),
    ])
    
    # x_new = x_u[:6] + jax.lax.select(np.sqrt(vx**2 + vy**2) > 3.0, deriv_x_hs, deriv_x_ls) * dt
    x_new = x_u[:6] + deriv_x_ls[1:] * dt
    return x_new
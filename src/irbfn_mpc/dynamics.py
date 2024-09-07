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
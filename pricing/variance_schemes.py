import numpy as np
from scipy.stats import norm

from config import *


def step_variance_qe(V, dt, kappa, theta, xi, Z, psi_c=QE_PSI_C):
    """Andersen QE step for CIR. See lsv/particle_method.py for full docstring."""
    if dt <= 0:
        return np.maximum(V, 0.0).copy()
    V = np.maximum(V, 0.0)
    e_kt = np.exp(-kappa * dt)
    one_minus = 1.0 - e_kt
    m = theta + (V - theta) * e_kt
    s2 = (V * xi * xi * e_kt / kappa) * one_minus \
         + (theta * xi * xi / (2.0 * kappa)) * one_minus * one_minus
    m_safe = np.maximum(m, 1e-30)
    psi = s2 / (m_safe * m_safe)

    V_new = np.empty_like(V)
    case_A = psi <= psi_c
    case_B = ~case_A

    if np.any(case_A):
        psi_A = psi[case_A]
        m_A = m[case_A]
        Z_A = Z[case_A]
        inv = 2.0 / np.maximum(psi_A, 1e-30)
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
        b = np.sqrt(b2)
        a = m_A / (1.0 + b2)
        V_new[case_A] = a * (b + Z_A) ** 2

    if np.any(case_B):
        psi_B = psi[case_B]
        m_B = m[case_B]
        U_B = norm.cdf(Z[case_B])
        p = (psi_B - 1.0) / (psi_B + 1.0)
        beta = (1.0 - p) / np.maximum(m_B, 1e-30)
        below_mass = U_B <= p
        V_new_B = np.zeros_like(U_B)
        if np.any(~below_mass):
            denom = np.maximum(1.0 - U_B[~below_mass], 1e-30)
            V_new_B[~below_mass] = -np.log(
                (1.0 - p[~below_mass]) / denom
            ) / beta[~below_mass]
        V_new[case_B] = V_new_B

    return np.maximum(V_new, 0.0)

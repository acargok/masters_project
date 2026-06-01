import numpy as np
from scipy import interpolate

from variance_schemes import step_variance_qe


class HestonVariance:
    """
    Heston CIR variance process with selectable discretisation scheme.

    dV = kappa * (theta - V) * dt + xi * sqrt(V) * dW^V

    Two schemes:
      - "euler": full-truncation Euler. step(V, dt, Z) advances V given a
        Brownian increment Z that should be the variance leg of a correlated
        (Z1, Z2) pair. The simulator does the standard log-Euler S update.
      - "qe": Andersen 2008 QE. The variance is sampled from the exact CIR
        conditional law (Case A quadratic-Gaussian or Case B exponential),
        and the spot is updated via the Broadie-Kaya decomposition that
        consumes V_old and V_new. The simulator detects scheme=="qe" and
        calls joint_step instead of step.
    """

    uses_spot_noise = False   # selects the simulator's variance-noise branch

    def __init__(self, kappa, theta, xi, V0, scheme="qe", rho=None):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.V0 = V0
        self.scheme = scheme
        # rho is required for the QE joint step; the simulator passes it
        # explicitly if not set here.
        self.rho = rho

    def initial_variance(self, n_paths):
        return np.full(n_paths, self.V0)

    def step(self, V, dt, Z):
        """Full-truncation Euler step (scheme='euler')."""
        sqrt_dt = np.sqrt(dt)
        V_pos = np.maximum(V, 0.0)
        V_new = (
            V
            + self.kappa * (self.theta - V_pos) * dt
            + self.xi * np.sqrt(V_pos) * sqrt_dt * Z
        )
        return np.maximum(V_new, 0.0)

    def qe_step(self, V, dt, Z):
        """Sample V_{t+dt} via Andersen QE. Z is a single standard normal."""
        return step_variance_qe(V, dt, self.kappa, self.theta, self.xi, Z)

    def qe_log_spot_increment(self, V_old, V_new, L, dt, r, q, rho, Z_perp):
        """
        Broadie-Kaya log-spot increment given V_old, V_new, leverage L, and an
        independent standard normal Z_perp. Returns the *increment* to add to
        log S_t.
        """
        V_bar = 0.5 * (V_old + V_new)
        V_bar_pos = np.maximum(V_bar, 0.0)
        drift_corr = V_new - V_old - self.kappa * self.theta * dt + self.kappa * V_bar * dt
        return (
            (r - q) * dt
            - 0.5 * (L ** 2) * V_bar * dt
            + (rho * L / self.xi) * drift_corr
            + L * np.sqrt((1.0 - rho ** 2) * V_bar_pos * dt) * Z_perp
        )


class BergomiVariance:
    """
    Bergomi two-factor spot variance process for cliquet pricing.

    The spot variance is:
        xi^t_t = xi^t_0 * exp(omega * x^t_t - omega^2/2 * chi(t,t))

    where x^t_t = alpha_theta * [(1-theta)*X1 + theta*X2]
    and X1, X2 are OU processes driven by W^1, W^2 correlated with W^S.

    The step() method receives Z_spot (the spot Brownian) and uses internal
    Cholesky decomposition to produce correlated OU increments.

    Interface matches HestonVariance: step(V, dt, Z) -> V_new.
    """

    def __init__(self, nu, theta, kappa1, kappa2, rho1, rho2, rho12,
                 fwd_var_curve, ttm_grid, seed=42):
        self.nu = nu
        self.theta = theta
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho12 = rho12
        # See lsv_bergomi/particle_method.py: omega = 2*nu (empirically
        # verified; omega = nu produces a systematic ME bias of about -55 bp).
        self.omega = 2.0 * nu

        # alpha_theta normalisation
        denom = np.sqrt((1 - theta)**2 + theta**2
                        + 2 * rho12 * theta * (1 - theta))
        self.alpha_theta = 1.0 / max(denom, 1e-10)

        # Forward variance interpolator
        self.fwd_var_interp = interpolate.interp1d(
            ttm_grid, fwd_var_curve, kind="linear",
            bounds_error=False, fill_value=(fwd_var_curve[0], fwd_var_curve[-1]),
        )
        self.ttm_min = ttm_grid[0]
        self.ttm_max = ttm_grid[-1]

        # Cholesky factor for (W^S, W^1, W^2) correlation
        corr = np.array([
            [1.0,   rho1,  rho2],
            [rho1,  1.0,   rho12],
            [rho2,  rho12, 1.0],
        ])
        # Ensure positive-definite (nearest PD fallback)
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() < 1e-8:
            from scipy.linalg import sqrtm
            corr = corr + (1e-6 - eigvals.min()) * np.eye(3)
        self.chol = np.linalg.cholesky(corr)

        self.rng = np.random.default_rng(seed)
        self.X1 = None
        self.X2 = None
        self.t = 0.0  # current time
        self.uses_spot_noise = True  # step() expects Z_spot, not Z_vol

    def initial_variance(self, n_paths):
        """Return initial spot variance xi^0_0 and set up OU states."""
        self.X1 = np.zeros(n_paths)
        self.X2 = np.zeros(n_paths)
        self.t = 0.0
        xi_0 = float(self.fwd_var_interp(np.clip(1e-4, self.ttm_min, self.ttm_max)))
        return np.full(n_paths, max(xi_0, 1e-8))

    def _compute_chi(self, t, T):
        """chi(t, T) = Var[x^T_t]."""
        th = self.theta
        k1, k2 = self.kappa1, self.kappa2
        rho12 = self.rho12
        alpha = self.alpha_theta
        tau = T - t

        var_X1 = (1.0 - np.exp(-2.0 * k1 * t)) / (2.0 * k1) if k1 > 1e-10 else t
        var_X2 = (1.0 - np.exp(-2.0 * k2 * t)) / (2.0 * k2) if k2 > 1e-10 else t
        cov_X12 = (rho12 * (1.0 - np.exp(-(k1 + k2) * t)) / (k1 + k2)
                   if (k1 + k2) > 1e-10 else rho12 * t)

        chi = alpha**2 * (
            (1.0 - th)**2 * np.exp(-2.0 * k1 * tau) * var_X1
            + th**2 * np.exp(-2.0 * k2 * tau) * var_X2
            + 2.0 * (1.0 - th) * th * np.exp(-(k1 + k2) * tau) * cov_X12
        )
        return max(chi, 0.0)

    def step(self, V, dt, Z_spot):
        """
        Advance variance by one time step.

        Z_spot is the spot Brownian increment (standard normal).
        Correlated OU increments are generated internally via Cholesky.
        Returns new spot variance xi^t_t at t + dt.
        """
        n = len(V)

        # Generate independent normals and correlate via Cholesky
        Z_indep = self.rng.standard_normal((n, 2))
        # W^S is Z_spot; derive W^1, W^2 correlated with it
        # [W^S, W^1, W^2] = chol @ [e1, e2, e3]
        # We have Z_spot (= e1 contribution). Need to back-solve:
        # W^1 = chol[1,0]*Z_spot + chol[1,1]*e2 + chol[1,2]*e3
        # W^2 = chol[2,0]*Z_spot + chol[2,1]*e2 + chol[2,2]*e3
        Z_W1 = (self.chol[1, 0] * Z_spot
                + self.chol[1, 1] * Z_indep[:, 0])
        Z_W2 = (self.chol[2, 0] * Z_spot
                + self.chol[2, 1] * Z_indep[:, 0]
                + self.chol[2, 2] * Z_indep[:, 1])

        # Exact OU simulation
        decay1 = np.exp(-self.kappa1 * dt)
        decay2 = np.exp(-self.kappa2 * dt)
        std1 = np.sqrt((1.0 - decay1**2) / (2.0 * self.kappa1)) if self.kappa1 > 1e-10 else np.sqrt(dt)
        std2 = np.sqrt((1.0 - decay2**2) / (2.0 * self.kappa2)) if self.kappa2 > 1e-10 else np.sqrt(dt)

        self.X1 = self.X1 * decay1 + std1 * Z_W1
        self.X2 = self.X2 * decay2 + std2 * Z_W2
        self.t += dt

        # Spot variance at new time
        t = self.t
        alpha = self.alpha_theta
        th = self.theta

        # x^t_t: at T=t, decay factors are e^0 = 1
        x_t_t = alpha * ((1.0 - th) * self.X1 + th * self.X2)

        # chi(t, t)
        chi_t_t = self._compute_chi(t, t)

        # f_t(t, x) = exp(omega * x - omega^2/2 * chi)
        f_val = np.exp(self.omega * x_t_t - 0.5 * self.omega**2 * chi_t_t)

        # xi^t_0
        t_clamped = np.clip(t, self.ttm_min, self.ttm_max)
        xi_t_0 = max(float(self.fwd_var_interp(t_clamped)), 1e-8)

        # xi^t_t = xi^t_0 * f_t
        V_new = xi_t_0 * f_val
        return np.maximum(V_new, 1e-8)

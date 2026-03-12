"""
MCMC fitting of dark matter velocity distributions to SPARC rotation curve data.

Model
-----
Total circular velocity squared:

    V_obs^2 = sign(Vgas)*Vgas^2
              + upsilon_disk  * Vdisk^2
              + upsilon_bulge * Vbul^2
              + V_DM^2(r; a, b)

The DM profile V_DM^2(r, a, b) is a pluggable function with that signature.
Default: pseudo-isothermal (pISO)

    V_DM^2 = a * (1 - arctan(r/b) / (r/b))

where  a  [km^2/s^2] is the asymptotic DM velocity squared and
       b  [kpc]      is the core radius.

Per-galaxy free parameters: upsilon_disk, upsilon_bulge, a, b
Sampled in log space: theta = [ln(upsilon_disk), ln(upsilon_bulge), ln(a), ln(b)]

Likelihood
----------
  Student-t on V_obs with fixed degrees of freedom _NU_T (default 3).
  Heavier tails than Gaussian; robust to outlier rotation curve points.
  Set _NU_T to a large value (e.g. 100) to recover near-Gaussian behaviour.

Priors
------
  ln(upsilon_disk)  ~ N(ln 0.5, sigma=0.1 dex)   (log-normal with median 0.5)
  ln(upsilon_bulge) ~ N(ln 0.7, sigma=0.1 dex)   (log-normal with median 0.7)
  ln(a)             ~ Uniform(-2, 20)      (broad, uninformative)
  ln(b)             ~ Uniform(-5, 8)       (broad, uninformative)

Usage
-----
    from datawrangling import produce_SPARC_df
    from mcmc_fit import fit_all_galaxies, plot_residuals, plot_galaxy_fit

    df = produce_SPARC_df("data/SPARC", reduced_load=True, n=10)
    results = fit_all_galaxies(df, n_walkers=32, n_steps=2000, n_burn=500)
    plot_residuals(df, results, save_path="residuals.png")
    plot_galaxy_fit(df, results, galaxy=results["galaxy"].iloc[0], save_path="fit.png")

Custom DM profile
-----------------
    def custom_profile(r, a, b):
        x = r / b
        return a * arctan(x)**2

    results = fit_all_galaxies(df, dm_profile=custom_profile)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from typing import Callable

import emcee
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dark matter profiles
# ---------------------------------------------------------------------------

def vdm2_pISO(r: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Pseudo-isothermal DM profile.

        V_DM^2 = a * (1 - arctan(r/b) / (r/b))

    a [km^2/s^2] : asymptotic velocity squared
    b [kpc]      : core radius
    """
    x = r / b
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.where(x == 0.0, 0.0, 1.0 - np.arctan(x) / x)
    return a * val

def nfw(r, a, b):
    """
    NFW profile.

        V_DM^2 = a/(r/b) * (ln(1+(r/b))-(r+b)/(1+(r/b)))

    a [km^2/s^2] : asymptotic velocity squared
    b [kpc]      : core radius
    """  
    x = r / b
    return a * (np.log(1 + x) - x / (1 + x)) / x


def vdm2_from_density_2param(h_callable: Callable) -> Callable:
    """
    Wrap a 2-parameter density shape h(x, gamma), x = r/b, as a V²_DM
    profile with signature (r, a, b, gamma) -> V²_DM.

    Physics identical to vdm2_from_density; gamma is a per-galaxy shape
    parameter (e.g. cuspiness) passed through to h.

    a [km²/s²/kpc²] : absorbs 4πGρ₀
    b [kpc]         : scale radius
    gamma           : dimensionless shape parameter
    """
    def _dm_profile(r: np.ndarray, a: float, b: float, gamma: float) -> np.ndarray:
        order = np.argsort(r)
        r_s = r[order]
        x_s = np.where(r_s > 0, r_s / b, 1.0)
        h_vals = np.where(r_s > 0, h_callable(x_s, gamma), 0.0)
        h_vals = np.where(np.isfinite(h_vals) & (h_vals >= 0.0), h_vals, 0.0)
        integrand = h_vals * r_s**2
        r_full  = np.concatenate([[0.0], r_s])
        ig_full = np.concatenate([[0.0], integrand])
        cum_I = cumulative_trapezoid(ig_full, r_full, initial=0.0)[1:]
        v2_s = np.where(r_s > 0, a * cum_I / r_s, 0.0)
        v2 = np.empty_like(r, dtype=float)
        v2[order] = v2_s
        return v2
    return _dm_profile


def vdm2_from_density(h_callable: Callable) -> Callable:
    """
    Wrap a density shape h(x), x = r/b, as a V²_DM profile function
    with signature (r, a, b) -> V²_DM, compatible with dm_profile everywhere.

    Physics:
        V²_DM(r_i) = (a / r_i) · ∫₀^{r_i} h(r′/b) · r′² dr′

    a [km²/s²/kpc²] absorbs 4πGρ₀; b [kpc] is the scale radius.
    The integrand h(r/b)·r² → 0 at r=0 for all physical profiles (pISO, NFW,
    Burkert, Sersic), so the virtual origin is set to zero automatically.
    """
    def _dm_profile(r: np.ndarray, a: float, b: float) -> np.ndarray:
        order = np.argsort(r)
        r_s = r[order]
        # Evaluate h safely: avoid calling h(0) for divergent profiles (e.g. NFW)
        x_s = np.where(r_s > 0, r_s / b, 1.0)   # dummy x=1 where r=0
        integrand = np.where(r_s > 0, h_callable(x_s) * r_s**2, 0.0)
        r_full  = np.concatenate([[0.0], r_s])
        ig_full = np.concatenate([[0.0], integrand])
        cum_I = cumulative_trapezoid(ig_full, r_full, initial=0.0)[1:]
        v2_s = np.where(r_s > 0, a * cum_I / r_s, 0.0)
        v2 = np.empty_like(r, dtype=float)
        v2[order] = v2_s
        return v2
    return _dm_profile


# ---------------------------------------------------------------------------
# Density-based DM profiles (via vdm2_from_density)
# ---------------------------------------------------------------------------

# Standard profiles
vdm2_pISO_density    = vdm2_from_density(lambda x: 1.0 / (1.0 + x**2))
vdm2_NFW_density     = vdm2_from_density(lambda x: 1.0 / (x * (1.0 + x)**2))
vdm2_Burkert_density = vdm2_from_density(lambda x: 1.0 / ((1.0 + x) * (1.0 + x**2)))

# SR stage-5 discoveries (density run, SPARC Q=1)
# c15 best physical form: sqrt(exp(x^{1/8}/(x+0.024) - sqrt(x)))
vdm2_SR_density_c15 = vdm2_from_density(
    lambda x: np.sqrt(np.exp(x**(1.0/8) / (x + 0.024309041) - np.sqrt(x))))

# c18 form: exp(sqrt(sqrt(x)/0.944)/(x+0.029)) / exp(sqrt(x))
vdm2_SR_density_c18 = vdm2_from_density(
    lambda x: np.exp(np.sqrt(np.sqrt(x) / 0.9440967) / (x + 0.028753186) - np.sqrt(x)))


# SR stage-6 discoveries (2-parameter density run, SPARC Q=1)
# c12: x^{-log1p(γ+x)/0.41468} — power law with γ-dependent exponent (cuspy for γ>0)
vdm2_SR6_c12 = vdm2_from_density_2param(
    lambda x, g: np.exp(np.log1p(g + x) / (-0.41468284) * np.log(np.maximum(x, 1e-10))))

# c21: algebraic cored form  0.993/((xγ−γ+x)·γ·(x−0.334)+x)
vdm2_SR6_c21 = vdm2_from_density_2param(
    lambda x, g: 0.99260813 / (((x*g - (g - x)) * (g * (x - 0.33422744))) + x + 1e-10))

# c11: 1/(x+(x−γ)²)  — numerator constant absorbed by free `a`
vdm2_SR6_c11 = vdm2_from_density_2param(
    lambda x, g: 1.0 / (x + (x - g)**2 + 1e-10))

# c18: 1/(√x+(x−1.466γ)(x−γ))  — numerator 5.146 absorbed by free `a`; best SR loss=0.258
vdm2_SR6_c18_hof = vdm2_from_density_2param(
    lambda x, g: 1.0 / (np.sqrt(np.maximum(x, 0.0)) + (x - 1.4656022*g) * (x - g) + 1e-10))


# default profile
vdm2 = vdm2_pISO


# ---------------------------------------------------------------------------
# Total velocity model
# ---------------------------------------------------------------------------

def v2_total(
    r: np.ndarray,
    vgas: np.ndarray,
    vdisk: np.ndarray,
    vbul: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
    a: float,
    b: float,
    dm_profile: Callable = vdm2_pISO,
) -> np.ndarray:
    """
    Total circular velocity squared.

    Vgas uses signed squares (sign(Vgas)*Vgas^2) to respect the SPARC
    convention where Vgas may be negative for pressure-dominated regions.
    """
    return (
        np.sign(vgas) * vgas**2
        + upsilon_disk  * vdisk**2
        + upsilon_bulge * vbul**2
        + dm_profile(r, a, b)
    )


# ---------------------------------------------------------------------------
# Priors, likelihood, posterior
# ---------------------------------------------------------------------------

# Prior bounds on ln(a) and ln(b)
_LN_A_MIN, _LN_A_MAX = -2.0, 20.0   # a ~ e^-2 to e^20  km^2/s^2
_LN_B_MIN, _LN_B_MAX = -5.0,  8.0   # b ~ 0.007 to 3000  kpc

# Degrees of freedom for the Student-t likelihood.
# nu=3 gives heavy tails (robust to outliers); nu=inf recovers Gaussian.
_NU_T = 3.0

# Log-normal prior parameters for upsilons (sampling in log space)
_MU_DISK   = np.log(0.5)
_MU_BULGE  = np.log(0.7)
_SIGMA_UPS = 0.1 * np.log(10)   # 0.1 dex in ln space


def log_prior(theta: np.ndarray) -> float:
    """
    Log prior.  theta = [ln_upsd, ln_upsb, ln_a, ln_b]

    upsilon_disk  ~ LogNormal(median=0.5, sigma=0.1 dex)
    upsilon_bulge ~ LogNormal(median=0.7, sigma=0.1 dex)
    a, b          ~ log-Uniform (broad)
    """
    ln_upsd, ln_upsb, ln_a, ln_b = theta

    # uniform 
    if not (_LN_A_MIN < ln_a < _LN_A_MAX):
        return -np.inf
    if not (_LN_B_MIN < ln_b < _LN_B_MAX):
        return -np.inf

    # log normal
    lp_upsd = -0.5 * ((ln_upsd  - _MU_DISK ) / _SIGMA_UPS)**2
    lp_upsb = -0.5 * ((ln_upsb  - _MU_BULGE) / _SIGMA_UPS)**2

    return lp_upsd + lp_upsb


def _log_likelihood(
    theta: np.ndarray,
    r: np.ndarray,
    vobs: np.ndarray,
    verr: np.ndarray,
    vgas: np.ndarray,
    vdisk: np.ndarray,
    vbul: np.ndarray,
    dm_profile: Callable,
) -> float:
    """
    Student-t log likelihood on V_obs (not V^2).

    log p ∝ -(nu+1)/2 · Σ log(1 + z_i²/nu), z_i = (V_obs,i - V_pred,i) / sigma_i

    Heavier tails than Gaussian (controlled by _NU_T) make the fit
    more robust to outlier data points.
    """
    ln_upsd, ln_upsb, ln_a, ln_b = theta

    upsilon_disk  = np.exp(ln_upsd)
    upsilon_bulge = np.exp(ln_upsb)
    a = np.exp(ln_a)
    b = np.exp(ln_b)

    v2_pred = v2_total(r, vgas, vdisk, vbul, upsilon_disk, upsilon_bulge, a, b,
                       dm_profile)

    if np.any(~np.isfinite(v2_pred)) or np.any(v2_pred < 0.0):
        return -np.inf

    v_pred = np.sqrt(v2_pred)
    z2 = ((vobs - v_pred) / verr) ** 2
    return -0.5 * (_NU_T + 1) * np.sum(np.log1p(z2 / _NU_T))


def _log_posterior(
    theta: np.ndarray,
    r: np.ndarray,
    vobs: np.ndarray,
    verr: np.ndarray,
    vgas: np.ndarray,
    vdisk: np.ndarray,
    vbul: np.ndarray,
    dm_profile: Callable,
) -> float:
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(theta, r, vobs, verr, vgas, vdisk, vbul, dm_profile)


# ---------------------------------------------------------------------------
# 5-parameter variants (adds per-galaxy gamma for 2-param density profiles)
# ---------------------------------------------------------------------------

_GAMMA_MIN, _GAMMA_MAX = 0.0, 3.0


def log_prior_5param(theta: np.ndarray) -> float:
    """
    Log prior for 5-parameter fit.
    theta = [ln_upsd, ln_upsb, ln_a, ln_b, gamma]

    gamma ~ Uniform(0, 3)  (per-galaxy shape / cuspiness)
    All other priors identical to log_prior.
    """
    ln_upsd, ln_upsb, ln_a, ln_b, gamma = theta

    if not (_LN_A_MIN < ln_a < _LN_A_MAX):
        return -np.inf
    if not (_LN_B_MIN < ln_b < _LN_B_MAX):
        return -np.inf
    if not (_GAMMA_MIN < gamma < _GAMMA_MAX):
        return -np.inf

    lp_upsd = -0.5 * ((ln_upsd - _MU_DISK ) / _SIGMA_UPS)**2
    lp_upsb = -0.5 * ((ln_upsb - _MU_BULGE) / _SIGMA_UPS)**2
    return lp_upsd + lp_upsb


def _log_likelihood_5param(
    theta: np.ndarray,
    r: np.ndarray,
    vobs: np.ndarray,
    verr: np.ndarray,
    vgas: np.ndarray,
    vdisk: np.ndarray,
    vbul: np.ndarray,
    dm_profile: Callable,
) -> float:
    ln_upsd, ln_upsb, ln_a, ln_b, gamma = theta

    upsilon_disk  = np.exp(ln_upsd)
    upsilon_bulge = np.exp(ln_upsb)
    a = np.exp(ln_a)
    b = np.exp(ln_b)

    v2_pred = (
        np.sign(vgas) * vgas**2
        + upsilon_disk  * vdisk**2
        + upsilon_bulge * vbul**2
        + dm_profile(r, a, b, gamma)
    )
    if np.any(~np.isfinite(v2_pred)) or np.any(v2_pred < 0.0):
        return -np.inf

    v_pred = np.sqrt(v2_pred)
    z2 = ((vobs - v_pred) / verr) ** 2
    return -0.5 * (_NU_T + 1) * np.sum(np.log1p(z2 / _NU_T))


def _log_posterior_5param(
    theta: np.ndarray,
    r: np.ndarray,
    vobs: np.ndarray,
    verr: np.ndarray,
    vgas: np.ndarray,
    vdisk: np.ndarray,
    vbul: np.ndarray,
    dm_profile: Callable,
) -> float:
    lp = log_prior_5param(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood_5param(theta, r, vobs, verr, vgas, vdisk, vbul, dm_profile)


def _initial_theta_5param(
    r: np.ndarray,
    vobs: np.ndarray,
    vgas: np.ndarray,
    vdisk: np.ndarray,
    vbul: np.ndarray,
    dm_profile: Callable,
) -> np.ndarray:
    """Optimise the 5-param posterior to find a good starting point for walkers."""
    ups_d, ups_b = 0.5, 0.7
    v2_bar = (
        np.sign(vgas) * vgas**2
        + ups_d * vdisk**2
        + ups_b * vbul**2
    )
    v2_dm_guess = np.maximum(vobs**2 - v2_bar, 10.0)
    a_guess = float(np.median(v2_dm_guess))
    b_guess = float(np.median(r)) if np.median(r) > 0 else 1.0

    x0 = np.array([np.log(ups_d), np.log(ups_b), np.log(a_guess), np.log(b_guess), 1.0])

    def neg_lp(theta):
        return -_log_posterior_5param(theta, r, vobs, np.ones_like(vobs),
                                      vgas, vdisk, vbul, dm_profile)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(neg_lp, x0, method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-4})

    return result.x if result.success else x0


def fit_galaxy_5param(
    galaxy_df: pd.DataFrame,
    dm_profile: Callable,
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burn: int = 500,
    progress: bool = False,
) -> dict:
    """
    Run 5-parameter MCMC fit for a single galaxy.

    dm_profile must have signature (r, a, b, gamma) -> V_DM^2.
    Use vdm2_from_density_2param to create such a profile.

    Returns dict with all 4-param keys plus:
      gamma, gamma_lo, gamma_hi
    """
    r     = galaxy_df["Rad_kpc"].values.astype(float)
    vobs  = galaxy_df["Vobs_km/s"].values.astype(float)
    verr  = np.maximum(galaxy_df["errV_km/s"].values.astype(float), 0.5)
    vgas  = galaxy_df["Vgas_km/s"].values.astype(float)
    vdisk = galaxy_df["Vdisk_km/s"].values.astype(float)
    vbul  = galaxy_df["Vbul_km/s"].values.astype(float)

    ndim = 5
    theta0 = _initial_theta_5param(r, vobs, vgas, vdisk, vbul, dm_profile)
    rng = np.random.default_rng()
    pos = theta0 + 1e-3 * rng.standard_normal((n_walkers, ndim))
    # clip gamma walkers to valid range
    pos[:, 4] = np.clip(pos[:, 4], _GAMMA_MIN + 0.01, _GAMMA_MAX - 0.01)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, _log_posterior_5param,
        args=(r, vobs, verr, vgas, vdisk, vbul, dm_profile),
    )
    sampler.run_mcmc(pos, n_steps, progress=progress)

    samples = sampler.get_chain(discard=n_burn, flat=True)
    acc_frac = float(np.mean(sampler.acceptance_fraction))

    p16, p50, p84 = np.percentile(samples, [16, 50, 84], axis=0)

    ups_d  = np.exp(p50[0])
    ups_b  = np.exp(p50[1])
    a_med  = np.exp(p50[2])
    b_med  = np.exp(p50[3])
    g_med  = p50[4]

    v2_pred = (
        np.sign(vgas) * vgas**2
        + ups_d  * vdisk**2
        + ups_b  * vbul**2
        + dm_profile(r, a_med, b_med, g_med)
    )
    v_pred  = np.sqrt(np.maximum(v2_pred, 0.0))
    chi2_red = float(np.sum(((vobs - v_pred) / verr) ** 2) / max(len(r) - ndim, 1))

    return {
        "upsilon_disk":    ups_d,
        "upsilon_disk_lo": np.exp(p16[0]),
        "upsilon_disk_hi": np.exp(p84[0]),
        "upsilon_bulge":    ups_b,
        "upsilon_bulge_lo": np.exp(p16[1]),
        "upsilon_bulge_hi": np.exp(p84[1]),
        "a":    a_med,
        "a_lo": np.exp(p16[2]),
        "a_hi": np.exp(p84[2]),
        "b_kpc":    b_med,
        "b_kpc_lo": np.exp(p16[3]),
        "b_kpc_hi": np.exp(p84[3]),
        "gamma":    g_med,
        "gamma_lo": p16[4],
        "gamma_hi": p84[4],
        "chi2_reduced":        chi2_red,
        "n_data":              len(r),
        "acceptance_fraction": acc_frac,
    }


def fit_all_galaxies_5param(
    df: pd.DataFrame,
    dm_profile: Callable,
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burn: int = 500,
    galaxies: list | None = None,
) -> pd.DataFrame:
    """
    Run per-galaxy 5-parameter MCMC fits for all (or a subset of) galaxies.

    dm_profile must have signature (r, a, b, gamma) -> V_DM^2.

    Returns DataFrame with all 4-param columns plus:
      gamma, gamma_lo, gamma_hi
    """
    if galaxies is None:
        galaxies = sorted(df["galaxy"].unique())

    rows = []
    n_failed = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for gal in tqdm(galaxies, desc="Fitting galaxies (5-param)"):
            gdf = df[df["galaxy"] == gal].copy()
            if len(gdf) < 6:   # need at least ndim+1 points
                n_failed += 1
                continue
            try:
                result = fit_galaxy_5param(gdf, dm_profile=dm_profile,
                                           n_walkers=n_walkers, n_steps=n_steps,
                                           n_burn=n_burn, progress=False)
                result["galaxy"] = gal
                rows.append(result)
            except Exception:
                n_failed += 1

    cols = [
        "galaxy",
        "upsilon_disk", "upsilon_disk_lo", "upsilon_disk_hi",
        "upsilon_bulge", "upsilon_bulge_lo", "upsilon_bulge_hi",
        "a", "a_lo", "a_hi",
        "b_kpc", "b_kpc_lo", "b_kpc_hi",
        "gamma", "gamma_lo", "gamma_hi",
        "chi2_reduced", "n_data", "acceptance_fraction",
    ]
    results = pd.DataFrame(rows)[cols].reset_index(drop=True)
    _print_summary(results, n_failed)
    return results


# ---------------------------------------------------------------------------
# Per-galaxy MCMC fit
# ---------------------------------------------------------------------------

def _initial_theta(
    r: np.ndarray,
    vobs: np.ndarray,
    vgas: np.ndarray,
    vdisk: np.ndarray,
    vbul: np.ndarray,
    dm_profile: Callable,
) -> np.ndarray:
    """Optimise the posterior to find a good starting point for walkers."""
    ups_d, ups_b = 0.5, 0.7
    v2_bar = (
        np.sign(vgas) * vgas**2
        + ups_d * vdisk**2
        + ups_b * vbul**2
    )
    v2_dm_guess = np.maximum(vobs**2 - v2_bar, 10.0)
    a_guess = float(np.median(v2_dm_guess))
    b_guess = float(np.median(r)) if np.median(r) > 0 else 1.0

    x0 = np.array([np.log(ups_d), np.log(ups_b), np.log(a_guess), np.log(b_guess)])

    def neg_lp(theta):
        return -_log_posterior(theta, r, vobs, np.ones_like(vobs),
                               vgas, vdisk, vbul, dm_profile)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(neg_lp, x0, method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-4})

    return result.x if result.success else x0


def fit_galaxy(
    galaxy_df: pd.DataFrame,
    dm_profile: Callable = vdm2_pISO,
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burn: int = 500,
    progress: bool = False,
    return_samples: bool = False,
) -> dict | tuple[dict, np.ndarray]:
    """
    Run MCMC fit for a single galaxy.

    Parameters
    ----------
    galaxy_df  : DataFrame for one galaxy (from produce_SPARC_df).
    dm_profile : Callable with signature (r, a, b) -> V_DM^2 array.
                 Defaults to pseudo-isothermal.
    n_walkers  : Number of emcee walkers.
    n_steps    : Total MCMC steps per walker.
    n_burn     : Steps discarded as burn-in.
    progress   : Show per-step progress bar.

    Returns
    -------
    dict with keys:
      upsilon_disk, upsilon_disk_lo, upsilon_disk_hi
      upsilon_bulge, upsilon_bulge_lo, upsilon_bulge_hi
      a, a_lo, a_hi             [km^2/s^2]
      b_kpc, b_kpc_lo, b_kpc_hi [kpc]
      chi2_reduced, n_data, acceptance_fraction
    """
    r     = galaxy_df["Rad_kpc"].values.astype(float)
    vobs  = galaxy_df["Vobs_km/s"].values.astype(float)
    verr  = galaxy_df["errV_km/s"].values.astype(float)
    vgas  = galaxy_df["Vgas_km/s"].values.astype(float)
    vdisk = galaxy_df["Vdisk_km/s"].values.astype(float)
    vbul  = galaxy_df["Vbul_km/s"].values.astype(float)

    verr = np.maximum(verr, 0.5)

    ndim = 4
    theta0 = _initial_theta(r, vobs, vgas, vdisk, vbul, dm_profile)
    rng = np.random.default_rng()
    pos = theta0 + 1e-3 * rng.standard_normal((n_walkers, ndim))

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, _log_posterior,
        args=(r, vobs, verr, vgas, vdisk, vbul, dm_profile),
    )
    sampler.run_mcmc(pos, n_steps, progress=progress)

    samples = sampler.get_chain(discard=n_burn, flat=True)
    acc_frac = float(np.mean(sampler.acceptance_fraction))

    p16, p50, p84 = np.percentile(samples, [16, 50, 84], axis=0)

    ups_d = np.exp(p50[0])
    ups_b = np.exp(p50[1])
    a_med = np.exp(p50[2])
    b_med = np.exp(p50[3])

    v2_pred = v2_total(r, vgas, vdisk, vbul, ups_d, ups_b, a_med, b_med, dm_profile)
    v_pred  = np.sqrt(np.maximum(v2_pred, 0.0))
    chi2_red = float(np.sum(((vobs - v_pred) / verr) ** 2) / max(len(r) - ndim, 1))

    summary = {
        "upsilon_disk":    ups_d,
        "upsilon_disk_lo": np.exp(p16[0]),
        "upsilon_disk_hi": np.exp(p84[0]),
        "upsilon_bulge":    ups_b,
        "upsilon_bulge_lo": np.exp(p16[1]),
        "upsilon_bulge_hi": np.exp(p84[1]),
        "a":    a_med,
        "a_lo": np.exp(p16[2]),
        "a_hi": np.exp(p84[2]),
        "b_kpc":    b_med,
        "b_kpc_lo": np.exp(p16[3]),
        "b_kpc_hi": np.exp(p84[3]),
        "chi2_reduced":        chi2_red,
        "n_data":              len(r),
        "acceptance_fraction": acc_frac,
    }
    if return_samples:
        return summary, np.exp(samples)
    return summary


# ---------------------------------------------------------------------------
# Fit all galaxies + summary
# ---------------------------------------------------------------------------

def _print_summary(results: pd.DataFrame, n_failed: int) -> None:
    chi2   = results["chi2_reduced"]
    acc    = results["acceptance_fraction"]
    ups_d  = results["upsilon_disk"]
    ups_b  = results["upsilon_bulge"]
    a_vals = results["a"]
    b_vals = results["b_kpc"]
    n      = len(results)

    sep = "─" * 52
    print(f"\n{'MCMC fit summary':^52}")
    print(sep)
    print(f"  Galaxies fitted : {n}   (failed / skipped: {n_failed})")
    print(sep)
    print(f"  {'Statistic':<26}  {'Mean':>8}  {'Median':>8}")
    print(f"  {'─'*26}  {'─'*8}  {'─'*8}")
    print(f"  {'χ²_ν':<26}  {chi2.mean():>8.3f}  {chi2.median():>8.3f}")
    print(f"  {'acceptance fraction':<26}  {acc.mean():>8.3f}  {acc.median():>8.3f}")
    print(f"  {'Υ_disk':<26}  {ups_d.mean():>8.3f}  {ups_d.median():>8.3f}")
    print(f"  {'Υ_bulge':<26}  {ups_b.mean():>8.3f}  {ups_b.median():>8.3f}")
    print(f"  {'a  [km²/s²]':<26}  {a_vals.mean():>8.1f}  {a_vals.median():>8.1f}")
    print(f"  {'b  [kpc]':<26}  {b_vals.mean():>8.3f}  {b_vals.median():>8.3f}")
    print(sep)
    good = (chi2 < 2).sum()
    warn = ((chi2 >= 2) & (chi2 < 5)).sum()
    poor = (chi2 >= 5).sum()
    print(f"  χ²_ν < 2  (good)  : {good:>3}  ({100*good/n:.0f}%)")
    print(f"  χ²_ν 2–5  (warn)  : {warn:>3}  ({100*warn/n:.0f}%)")
    print(f"  χ²_ν ≥ 5  (poor)  : {poor:>3}  ({100*poor/n:.0f}%)")
    low_acc = (acc < 0.15).sum()
    if low_acc:
        print(f"\n  ⚠  {low_acc} galaxies have acceptance fraction < 0.15")
        print("     Consider increasing n_steps or n_walkers.")
    print(sep)


def fit_all_galaxies(
    df: pd.DataFrame,
    dm_profile: Callable = vdm2_pISO,
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burn: int = 500,
    galaxies: list | None = None,
    return_samples: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Run per-galaxy MCMC fits for all (or a subset of) galaxies.

    Parameters
    ----------
    df         : Combined SPARC DataFrame from produce_SPARC_df.
    dm_profile : Callable (r, a, b) -> V_DM^2.  Defaults to pISO.
    n_walkers  : emcee walkers.
    n_steps    : Total steps per walker.
    n_burn     : Burn-in steps to discard.
    galaxies   : Optional list of galaxy names; defaults to all in df.

    Returns
    -------
    DataFrame with columns:
      galaxy,
      upsilon_disk, upsilon_disk_lo, upsilon_disk_hi,
      upsilon_bulge, upsilon_bulge_lo, upsilon_bulge_hi,
      a, a_lo, a_hi,
      b_kpc, b_kpc_lo, b_kpc_hi,
      chi2_reduced, n_data, acceptance_fraction
    """
    if galaxies is None:
        galaxies = sorted(df["galaxy"].unique())

    rows = []
    samples_dict = {}
    n_failed = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for gal in tqdm(galaxies, desc="Fitting galaxies"):
            gdf = df[df["galaxy"] == gal].copy()
            if len(gdf) < 5:
                n_failed += 1
                continue
            try:
                out = fit_galaxy(gdf, dm_profile=dm_profile,
                                 n_walkers=n_walkers, n_steps=n_steps,
                                 n_burn=n_burn, progress=False,
                                 return_samples=return_samples)
                if return_samples:
                    result, samps = out
                    samples_dict[gal] = samps
                else:
                    result = out
                result["galaxy"] = gal
                rows.append(result)
            except Exception:
                n_failed += 1

    cols = [
        "galaxy",
        "upsilon_disk", "upsilon_disk_lo", "upsilon_disk_hi",
        "upsilon_bulge", "upsilon_bulge_lo", "upsilon_bulge_hi",
        "a", "a_lo", "a_hi",
        "b_kpc", "b_kpc_lo", "b_kpc_hi",
        "chi2_reduced", "n_data", "acceptance_fraction",
    ]
    results = pd.DataFrame(rows)[cols].reset_index(drop=True)
    _print_summary(results, n_failed)
    if return_samples:
        return results, samples_dict
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_residuals(
    df: pd.DataFrame,
    results: pd.DataFrame,
    dm_profile: Callable = vdm2_pISO,
    save_path: str | None = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """
    Plot a histogram of normalised residuals (V_obs - V_model) / sigma
    across all galaxies in results.

    Parameters
    ----------
    df         : Combined SPARC DataFrame.
    results    : Output of fit_all_galaxies.
    dm_profile : Must match the profile used when fitting.
    save_path  : If given, save figure to this path.
    figsize    : Figure size.
    """
    all_r, all_res = [], []

    for _, row in results.iterrows():
        gdf = df[df["galaxy"] == row["galaxy"]]
        r     = gdf["Rad_kpc"].values.astype(float)
        vobs  = gdf["Vobs_km/s"].values.astype(float)
        verr  = np.maximum(gdf["errV_km/s"].values.astype(float), 0.5)
        vgas  = gdf["Vgas_km/s"].values.astype(float)
        vdisk = gdf["Vdisk_km/s"].values.astype(float)
        vbul  = gdf["Vbul_km/s"].values.astype(float)

        v2_pred = v2_total(r, vgas, vdisk, vbul,
                           row["upsilon_disk"], row["upsilon_bulge"],
                           row["a"], row["b_kpc"], dm_profile)
        v_pred  = np.sqrt(np.maximum(v2_pred, 0.0))
        all_r.extend(r.tolist())
        all_res.extend(((vobs - v_pred) / verr).tolist())

    all_r   = np.array(all_r)
    all_res = np.array(all_res)

    bins = np.linspace(0, np.percentile(all_r, 95), 20)
    bin_idx = np.digitize(all_r, bins)
    bin_centres, bin_medians, bin_errs = [], [], []
    for i in range(1, len(bins)):
        mask = bin_idx == i
        if mask.sum() < 3:
            continue
        vals = all_res[mask]
        bin_centres.append(0.5 * (bins[i - 1] + bins[i]))
        bin_medians.append(np.median(vals))
        bin_errs.append(np.std(vals))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(bin_centres, bin_medians, width=bins[1] - bins[0], color="steelblue",
           alpha=0.7, edgecolor="white", linewidth=0.4, label="Median residual")
    ax.errorbar(bin_centres, bin_medians, yerr=bin_errs, fmt="none",
                color="black", capsize=3, linewidth=1)

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")

    ax.set_xlabel("Radius  [kpc]", fontsize=11)
    ax.set_ylabel(r"$(V_\mathrm{obs} - V_\mathrm{model})\,/\,\sigma$", fontsize=11)
    ax.set_title(f"Residuals – MCMC fit  ({len(results)} galaxies)", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_galaxy_fit(
    df: pd.DataFrame,
    results: pd.DataFrame,
    galaxy: str,
    dm_profile: Callable = vdm2_pISO,
    dm_label: str = "Dark matter",
    save_path: str | None = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Plot the rotation curve decomposition for one galaxy.

    Parameters
    ----------
    df         : Combined SPARC DataFrame.
    results    : Output of fit_all_galaxies.
    galaxy     : Name of galaxy to plot.
    dm_profile : Must match the profile used when fitting.
    dm_label   : Legend label for the DM curve.
    save_path  : If given, save figure to this path.
    figsize    : Figure size.
    """
    row = results[results["galaxy"] == galaxy]
    if row.empty:
        raise ValueError(f"Galaxy '{galaxy}' not found in results.")
    row = row.iloc[0]

    gdf  = df[df["galaxy"] == galaxy].sort_values("Rad_kpc")
    r     = gdf["Rad_kpc"].values.astype(float)
    vobs  = gdf["Vobs_km/s"].values.astype(float)
    verr  = gdf["errV_km/s"].values.astype(float)
    vgas  = gdf["Vgas_km/s"].values.astype(float)
    vdisk = gdf["Vdisk_km/s"].values.astype(float)
    vbul  = gdf["Vbul_km/s"].values.astype(float)

    r_fine = np.linspace(0.0, r.max() * 1.1, 300)

    ups_d = row["upsilon_disk"]
    ups_b = row["upsilon_bulge"]
    a     = row["a"]
    b     = row["b_kpc"]

    v_dm        = np.sqrt(np.maximum(dm_profile(r_fine, a, b), 0.0))
    v_disk_fine = np.sqrt(ups_d) * np.interp(r_fine, r, np.abs(vdisk))
    v_bul_fine  = np.sqrt(ups_b) * np.interp(r_fine, r, np.abs(vbul))
    v_gas_fine  = np.interp(r_fine, r,
                            np.sqrt(np.maximum(np.sign(vgas) * vgas**2, 0.0)))

    v2_tot = (
        np.interp(r_fine, r, np.sign(vgas) * vgas**2)
        + ups_d * np.interp(r_fine, r, vdisk**2)
        + ups_b * np.interp(r_fine, r, vbul**2)
        + dm_profile(r_fine, a, b)
    )
    v_tot = np.sqrt(np.maximum(v2_tot, 0.0))

    # Uncertainty band from 16th/84th percentile parameter corners
    param_combos = [
        (row["upsilon_disk_lo"], row["upsilon_bulge_lo"], row["a_lo"], row["b_kpc_lo"]),
        (row["upsilon_disk_hi"], row["upsilon_bulge_hi"], row["a_hi"], row["b_kpc_hi"]),
        (row["upsilon_disk_lo"], row["upsilon_bulge_hi"], row["a_lo"], row["b_kpc_hi"]),
        (row["upsilon_disk_hi"], row["upsilon_bulge_lo"], row["a_hi"], row["b_kpc_lo"]),
    ]
    v_band = []
    for ud, ub, ai, bi in param_combos:
        v2 = (
            np.interp(r_fine, r, np.sign(vgas) * vgas**2)
            + ud * np.interp(r_fine, r, vdisk**2)
            + ub * np.interp(r_fine, r, vbul**2)
            + dm_profile(r_fine, ai, bi)
        )
        v_band.append(np.sqrt(np.maximum(v2, 0.0)))
    v_lo = np.min(v_band, axis=0)
    v_hi = np.max(v_band, axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(r, vobs, yerr=verr, fmt="ko", markersize=5,
                capsize=3, linewidth=1, label="Observed", zorder=5)

    ax.plot(r_fine, v_dm,        color="navy",   linewidth=1.5, linestyle="--",
            label=dm_label)
    ax.plot(r_fine, v_gas_fine,  color="green",  linewidth=1.2, linestyle=":",
            label="Gas")
    ax.plot(r_fine, v_disk_fine, color="orange", linewidth=1.2, linestyle="-.",
            label=rf"Disk  ($\Upsilon_\mathrm{{disk}}={ups_d:.2f}$)")
    if vbul.max() > 0:
        ax.plot(r_fine, v_bul_fine, color="purple", linewidth=1.2, linestyle="-.",
                label=rf"Bulge  ($\Upsilon_\mathrm{{bulge}}={ups_b:.2f}$)")

    ax.plot(r_fine, v_tot, color="red", linewidth=2.0, label="Total model")
    ax.fill_between(r_fine, v_lo, v_hi, color="red", alpha=0.15,
                    label=r"$1\sigma$ band")

    ax.set_xlabel("Radius  [kpc]", fontsize=11)
    ax.set_ylabel("Circular velocity  [km/s]", fontsize=11)
    ax.set_title(
        f"{galaxy}   MCMC fit\n"
        rf"$a={a:.0f}$ km$^2$/s$^2$,  $b={b:.2f}$ kpc,  "
        rf"$\Upsilon_\mathrm{{disk}}={ups_d:.2f}$,  $\Upsilon_\mathrm{{bulge}}={ups_b:.2f}$,  "
        rf"$\chi^2_\nu={row['chi2_reduced']:.2f}$",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


# ---------------------------------------------------------------------------
# Corner / degeneracy plots
# ---------------------------------------------------------------------------

_PARAM_IDX = {"upsilon_disk": 0, "upsilon_bulge": 1, "a": 2, "b_kpc": 3}
_PARAM_LABELS = {
    "upsilon_disk":  r"$\Upsilon_\mathrm{disk}$",
    "upsilon_bulge": r"$\Upsilon_\mathrm{bulge}$",
    "a":             r"$a\ [\mathrm{km}^2\,\mathrm{s}^{-2}]$",
    "b_kpc":         r"$b\ [\mathrm{kpc}]$",
}


def get_samples(
    galaxy_df: pd.DataFrame,
    dm_profile: Callable = vdm2_pISO,
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burn: int = 500,
) -> np.ndarray:
    """
    Run MCMC for one galaxy and return flat posterior samples in physical space.

    Returns
    -------
    np.ndarray of shape (n_samples, 4):
        columns: upsilon_disk, upsilon_bulge, a, b_kpc
    """
    r     = galaxy_df["Rad_kpc"].values.astype(float)
    vobs  = galaxy_df["Vobs_km/s"].values.astype(float)
    verr  = np.maximum(galaxy_df["errV_km/s"].values.astype(float), 0.5)
    vgas  = galaxy_df["Vgas_km/s"].values.astype(float)
    vdisk = galaxy_df["Vdisk_km/s"].values.astype(float)
    vbul  = galaxy_df["Vbul_km/s"].values.astype(float)

    theta0 = _initial_theta(r, vobs, vgas, vdisk, vbul, dm_profile)
    rng = np.random.default_rng()
    pos = theta0 + 1e-3 * rng.standard_normal((n_walkers, 4))

    sampler = emcee.EnsembleSampler(
        n_walkers, 4, _log_posterior,
        args=(r, vobs, verr, vgas, vdisk, vbul, dm_profile),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler.run_mcmc(pos, n_steps, progress=True)

    return np.exp(sampler.get_chain(discard=n_burn, flat=True))


def plot_corner(
    df: pd.DataFrame,
    galaxy: str,
    params: list[str] = ["a", "b_kpc"],
    dm_profile: Callable = vdm2_pISO,
    samples_dict: dict | None = None,
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burn: int = 500,
    save_path: str | None = None,
    figsize: tuple = (6, 6),
) -> plt.Figure:
    """
    Corner plot for two posterior parameters for a single galaxy.

    Diagonal panels: 1-D marginals with 16/50/84th percentile lines.
    Lower-left panel: 2-D hexbin density with Pearson correlation coefficient.

    Parameters
    ----------
    df         : Combined SPARC DataFrame.
    galaxy     : Name of the galaxy to analyse.
    params     : Exactly 2 names from ["upsilon_disk", "upsilon_bulge", "a", "b_kpc"].
    dm_profile : Must match the profile you want to sample.
    """
    if len(params) != 2:
        raise ValueError("params must contain exactly 2 parameter names.")
    for p in params:
        if p not in _PARAM_IDX:
            raise ValueError(f"Unknown parameter '{p}'. Choose from {list(_PARAM_IDX)}")

    gdf = df[df["galaxy"] == galaxy]
    if gdf.empty:
        raise ValueError(f"Galaxy '{galaxy}' not found in df.")

    if samples_dict is not None and galaxy in samples_dict:
        samples = samples_dict[galaxy]
    else:
        samples = get_samples(gdf, dm_profile=dm_profile,
                              n_walkers=n_walkers, n_steps=n_steps, n_burn=n_burn)

    s0  = samples[:, _PARAM_IDX[params[0]]]
    s1  = samples[:, _PARAM_IDX[params[1]]]
    rho = float(np.corrcoef(s0, s1)[0, 1])
    lab0, lab1 = _PARAM_LABELS[params[0]], _PARAM_LABELS[params[1]]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # diagonal: 1-D marginals
    for ax, s, lab in [(axes[0, 0], s0, lab0), (axes[1, 1], s1, lab1)]:
        ax.hist(s, bins=40, color="steelblue", edgecolor="white",
                linewidth=0.4, density=True)
        for q, ls in zip(np.percentile(s, [16, 50, 84]), ["--", "-", "--"]):
            ax.axvline(q, color="tomato", linewidth=1.0, linestyle=ls)
        ax.set_xlabel(lab, fontsize=10)
        ax.set_ylabel("Density", fontsize=9)

    # lower-left: 2-D density + correlation
    ax = axes[1, 0]
    ax.hexbin(s0, s1, gridsize=30, cmap="Blues", mincnt=1)
    ax.set_xlabel(lab0, fontsize=10)
    ax.set_ylabel(lab1, fontsize=10)
    degeneracy = "  (strong)" if abs(rho) > 0.7 else ("  (moderate)" if abs(rho) > 0.4 else "")
    ax.set_title(rf"$\rho = {rho:+.3f}$" + degeneracy, fontsize=9)

    # upper-right: hide
    axes[0, 1].set_visible(False)

    fig.suptitle(f"{galaxy}  –  posterior correlations", fontsize=11)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig
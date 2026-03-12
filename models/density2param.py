from pysr import PySRRegressor, TemplateExpressionSpec
import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid


def fit_density_2param(df: pd.DataFrame,
                       output_directory: str = "outputs",
                       error_weighting: bool = True,
                       iterations: int = 99999,
                       n_galaxies: int | None = 5,
                       n_d_grid: int = 15,
                       n_gamma_grid: int = 8,
                       gamma_range: tuple = (0.0, 2.0),
                       populations: int = 15,
                       population_size: int = 40,
                       ncycles_per_iteration: int = 100,
                       weight_optimize: float = 0.1,
                       optimizer_iterations: int = 3,
                       upsilon_weight: float = 1.0,
                       nu_t: float = 3.0,
                       n_irls: int = 3,
                       min_points: int = 5,
                       unary_operators: list | None = None,
                       guesses: list | None = None,
                       fraction_replaced_guesses: float = 0.001,
                       weight_monotone: float = 0.1,
                       weight_nonneg: float = 0.01,
                       n_gamma_refine: int = 10):
    """
    Two-parameter symbolic regression for galaxy rotation curves: learns DM
    density shape h(x, γ) where x = r/d and γ is a per-galaxy parameter.

    Extends fit_density_inner_opt (models/density.py) by adding a second
    per-galaxy parameter γ optimised on a linear grid alongside scale radius d.

    Model structure (per data point i in galaxy g):
        V²_obs = sign(Vgas)·V²_gas + a[g]·V²_disk + b[g]·V²_bul + c[g]·DM_col[i]

    where the DM column integrates the 2-parameter density shape:
        DM_col[i] = (1/r_i) · ∫₀^{r_i} h(r'/d[g], γ[g]) · r'² dr'

    Per-galaxy parameters:
        a, b  — M/L ratios (solved analytically via WLS/IRLS)
        c     — DM amplitude (solved analytically)
        d     — scale radius (optimised on n_d_grid log-spaced grid + 20-pt refinement)
        γ     — galaxy specific parameter

    The SR expression f(#1, #2) has:
        #1 = r/d  (updated per d-iteration)
        #2 = γ    (updated per γ-iteration; overwrites the galaxy-code row in X_eval)
    """
    if n_galaxies is not None:
        available = df["galaxy"].unique()
        rng = np.random.default_rng(seed=42)
        selected = rng.choice(available, size=min(n_galaxies, len(available)), replace=False)
        df = df[df["galaxy"].isin(selected)].copy()

    counts = df.groupby("galaxy")["galaxy"].transform("count")
    df = df[counts >= min_points].copy()

    galaxy_codes, galaxies = pd.factorize(df["galaxy"])
    df["galaxy_code"] = galaxy_codes + 1  # Julia is 1-indexed

    # Sort by (galaxy_code, r) — required for the cumulative trapezoid in Julia.
    df = df.sort_values(["galaxy_code", "Rad_kpc"]).reset_index(drop=True)

    y = df["Vobs_km/s"] ** 2
    X = pd.DataFrame({
        "r":       df["Rad_kpc"],
        "galaxy":  df["galaxy_code"],    # col 2 in Julia dataset.X (galaxy ID)
        "Vgas2":   np.sign(df["Vgas_km/s"]) * df["Vgas_km/s"] ** 2,
        "Vdisk2":  df["Vdisk_km/s"] ** 2,
        "Vbulge2": df["Vbul_km/s"] ** 2,
    })

    gamma_lo, gamma_hi = gamma_range

    inner_opt_loss = f"""
    function physicsloss(tree, dataset::Dataset{{T,L}}, options)::L where {{T,L}}
        n_pts      = length(dataset.y)
        n_gal      = Int(maximum(dataset.X[2, :]))
        total_loss = L(0)

        # Student-t degrees of freedom (matching mcmc_fit.py: ν=3)
        ν  = L({nu_t})
        ν1 = ν + L(1)

        # Log-normal priors on upsilon_disk (a) and upsilon_bulge (b),
        # matching mcmc_fit.py: lnN(ln 0.5, 0.1 dex) and lnN(ln 0.7, 0.1 dex).
        upsilon_wt = L({upsilon_weight})
        σ_ml       = L(0.1 * log(10))
        μ_disk     = log(L(0.5))
        μ_bulge    = log(L(0.7))

        # Cuspiness parameter grid (linearly spaced, per-galaxy optimised)
        gamma_lo = T({gamma_lo})
        gamma_hi = T({gamma_hi})
        n_gamma  = {n_gamma_grid}

        # Physics penalty weights
        weight_monotone = L({weight_monotone})
        weight_nonneg   = L({weight_nonneg})

        # Physics test points (fixed x = r/d values; only X_phys[2,:] = γ updated per galaxy)
        # x=0.01 catches peaks pushed below 0.1 by optimising γ small.
        n_phys = 6
        X_phys = zeros(T, 5, n_phys)
        X_phys[1, :] = [T(0.01), T(0.1), T(0.5), T(1.0), T(3.0), T(10.0)]
        X_phys[3, :] .= T(1); X_phys[4, :] .= T(1); X_phys[5, :] .= T(1)

        # ---- pre-compute per-galaxy index ranges (data sorted by galaxy_code) ----
        g_start = zeros(Int, n_gal)
        g_end   = zeros(Int, n_gal)
        for i in 1:n_pts
            g = Int(dataset.X[2, i])
            if g_start[g] == 0
                g_start[g] = i
            end
            g_end[g] = i
        end

        for g in 1:n_gal
            g_start[g] == 0 && continue
            n_g = g_end[g] - g_start[g] + 1
            n_g < 3 && continue

            rng_g    = g_start[g]:g_end[g]
            r_g      = view(dataset.X, 1, rng_g)
            Vgas2_g  = view(dataset.X, 3, rng_g)
            Vdisk2_g = view(dataset.X, 4, rng_g)
            Vbul2_g  = view(dataset.X, 5, rng_g)
            y_g      = view(dataset.y, rng_g)
            w_g      = view(dataset.weights, rng_g)
            resid_g  = y_g .- Vgas2_g

            # ---- pre-allocate work arrays ONCE per galaxy (outside 2D grid loop) ----
            # X_eval[1, :] = r/d        (updated per d-iteration in eval_candidate_g)
            # X_eval[2, :] = γ          (updated per γ-iteration in the outer loop below)
            # X_eval[3:5, :] = Vgas2, Vdisk2, Vbulge2 (static; expression can access but
            #                  typically ignores these baryonic columns)
            X_eval = Matrix{{T}}(undef, 5, n_g)
            X_eval[3, :]  = Vgas2_g
            X_eval[4, :]  = Vdisk2_g
            X_eval[5, :]  = Vbul2_g

            A           = Matrix{{T}}(undef, n_g, 3)
            A_w         = Matrix{{T}}(undef, n_g, 3)
            A_irls      = Matrix{{T}}(undef, n_g, 3)
            r_work      = Vector{{T}}(undef, n_g)
            w_irls      = Vector{{T}}(undef, n_g)
            W_irls_sqrt = Vector{{T}}(undef, n_g)
            resid_irls  = Vector{{T}}(undef, n_g)
            dm_col      = Vector{{T}}(undef, n_g)

            A[:, 1]   = Vdisk2_g
            A[:, 2]   = Vbul2_g
            W_sqrt    = sqrt.(w_g)
            resid_w   = W_sqrt .* resid_g
            A_w[:, 1] = W_sqrt .* Vdisk2_g
            A_w[:, 2] = W_sqrt .* Vbul2_g

            # ---- nested helper: evaluate loss at a given log(d) ----
            # X_eval[2, :] must already hold the current γ value (set in outer loop).
            function eval_candidate_g(log_d_val)
                d_val = exp(log_d_val)
                X_eval[1, :] .= r_g ./ d_val
                # X_eval[2, :] holds γ — set by caller before invoking this function.

                f_v, fl = eval_tree_array(tree.trees.f, X_eval, options)
                !fl && return L(Inf)
                any(!isfinite, f_v) && return L(Inf)

                cum_I_loc = T(0); prev_fr2 = T(0); prev_r_loc = T(0)
                for i in 1:n_g
                    r_i    = r_g[i]
                    f_r2_i = f_v[i] * r_i * r_i
                    cum_I_loc  += (f_r2_i + prev_fr2) * T(0.5) * (r_i - prev_r_loc)
                    dm_col[i]   = cum_I_loc / r_i
                    prev_fr2    = f_r2_i
                    prev_r_loc  = r_i
                end
                any(!isfinite, dm_col) && return L(Inf)

                A[:, 3]   .= dm_col
                A_w[:, 3] .= W_sqrt .* dm_col
                maximum(abs, A) < T(1e-10) && return L(Inf)

                par = try
                    A_w \\ resid_w
                catch
                    return L(Inf)
                end
                par = max.(par, L(0))

                for _ in 1:{n_irls}
                    r_work .= A * par
                    r_work .-= resid_g
                    w_irls .= w_g .* ν1 ./ (ν .+ w_g .* r_work .^ 2)
                    W_irls_sqrt .= sqrt.(w_irls)
                    A_irls[:, 1] .= W_irls_sqrt .* Vdisk2_g
                    A_irls[:, 2] .= W_irls_sqrt .* Vbul2_g
                    A_irls[:, 3] .= W_irls_sqrt .* dm_col
                    resid_irls .= W_irls_sqrt .* resid_g
                    new_par = try
                        A_irls \\ resid_irls
                    catch
                        break
                    end
                    par = max.(new_par, L(0))
                end

                r_work .= A * par
                r_work .-= resid_g
                st = sum(log.(L(1) .+ w_g .* r_work .^ 2 ./ ν)) / n_g
                (isinf(st) || isnan(st)) && return L(Inf)

                pr = L(0)
                if upsilon_wt > L(0)
                    a_opt = par[1]; b_opt = par[2]
                    if a_opt > L(0)
                        pr += (log(a_opt) - μ_disk)^2 / (2 * σ_ml^2)
                    else
                        pr += L(1000) * a_opt^2
                    end
                    has_bulge = maximum(abs, Vbul2_g) > T(1e-6)
                    if has_bulge && b_opt > L(0)
                        pr += (log(b_opt) - μ_bulge)^2 / (2 * σ_ml^2)
                    elseif has_bulge
                        pr += L(1000) * b_opt^2
                    end
                end
                return st + upsilon_wt * pr
            end   # eval_candidate_g

            # ---- 2D grid search: outer loop over γ, inner loop over d ----
            # γ is linearly spaced in [gamma_lo, gamma_hi].
            # For each γ value, X_eval[2, :] is set so the expression h(#1, #2)
            # evaluates as h(r/d, γ) for all data points of this galaxy.
            best_g     = L(Inf)
            best_log_d = log(T(1.0))
            best_gamma = (gamma_lo + gamma_hi) / T(2)

            for j in 1:n_gamma
                gamma_j = n_gamma == 1 ?
                          (gamma_lo + gamma_hi) / T(2) :
                          gamma_lo + (gamma_hi - gamma_lo) * T(j - 1) / T(n_gamma - 1)
                X_eval[2, :] .= gamma_j
                for log_d in range(log(T(0.05)), log(T(500.0)); length={n_d_grid})
                    c = eval_candidate_g(log_d)
                    if c < best_g
                        best_g     = c
                        best_log_d = log_d
                        best_gamma = gamma_j
                    end
                end
            end

            # ---- Fine local d-refinement at best γ ----
            # Fix γ = best_gamma, scan 20 points within ±1 coarse d-step.
            # Same rationale as density.py: near-continuous d-optimisation prevents
            # BFGS-tunable denominator constants from gaining an artificial edge over
            # exact constant-free forms.
            X_eval[2, :] .= best_gamma
            Δlog   = log(T(500.0) / T(0.05)) / ({n_d_grid} - 1)
            lo_ref = max(log(T(0.05)), best_log_d - Δlog)
            hi_ref = min(log(T(500.0)), best_log_d + Δlog)
            for log_d in range(lo_ref, hi_ref; length=20)
                c = eval_candidate_g(log_d)
                if c < best_g; best_g = c; best_log_d = log_d; end
            end

            # ---- Fine local γ-refinement at best_log_d ----
            # Mirrors d-refinement: scan n_gamma_refine points within ±1 coarse γ-step.
            Δgamma_step = n_gamma > 1 ? (gamma_hi - gamma_lo) / T(n_gamma - 1) : T(0.25)
            lo_ref_g = max(gamma_lo, best_gamma - Δgamma_step)
            hi_ref_g = min(gamma_hi, best_gamma + Δgamma_step)
            for gamma_j in range(lo_ref_g, hi_ref_g; length={n_gamma_refine})
                X_eval[2, :] .= gamma_j
                c = eval_candidate_g(best_log_d)
                if c < best_g
                    best_g     = c
                    best_gamma = gamma_j
                end
            end
            X_eval[2, :] .= best_gamma   # restore

            # ---- Physics penalties at optimal (d, γ) ----
            # Evaluated at 5 fixed x = r/d test points covering the profile range.
            # Monotonicity penalty penalises h(x[k+1]) > h(x[k]) (non-decreasing density).
            # Non-negativity penalty penalises h < 0.
            # Applied after refinement so the penalty reflects the true optimal parameters.
            if weight_monotone > L(0) || weight_nonneg > L(0)
                X_phys[2, :] .= best_gamma
                h_phys, ok_phys = eval_tree_array(tree.trees.f, X_phys, options)
                if ok_phys && all(isfinite, h_phys)
                    neg_pen  = sum(max(-v, T(0)) for v in h_phys)
                    mono_pen = L(0)
                    for k in 1:n_phys-1
                        mono_pen += max(h_phys[k+1] - h_phys[k], T(0))
                    end
                    best_g += weight_nonneg * neg_pen + weight_monotone * mono_pen
                end
            end

            isinf(best_g) && (best_g = L(1e12))
            total_loss += best_g
        end

        return total_loss / n_gal
    end
    """

    template = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["r", "gamma", "Vgas2", "Vdisk2", "Vbulge2"],
        combine="f(r, gamma)",   # display: density shape h(r/d, γ), params recovered post-hoc
    )

    # Default operators: include log so the SR can build x^γ via exp(γ·log(x)).
    # Without log, γ can only enter as a multiplicative rescaling (e.g. γ/(γ+x²) = pISO
    # at d_eff=d/√γ), which adds no new physics. With log, Einasto-type exp(-x^γ) and
    # gNFW-type x^{-γ}/(1+x)^{3-γ} become discoverable.
    _unary = unary_operators if unary_operators is not None else \
             ["sqrt", "log", "log1p", "exp"]

    _all_nested = {
        "atan":  {"atan": 0, "log": 0},
        "log":   {"log": 0, "atan": 0, "log1p": 0},
        "log1p": {"log1p": 0},
        "sqrt":  {"log": 0},
        "exp":   {"exp": 0},
    }
    _nested = {k: {ki: vi for ki, vi in v.items() if ki in _unary}
               for k, v in _all_nested.items() if k in _unary}

    model = PySRRegressor(
        expression_spec=template,
        output_directory=output_directory,
        niterations=iterations,
        binary_operators=["*", "/", "-", "+"],
        unary_operators=_unary,
        nested_constraints=_nested,
        maxsize=22,
        populations=populations,
        population_size=population_size,
        ncycles_per_iteration=ncycles_per_iteration,
        weight_optimize=weight_optimize,
        optimizer_iterations=optimizer_iterations,
        batching=False,
        complexity_of_constants=3,
        loss_function_expression=inner_opt_loss,
        guesses=guesses,
        fraction_replaced_guesses=fraction_replaced_guesses,
    )

    if error_weighting:
        errV_safe = np.maximum(df["errV_km/s"], 0.5)
        weights = 1.0 / (2 * df["Vobs_km/s"] * errV_safe) ** 2
    else:
        weights = np.ones(len(df))
    model.fit(X, y, weights=weights)


def recover_parameters_density_2param(df: pd.DataFrame,
                                       h_callable,
                                       n_d_grid: int = 200,
                                       n_gamma_grid: int = 50,
                                       gamma_range: tuple = (0.0, 2.0),
                                       upsilon_weight: float = 0.0) -> pd.DataFrame:
    """
    Post-hoc parameter recovery for a density shape h(x, gamma) discovered by
    fit_density_2param.

    Given h as a callable h(x: np.ndarray, gamma: float) -> np.ndarray, recovers
    optimal (upsilon_disk, upsilon_bulge, c_DM, d_kpc, gamma) per galaxy via a
    2D grid search over (d, gamma) followed by WLS.

    Returns a DataFrame with columns:
        galaxy, upsilon_disk, upsilon_bulge, c_DM, d_kpc, gamma, wls

    Example
    -------
    >>> # generalised NFW
    >>> h = lambda x, g: x**(-g) / (1.0 + x)**(3.0 - g)
    >>> params = recover_parameters_density_2param(df, h)
    """
    results = []
    gamma_vals = np.linspace(gamma_range[0], gamma_range[1], n_gamma_grid)

    for g, gdf in df.groupby("galaxy"):
        gdf = gdf.sort_values("Rad_kpc")
        r     = gdf["Rad_kpc"].values.astype(float)
        y_obs = gdf["Vobs_km/s"].values ** 2
        vgas2 = np.sign(gdf["Vgas_km/s"].values) * gdf["Vgas_km/s"].values ** 2
        vd2   = gdf["Vdisk_km/s"].values ** 2
        vb2   = gdf["Vbul_km/s"].values ** 2
        errV  = np.maximum(gdf["errV_km/s"].values, 0.5)
        w     = 1.0 / (2 * gdf["Vobs_km/s"].values * errV) ** 2
        resid = y_obs - vgas2

        best = (np.inf, None)
        for gamma in gamma_vals:
            for d in np.exp(np.linspace(np.log(0.05), np.log(500.0), n_d_grid)):
                try:
                    h_vals = h_callable(r / d, gamma)
                except Exception:
                    continue
                if not np.all(np.isfinite(h_vals)):
                    continue
                integrand = h_vals * r ** 2
                r_nodes   = np.concatenate([[0.0], r])
                ig_nodes  = np.concatenate([[0.0], integrand])
                cum_I     = cumulative_trapezoid(ig_nodes, r_nodes, initial=0.0)[1:]
                dm_col    = cum_I / r
                if not np.all(np.isfinite(dm_col)):
                    continue
                A = np.column_stack([vd2, vb2, dm_col])
                if np.max(np.abs(A)) < 1e-10:
                    continue
                try:
                    A_w = (w ** 0.5)[:, None] * A
                    r_w = (w ** 0.5) * resid
                    abc, *_ = np.linalg.lstsq(A_w, r_w, rcond=None)
                    abc = np.maximum(abc, 0.0)
                    wls = float(np.sum(w * (A @ abc - resid) ** 2) / len(r))
                    if wls < best[0]:
                        best = (wls, dict(galaxy=g, upsilon_disk=abc[0],
                                          upsilon_bulge=abc[1], c_DM=abc[2],
                                          d_kpc=d, gamma=gamma, wls=wls))
                except Exception:
                    pass

        if best[1] is not None:
            results.append(best[1])

    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from datawrangling import produce_SPARC_df
    df = produce_SPARC_df("data/SPARC", quality=1)

    # ---- Stage 7 production: 99 Q=1 galaxies, physics-constrained ----
    # Monotonicity + non-negativity penalties steer SR away from unphysical
    # peaked density forms (pathology of Stage 6).
    # Test points include x=0.01 to close the small-γ loophole.
    # Expected runtime: ~3–4 days.
    os.makedirs("outputs/SPARC/stage7_phys_constrained", exist_ok=True)
    fit_density_2param(
        df,
        output_directory="outputs/SPARC/stage7_phys_constrained",
        error_weighting=True,
        iterations=99999,
        n_galaxies=None,
        n_d_grid=15,
        n_gamma_grid=8,
        gamma_range=(0.0, 2.0),
        populations=15,
        population_size=40,
        ncycles_per_iteration=100,
        weight_optimize=0.1,
        optimizer_iterations=3,
        upsilon_weight=1.0,
        nu_t=3.0,
        n_irls=3,
        min_points=5,
        unary_operators=["sqrt", "log", "log1p", "exp"],
        weight_monotone=0.1,
        weight_nonneg=0.01,
        n_gamma_refine=10,
    )

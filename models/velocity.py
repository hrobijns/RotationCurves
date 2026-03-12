from pysr import PySRRegressor, TemplateExpressionSpec
import pandas as pd
import numpy as np


def fit_vr_inner_opt(df: pd.DataFrame,
                     output_directory: str = "outputs",
                     error_weighting: bool = True,
                     iterations: int = 99999,
                     n_galaxies: int | None = 5,
                     n_d_grid: int = 50,
                     n_d_refine: int = 20,
                     populations: int = 15,
                     population_size: int = 40,
                     ncycles_per_iteration: int = 500,
                     weight_optimize: float = 0.0,
                     optimizer_iterations: int = 10,
                     upsilon_weight: float = 1.0,
                     origin_weight: float = 1.0,
                     nu_t: float = 3.0,
                     n_irls: int = 3,
                     min_points: int = 5,
                     unary_operators: list = None,
                     maxsize: int = 22):
    """
    Symbolic regression for galaxy rotation curves using per-expression inner
    optimisation: for each candidate symbolic form f, the per-galaxy parameters
    (upsilon_disk a, upsilon_bulge b, DM amplitude c, scale radius d) are found
    analytically / via 1-D grid search.

    Model structure (per data point i in galaxy g):
        V²_obs = sign(Vgas)·V²_gas + a[g]·V²_disk + b[g]·V²_bul + c[g]·f(r/d[g])

    For fixed d[g], the model is LINEAR in a[g], b[g], c[g].  We solve via
    Iteratively Reweighted Least Squares (IRLS) for a Student-t(ν=nu_t) likelihood,
    using measurement-error weights w_i = 1/(2·Vobs_i·errV_i)².  The IRLS starts
    from a Gaussian WLS solution and refines it over n_irls iterations.  d[g] is
    chosen by a log-spaced grid search.  Galaxies are independent.

    Log-normal priors on upsilon_disk and upsilon_bulge (matching mcmc_fit.py:
    lnN(ln 0.5, 0.1 dex) and lnN(ln 0.7, 0.1 dex)) are added as penalty terms
    weighted by upsilon_weight (default 1.0 = always active).
    """
    if n_galaxies is not None:
        available = df["galaxy"].unique()
        rng = np.random.default_rng(seed=42)
        selected = rng.choice(available, size=min(n_galaxies, len(available)), replace=False)
        df = df[df["galaxy"].isin(selected)].copy()

    # drop galaxies with too few data points for the 3-parameter solve to be stable
    counts = df.groupby("galaxy")["galaxy"].transform("count")
    df = df[counts >= min_points].copy()

    galaxy_codes, galaxies = pd.factorize(df["galaxy"])
    df["galaxy_code"] = galaxy_codes + 1  # Julia is 1-indexed
    
    df = df.sort_values("galaxy_code").reset_index(drop=True)
    # sort by galaxy_code so Julia can use O(n_total) index scan instead of
    # O(n_gal × n_total) BitVector masks.

    y = df["Vobs_km/s"] ** 2
    X = pd.DataFrame({
        "r":       df["Rad_kpc"],
        "galaxy":  df["galaxy_code"],
        "Vgas2":   np.sign(df["Vgas_km/s"]) * df["Vgas_km/s"] ** 2,
        "Vdisk2":  df["Vdisk_km/s"] ** 2,
        "Vbulge2": df["Vbul_km/s"] ** 2,
    })

    # -------------------------------------------------------------------------
    # Custom loss: for each candidate f, find optimal (a, b, c, d) per galaxy.
    #
    # Step 1 – outer loop: for each galaxy g, sweep d on a log grid.
    # Step 2 – inner step: given d, evaluate f(r/d) then solve IRLS for a,b,c.
    # Step 3 – return total Student-t NLL at the best d for each galaxy.
    #
    # IRLS (Iteratively Reweighted Least Squares) for Student-t(ν):
    #   IRLS weight: w_irls_i = w_i * (ν+1) / (ν + w_i * r²_i)
    #   where r_i = V²_pred_i - V²_obs_i.  Converges to the Student-t MLE.
    #
    # Student-t NLL: Σ log(1 + w_i * r²_i / ν) / n_g
    #   (proportional to the Student-t negative log-likelihood in V²-space)
    #
    # `tree.trees.f`  accesses the f sub-expression from the TemplateExpression,
    # letting us evaluate f at custom inputs (r/d) independently of the template.
    # -------------------------------------------------------------------------
    inner_opt_loss = f"""
    function physicsloss(tree, dataset::Dataset{{T,L}}, options)::L where {{T,L}}
        n_pts      = length(dataset.y)
        n_gal      = Int(maximum(dataset.X[2, :]))
        total_loss = L(0)

        # Student-t degrees of freedom (matching mcmc_fit.py: ν=3)
        ν  = L({nu_t})
        ν1 = ν + L(1)   # ν + 1, used in IRLS weight: w_irls = w * (ν+1) / (ν + w*r²)

        # Log-normal priors on upsilon_disk (a) and upsilon_bulge (b),
        # matching mcmc_fit.py: lnN(ln 0.5, 0.1 dex) and lnN(ln 0.7, 0.1 dex).
        upsilon_wt = L({upsilon_weight})
        σ_ml       = L(0.1 * log(10))   # 0.1 dex in natural log units
        μ_disk     = log(L(0.5))
        μ_bulge    = log(L(0.7))

        # ---- zero-at-origin penalty ----
        # Physical DM profiles satisfy f(0) = 0 (enclosed mass → 0 as r → 0).
        # Evaluate f at a tiny x = 0.001 (well below any r/d in the data) and
        # penalise |f(x_small)|.  Capped at 1000 to avoid Inf from divergent forms.
        # Rows 2-5 of X_tiny are unused by f; set to zero.
        X_tiny = zeros(T, 5, 1)
        X_tiny[1, 1] = T(0.001)
        f_tiny_vec, ok_tiny = eval_tree_array(tree.trees.f, X_tiny, options)
        origin_val     = (ok_tiny && isfinite(f_tiny_vec[1])) ? abs(f_tiny_vec[1]) : L(1000)
        origin_penalty = min(origin_val, L(1000))

        # ---- pre-compute per-galaxy index ranges (data sorted by galaxy_code) ----
        # O(n_pts) scan replaces O(n_gal × n_pts) BitVector masks per expression eval.
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
            g_start[g] == 0 && continue   # galaxy absent from dataset
            n_g = g_end[g] - g_start[g] + 1
            n_g < 3 && continue   # need at least 3 points for 3-parameter solve

            rng_g    = g_start[g]:g_end[g]
            r_g      = view(dataset.X, 1, rng_g)
            Vgas2_g  = view(dataset.X, 3, rng_g)   # sign(Vgas)·Vgas² (signed)
            Vdisk2_g = view(dataset.X, 4, rng_g)
            Vbul2_g  = view(dataset.X, 5, rng_g)
            y_g      = view(dataset.y, rng_g)
            w_g      = view(dataset.weights, rng_g)  # 1/(2·Vobs·errV)²
            resid_g  = y_g .- Vgas2_g   # target: a·Vd² + b·Vb² + c·f

            # ---- pre-allocate work arrays ONCE per galaxy (outside d-loop) ----
            X_eval = Matrix{{T}}(undef, 5, n_g)
            X_eval[2, :] .= T(g)
            X_eval[3, :]  = Vgas2_g
            X_eval[4, :]  = Vdisk2_g
            X_eval[5, :]  = Vbul2_g

            A           = Matrix{{T}}(undef, n_g, 3)   # design matrix [Vd², Vb², f]
            A_w         = Matrix{{T}}(undef, n_g, 3)   # sqrt(w)-weighted (initial WLS)
            A_irls      = Matrix{{T}}(undef, n_g, 3)   # IRLS-weighted design matrix
            r_work      = Vector{{T}}(undef, n_g)       # residual work vector
            w_irls      = Vector{{T}}(undef, n_g)       # Student-t IRLS weights
            W_irls_sqrt = Vector{{T}}(undef, n_g)       # sqrt(w_irls)
            resid_irls  = Vector{{T}}(undef, n_g)       # IRLS-weighted residual

            A[:, 1]   = Vdisk2_g
            A[:, 2]   = Vbul2_g
            W_sqrt    = sqrt.(w_g)
            resid_w   = W_sqrt .* resid_g
            A_w[:, 1] = W_sqrt .* Vdisk2_g
            A_w[:, 2] = W_sqrt .* Vbul2_g

            best_g = L(Inf)
            best_log_d = log(T(1.0))   # fallback initial value

            # ---- inner evaluator: WLS→IRLS→Student-t loss at a given log_d ----
            # Captures pre-allocated work arrays from enclosing galaxy scope.
            function eval_candidate_f(log_d_val)
                d_val = exp(log_d_val)
                X_eval[1, :] .= r_g ./ d_val
                f_vals, flag = eval_tree_array(tree.trees.f, X_eval, options)
                !flag && return L(Inf)
                any(!isfinite, f_vals) && return L(Inf)
                A[:, 3]   .= f_vals
                A_w[:, 3] .= W_sqrt .* f_vals
                maximum(abs, A) < T(1e-10) && return L(Inf)
                params = try
                    A_w \\ resid_w
                catch
                    return L(Inf)
                end
                params = max.(params, L(0))
                for _ in 1:{n_irls}
                    r_work .= A * params
                    r_work .-= resid_g
                    w_irls .= w_g .* ν1 ./ (ν .+ w_g .* r_work .^ 2)
                    W_irls_sqrt .= sqrt.(w_irls)
                    A_irls[:, 1] .= W_irls_sqrt .* Vdisk2_g
                    A_irls[:, 2] .= W_irls_sqrt .* Vbul2_g
                    A_irls[:, 3] .= W_irls_sqrt .* f_vals
                    resid_irls .= W_irls_sqrt .* resid_g
                    new_params = try
                        A_irls \\ resid_irls
                    catch
                        break
                    end
                    params = max.(new_params, L(0))
                end
                r_work .= A * params
                r_work .-= resid_g
                st_loss_g = sum(log.(L(1) .+ w_g .* r_work .^ 2 ./ ν)) / n_g
                (isinf(st_loss_g) || isnan(st_loss_g)) && return L(Inf)
                prior_g = L(0)
                if upsilon_wt > L(0)
                    a_opt = params[1]; b_opt = params[2]
                    if a_opt > L(0)
                        prior_g += (log(a_opt) - μ_disk)^2 / (2 * σ_ml^2)
                    else
                        prior_g += L(1000) * a_opt^2
                    end
                    has_bulge = maximum(abs, Vbul2_g) > T(1e-6)
                    if has_bulge && b_opt > L(0)
                        prior_g += (log(b_opt) - μ_bulge)^2 / (2 * σ_ml^2)
                    elseif has_bulge
                        prior_g += L(1000) * b_opt^2
                    end
                end
                return st_loss_g + upsilon_wt * prior_g
            end

            # ---- coarse grid search over scale radius d ----
            for log_d in range(log(T(0.05)), log(T(500.0)); length={n_d_grid})
                c = eval_candidate_f(log_d)
                if c < best_g; best_g = c; best_log_d = log_d; end
            end

            # ---- fine d-refinement (±1 coarse step around best_log_d) ----
            log_d_step = (log(T(500.0)) - log(T(0.05))) / T({n_d_grid} - 1)
            lo_ref = best_log_d - log_d_step
            hi_ref = best_log_d + log_d_step
            for log_d in range(lo_ref, hi_ref; length={n_d_refine})
                c = eval_candidate_f(log_d)
                if c < best_g; best_g = c; end
            end

            isinf(best_g) && (best_g = L(1e12))
            total_loss += best_g
        end

        return total_loss / n_gal + L({origin_weight}) * origin_penalty
    end
    """

    template = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["r", "galaxy", "Vgas2", "Vdisk2", "Vbulge2"],
        # No per-galaxy parameters: they are found analytically in the custom loss
        # and never used by the Julia template machinery.  Removing them eliminates
        # the SVector{N_galaxies} Julia type whose compilation scales super-linearly.
        combine="f(r)",   # display: raw DM profile shape f(r/d) ← d recovered post-hoc
    )

    active_unary = unary_operators if unary_operators is not None else ["atan", "log", "sqrt", "log1p"]
    _all_nested = {
        "atan":  {"atan": 0, "log": 0},
        "log":   {"log": 0, "atan": 0, "log1p": 0},
        "log1p": {"log1p": 0},
        "sqrt":  {"log": 0},
    }
    nested = {
        op: {t: d for t, d in targets.items() if t in active_unary}
        for op, targets in _all_nested.items()
        if op in active_unary
    }

    model = PySRRegressor(
        expression_spec=template,
        output_directory=output_directory,
        niterations=iterations,
        binary_operators=["*", "/", "-", "+"],
        unary_operators=active_unary,
        nested_constraints=nested,
        maxsize=maxsize,
        populations=populations,
        population_size=population_size,
        ncycles_per_iteration=ncycles_per_iteration,
        weight_optimize=weight_optimize,   # If > 0, BFGS tunes constants *within* f
                                           # (e.g. the "1" in 1−atan(x)/x).
                                           # Template params a,b,c,d have zero gradient
                                           # in this loss so BFGS leaves them untouched.
        optimizer_iterations=optimizer_iterations,
        batching=False,
        complexity_of_constants = 3,
        loss_function_expression=inner_opt_loss,
    )

    if error_weighting:
        errV_safe = np.maximum(df["errV_km/s"], 0.5)   # floor at 0.5 km/s (mcmc_fit.py convention)
        weights = 1.0 / (2 * df["Vobs_km/s"] * errV_safe) ** 2
    else:
        weights = np.ones(len(df))  # uniform → reduces to plain OLS
    # Always pass weights so dataset.weights is never `nothing` in Julia.
    model.fit(X, y, weights=weights)


def recover_parameters(df: pd.DataFrame,
                       f_callable,
                       n_d_grid: int = 200,
                       upsilon_weight: float = 0.0) -> pd.DataFrame:
    """
    Post-hoc parameter recovery for the best symbolic form found by fit_vr_inner_opt.

    Given the discovered DM profile shape as a Python callable f(x) (where x = r/d),
    re-runs the same inner optimisation (log-spaced d-grid + weighted OLS) in Python
    to find the optimal (upsilon_disk, upsilon_bulge, c_DM, d_kpc) per galaxy.

    Returns a DataFrame with columns galaxy, upsilon_disk, upsilon_bulge, c_DM, d_kpc, wls
    — directly comparable to outputs/fits/pISO.csv from mcmc_fit.py.

    Example
    -------
    >>> f = lambda x: (x - np.arctan(x)) / x   # best complexity-6 expression
    >>> params = recover_parameters(df, f)
    """
    results = []
    for g, gdf in df.groupby("galaxy"):
        r     = gdf["Rad_kpc"].values.astype(float)
        y     = gdf["Vobs_km/s"].values ** 2
        vgas2 = np.sign(gdf["Vgas_km/s"].values) * gdf["Vgas_km/s"].values ** 2
        vd2   = gdf["Vdisk_km/s"].values ** 2
        vb2   = gdf["Vbul_km/s"].values ** 2
        errV  = np.maximum(gdf["errV_km/s"].values, 0.5)
        w     = 1.0 / (2 * gdf["Vobs_km/s"].values * errV) ** 2
        resid = y - vgas2

        best = (np.inf, None)
        for d in np.exp(np.linspace(np.log(0.05), np.log(500.0), n_d_grid)):
            fv = f_callable(r / d)
            if not np.all(np.isfinite(fv)):
                continue
            A = np.column_stack([vd2, vb2, fv])
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
                                      d_kpc=d, wls=wls))
            except Exception:
                pass

        if best[1] is not None:
            results.append(best[1])

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Open-ended production run: real SPARC data, quality Q=1 (99 galaxies).
    # Uses Student-t(ν=3) likelihood + log-normal priors on upsilons, matching
    # mcmc_fit.py.  Expanded operator set (log, sqrt, exp, log1p) allows the
    # search to find any profile shape, not just the pISO atan family.
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from datawrangling import produce_SPARC_df
    df = produce_SPARC_df("data/SPARC", quality=1)
    fit_vr_inner_opt(
        df,
        output_directory="outputs/SPARC/stage8_velocity",
        error_weighting=True,
        iterations=99999,
        n_galaxies=None,   # use all Q=1 galaxies
        n_d_grid=30,
        n_d_refine=20,
        populations=15,
        population_size=40,
        ncycles_per_iteration=100,
        weight_optimize=0.1,
        optimizer_iterations=3,
        upsilon_weight=1.0,
        origin_weight=1.0,
        nu_t=3.0,
        n_irls=3,
        min_points=5,
        unary_operators=["atan", "log", "sqrt", "log1p"],   # no exp: origin penalty kills it
        maxsize=22,
    )

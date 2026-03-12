from pysr import PySRRegressor, TemplateExpressionSpec
import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid


def fit_density_inner_opt(df: pd.DataFrame,
                          output_directory: str = "outputs",
                          error_weighting: bool = True,
                          iterations: int = 99999,
                          n_galaxies: int | None = 5,
                          n_d_grid: int = 50,
                          populations: int = 15,
                          population_size: int = 40,
                          ncycles_per_iteration: int = 500,
                          weight_optimize: float = 0.0,
                          optimizer_iterations: int = 10,
                          upsilon_weight: float = 1.0,
                          nu_t: float = 3.0,
                          n_irls: int = 3,
                          min_points: int = 5,
                          unary_operators: list[str] | None = None,
                          guesses: list | None = None,
                          fraction_replaced_guesses: float = 0.001):
    """
    Symbolic regression for galaxy rotation curves: learns the DM *density*
    profile shape h(x) where x = r/d, rather than the velocity-squared shape.

    Model structure (per data point i in galaxy g):
        V²_obs = sign(Vgas)·V²_gas + a[g]·V²_disk + b[g]·V²_bul + c[g]·DM_col[i]

    where the DM column is built by spherical mass integration:
        DM_col[i] = (1/r_i) · ∫₀^{r_i} h(r'/d[g]) · r'² dr'

    This converts the symbolic density shape h(x) to a velocity-squared
    contribution via the spherical Jeans / circular-velocity relation
        V²_DM(r) = G·M(r)/r = (4πG·ρ₀·d³ / r) · ∫₀^{r/d} h(x) x² dx.
    The constant c[g] absorbs 4πG·ρ₀·d³.

    For fixed d[g], the model is LINEAR in a[g], b[g], c[g].  We solve via
    Iteratively Reweighted Least Squares (IRLS) for a Student-t(ν=nu_t)
    likelihood, using measurement-error weights w_i = 1/(2·Vobs_i·errV_i)².
    d[g] is chosen by a log-spaced grid search followed by local refinement.

    Physical motivation
    -------------------
    Standard density profiles are simpler expressions than their velocity
    counterparts:
        pISO:  h(x) = 1/(1+x²)            complexity ~5
               vs.  f(x) = 1−atan(x)/x    complexity ~6
        NFW:   h(x) = 1/(x·(1+x)²)        complexity ~7
               vs.  f(x) = [ln(1+x)−x/(1+x)]/x  complexity ~10
    The zero-at-origin constraint V²_DM(0)=0 is automatically satisfied by
    the integration, so no origin penalty is needed.

    NOTE: data is sorted by (galaxy_code, r) in Python before fitting.
    This is required so the Julia trapezoid loop can iterate i=1:n_g in
    ascending radius order within each galaxy.
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

    # Sort by (galaxy_code, r) — essential for the cumulative trapezoid in Julia.
    # Within each galaxy, data points must be in ascending radius order.
    df = df.sort_values(["galaxy_code", "Rad_kpc"]).reset_index(drop=True)

    y = df["Vobs_km/s"] ** 2
    X = pd.DataFrame({
        "r":       df["Rad_kpc"],
        "galaxy":  df["galaxy_code"],
        "Vgas2":   np.sign(df["Vgas_km/s"]) * df["Vgas_km/s"] ** 2,
        "Vdisk2":  df["Vdisk_km/s"] ** 2,
        "Vbulge2": df["Vbul_km/s"] ** 2,
    })

    # -------------------------------------------------------------------------
    # Custom loss: for each candidate density shape h, find optimal (a,b,c,d).
    #
    # Step 1 – outer loop: for each galaxy g, sweep d on a log grid.
    # Step 2 – inner step: given d, evaluate h(r/d) then integrate to get
    #          the DM column: dm_col[i] = (1/r_i) ∫₀^{r_i} h(r'/d) r'² dr'
    #          using cumulative trapezoid with a virtual point at (r=0, h·r²=0).
    # Step 3 – use dm_col as the third column of the design matrix and solve
    #          IRLS for a, b, c (same as velocity.py).
    # Step 4 – fine local refinement: 20 evaluations around the best grid d.
    #          This gives near-continuous d-optimisation, so that exact constant-
    #          free forms (e.g. 1/(1+x²)) achieve loss ≈ 0 on noiseless data and
    #          are NOT penalised vs. forms with BFGS-tunable denominator constants
    #          (which are just pISO at a rescaled d and would otherwise win on any
    #          coarse grid).
    # Step 5 – return total Student-t NLL at the best d for each galaxy.
    #
    # No origin penalty: V²_DM(0)=0 is automatically satisfied because the
    # cumulative integral from 0 to 0 is zero.
    #
    # Data MUST be sorted by (galaxy_code, r) before this loss is called
    # (done in Python above).  The Julia loop iterates i=1:n_g in radius order.
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
            resid_g  = y_g .- Vgas2_g   # target: a·Vd² + b·Vb² + c·dm_col

            # ---- pre-allocate work arrays ONCE per galaxy (outside d-loop) ----
            X_eval = Matrix{{T}}(undef, 5, n_g)
            X_eval[2, :] .= T(g)
            X_eval[3, :]  = Vgas2_g
            X_eval[4, :]  = Vdisk2_g
            X_eval[5, :]  = Vbul2_g

            A           = Matrix{{T}}(undef, n_g, 3)   # design matrix [Vd², Vb², dm_col]
            A_w         = Matrix{{T}}(undef, n_g, 3)   # sqrt(w)-weighted (initial WLS)
            A_irls      = Matrix{{T}}(undef, n_g, 3)   # IRLS-weighted design matrix
            r_work      = Vector{{T}}(undef, n_g)       # residual work vector
            w_irls      = Vector{{T}}(undef, n_g)       # Student-t IRLS weights
            W_irls_sqrt = Vector{{T}}(undef, n_g)       # sqrt(w_irls)
            resid_irls  = Vector{{T}}(undef, n_g)       # IRLS-weighted residual
            dm_col      = Vector{{T}}(undef, n_g)       # cumulative-trapezoid DM column

            A[:, 1]   = Vdisk2_g
            A[:, 2]   = Vbul2_g
            W_sqrt    = sqrt.(w_g)
            resid_w   = W_sqrt .* resid_g
            A_w[:, 1] = W_sqrt .* Vdisk2_g
            A_w[:, 2] = W_sqrt .* Vbul2_g

            # ---- nested helper: evaluate loss at a given log(d) ----
            # Returns L(Inf) on failure.  Captures all per-galaxy arrays.
            # Note: A[:, 3], A_w[:, 3], dm_col are mutated on each call.
            function eval_candidate_g(log_d_val)
                d_val = exp(log_d_val)
                X_eval[1, :] .= r_g ./ d_val

                # Evaluate density shape h(r/d)
                f_v, fl = eval_tree_array(tree.trees.f, X_eval, options)
                !fl && return L(Inf)
                any(!isfinite, f_v) && return L(Inf)

                # Cumulative trapezoid with virtual origin (r=0, h·r²=0)
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

            # ---- Coarse grid search: log-spaced d ----
            best_g     = L(Inf)
            best_log_d = log(T(1.0))
            for log_d in range(log(T(0.05)), log(T(500.0)); length={n_d_grid})
                c = eval_candidate_g(log_d)
                if c < best_g
                    best_g     = c
                    best_log_d = log_d
                end
            end

            # ---- Fine local refinement around best_log_d ----
            # Scan 20 points within one coarse grid step of best_log_d.
            # This removes the BFGS-constant advantage: forms like c/(x²+a)
            # are exactly pISO at rescaled d — with near-continuous d-opt,
            # the exact constant-free form 1/(1+x²) ties them on loss and
            # PySR's complexity criterion correctly picks the simpler one.
            Δlog   = log(T(500.0) / T(0.05)) / ({n_d_grid} - 1)
            lo_ref = max(log(T(0.05)), best_log_d - Δlog)
            hi_ref = min(log(T(500.0)), best_log_d + Δlog)
            for log_d in range(lo_ref, hi_ref; length=20)
                c = eval_candidate_g(log_d)
                if c < best_g; best_g = c; end
            end

            isinf(best_g) && (best_g = L(1e12))
            total_loss += best_g
        end

        return total_loss / n_gal
    end
    """

    template = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["r", "galaxy", "Vgas2", "Vdisk2", "Vbulge2"],
        combine="f(r)",   # display: raw density shape h(r/d) ← d recovered post-hoc
    )

    # Default unary operators for full production runs.
    # exp: Gaussian/exponential density profiles are physically motivated.
    # For quick verification tests, pass unary_operators=[] to restrict to
    # pure algebraic forms and avoid cheap exp-based approximations.
    _unary = unary_operators if unary_operators is not None else \
             ["atan", "log", "sqrt", "log1p", "exp"]

    _all_nested = {
        "atan":  {"atan": 0, "log": 0},
        "log":   {"log": 0, "atan": 0, "log1p": 0},
        "log1p": {"log1p": 0},
        "sqrt":  {"log": 0},
        "exp":   {"exp": 0},
    }
    # Only include constraint entries for operators actually in the operator list
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


def recover_parameters_density(df: pd.DataFrame,
                                h_callable,
                                n_d_grid: int = 200,
                                upsilon_weight: float = 0.0) -> pd.DataFrame:
    """
    Post-hoc parameter recovery for the best density shape found by
    fit_density_inner_opt.

    Given the discovered DM density shape as a Python callable h(x) (where
    x = r/d), re-runs the same inner optimisation (log-spaced d-grid + weighted
    OLS) in Python to find the optimal (upsilon_disk, upsilon_bulge, c_DM, d_kpc)
    per galaxy.

    The DM column is computed via cumulative trapezoid (matching the Julia loss):
        dm_col[i] = (1/r_i) ∫₀^{r_i} h(r'/d) r'² dr'

    Returns a DataFrame with columns galaxy, upsilon_disk, upsilon_bulge,
    c_DM, d_kpc, wls.  c_DM absorbs 4πG·ρ₀·d³ (units: (km/s)²/kpc²).

    Example
    -------
    >>> h = lambda x: 1.0 / (1.0 + x**2)   # pISO density shape
    >>> params = recover_parameters_density(df, h)
    """
    results = []
    for g, gdf in df.groupby("galaxy"):
        gdf = gdf.sort_values("Rad_kpc")   # must be r-sorted for trapezoid
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
            h_vals = h_callable(r / d)
            if not np.all(np.isfinite(h_vals)):
                continue
            integrand = h_vals * r ** 2
            # Prepend virtual origin (r=0, integrand=0) for lower bound
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
                                      d_kpc=d, wls=wls))
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
    fit_density_inner_opt(
        df,
        output_directory="outputs/SPARC/stage5_density",
        error_weighting=True,
        iterations=99999,
        n_galaxies=None,        # use all Q=1 galaxies
        n_d_grid=30,            # coarse scan + 20-pt local refinement → adequate d-resolution
        populations=15,
        population_size=40,
        ncycles_per_iteration=100,
        weight_optimize=0.1,
        optimizer_iterations=3,
        upsilon_weight=1.0,
        nu_t=3.0,
        n_irls=3,
        min_points=5,
        unary_operators=["sqrt", "log1p", "exp"],  # physically motivated for density; no atan (velocity artefact), no log (diverges at x=0)
    )

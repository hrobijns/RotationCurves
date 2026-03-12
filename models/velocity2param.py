from pysr import PySRRegressor, TemplateExpressionSpec
import pandas as pd
import numpy as np


def fit_vr_2param(df: pd.DataFrame,
                  output_directory: str = "outputs",
                  error_weighting: bool = True,
                  iterations: int = 99999,
                  n_galaxies: int | None = 5,
                  n_d_grid: int = 15,
                  n_d_refine: int = 20,
                  n_gamma_grid: int = 8,
                  n_gamma_refine: int = 10,
                  gamma_range: tuple = (0.0, 2.0),
                  populations: int = 15,
                  population_size: int = 40,
                  ncycles_per_iteration: int = 100,
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
    2D symbolic regression: learn f(r/d, γ) where γ is a per-galaxy shape parameter
    optimised jointly with scale radius d.

    Model (per data point i in galaxy g):
        V²_obs = sign(Vgas)·V²_gas + a[g]·V²_disk + b[g]·V²_bul + c[g]·f(r/d[g], γ[g])

    Per-galaxy parameters:
      a, b — upsilon_disk, upsilon_bulge (linear, solved via WLS/IRLS)
      c    — DM amplitude (linear)
      d    — DM scale radius (nonlinear, log-spaced grid + refinement)
      γ    — shape parameter ∈ gamma_range (nonlinear, linear-spaced grid + refinement)

    Variable layout in X_eval (passed to eval_tree_array for SR expression f):
      row 1: x = r/d
      row 2: γ  (set per-galaxy during optimisation)
      rows 3-5: Vgas2, Vdisk2, Vbulge2 (present for template machinery; not used by f)

    Note: dataset.X[2,:] still contains integer galaxy codes for O(n_pts) index scan.
    X_eval is a separate matrix used only for eval_tree_array calls.
    """
    if n_galaxies is not None:
        available = df["galaxy"].unique()
        rng = np.random.default_rng(seed=42)
        selected = rng.choice(available, size=min(n_galaxies, len(available)), replace=False)
        df = df[df["galaxy"].isin(selected)].copy()

    counts = df.groupby("galaxy")["galaxy"].transform("count")
    df = df[counts >= min_points].copy()

    galaxy_codes, galaxies = pd.factorize(df["galaxy"])
    df["galaxy_code"] = galaxy_codes + 1
    df = df.sort_values("galaxy_code").reset_index(drop=True)

    y = df["Vobs_km/s"] ** 2
    X = pd.DataFrame({
        "r":       df["Rad_kpc"],
        "galaxy":  df["galaxy_code"],   # row 2 in dataset.X — used for galaxy index scan
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

        ν  = L({nu_t})
        ν1 = ν + L(1)

        upsilon_wt = L({upsilon_weight})
        σ_ml       = L(0.1 * log(10))
        μ_disk     = log(L(0.5))
        μ_bulge    = log(L(0.7))

        gamma_lo = T({gamma_lo})
        gamma_hi = T({gamma_hi})
        n_gamma  = {n_gamma_grid}

        # Pre-allocate origin-check matrix (5×1); row 2 updated per galaxy at best_gamma.
        X_origin = zeros(T, 5, 1)
        X_origin[1, 1] = T(0.001)
        X_origin[3, 1] = T(1); X_origin[4, 1] = T(1); X_origin[5, 1] = T(1)

        # O(n_pts) galaxy index scan (data sorted by galaxy_code in Python).
        g_start = zeros(Int, n_gal)
        g_end   = zeros(Int, n_gal)
        for i in 1:n_pts
            g = Int(dataset.X[2, i])
            if g_start[g] == 0; g_start[g] = i; end
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

            # Pre-allocate work arrays once per galaxy (outside d/γ loops).
            X_eval      = Matrix{{T}}(undef, 5, n_g)
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

            A[:, 1]   = Vdisk2_g
            A[:, 2]   = Vbul2_g
            W_sqrt    = sqrt.(w_g)
            resid_w   = W_sqrt .* resid_g
            A_w[:, 1] = W_sqrt .* Vdisk2_g
            A_w[:, 2] = W_sqrt .* Vbul2_g

            best_g     = L(Inf)
            best_log_d = log(T(1.0))
            best_gamma = T(1.0)

            # ---- inner evaluator: WLS→IRLS→Student-t loss at given log_d ----
            # Reads current γ from X_eval[2,:] (set by outer γ loop or refinement).
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

            # ---- coarse 2D grid: outer γ loop, inner d loop ----
            for gamma_j in range(gamma_lo, gamma_hi; length=n_gamma)
                X_eval[2, :] .= gamma_j
                for log_d in range(log(T(0.05)), log(T(500.0)); length={n_d_grid})
                    c = eval_candidate_f(log_d)
                    if c < best_g
                        best_g     = c
                        best_log_d = log_d
                        best_gamma = gamma_j
                    end
                end
            end

            # ---- fine d-refinement at best_gamma (±1 coarse d step) ----
            X_eval[2, :] .= best_gamma
            log_d_step = (log(T(500.0)) - log(T(0.05))) / T({n_d_grid} - 1)
            lo_ref = best_log_d - log_d_step
            hi_ref = best_log_d + log_d_step
            for log_d in range(lo_ref, hi_ref; length={n_d_refine})
                c = eval_candidate_f(log_d)
                if c < best_g; best_g = c; best_log_d = log_d; end
            end

            # ---- fine γ-refinement at best_log_d (±1 coarse γ step) ----
            Δgamma = n_gamma > 1 ? (gamma_hi - gamma_lo) / T(n_gamma - 1) : T(0.25)
            lo_ref_g = max(gamma_lo, best_gamma - Δgamma)
            hi_ref_g = min(gamma_hi, best_gamma + Δgamma)
            for gamma_j in range(lo_ref_g, hi_ref_g; length={n_gamma_refine})
                X_eval[2, :] .= gamma_j
                c = eval_candidate_f(best_log_d)
                if c < best_g
                    best_g     = c
                    best_gamma = gamma_j
                end
            end
            X_eval[2, :] .= best_gamma   # restore for origin check

            # ---- origin penalty: enforce f(0, γ) ≈ 0 at optimal γ ----
            if L({origin_weight}) > L(0)
                X_origin[2, 1] = best_gamma
                f_orig, ok_orig = eval_tree_array(tree.trees.f, X_origin, options)
                if ok_orig && isfinite(f_orig[1])
                    best_g += L({origin_weight}) * min(abs(f_orig[1]), L(1000))
                else
                    best_g += L({origin_weight}) * L(1000)
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
        combine="f(r, gamma)",
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
        weight_optimize=weight_optimize,
        optimizer_iterations=optimizer_iterations,
        batching=False,
        complexity_of_constants=3,
        loss_function_expression=inner_opt_loss,
    )

    if error_weighting:
        errV_safe = np.maximum(df["errV_km/s"], 0.5)
        weights = 1.0 / (2 * df["Vobs_km/s"] * errV_safe) ** 2
    else:
        weights = np.ones(len(df))
    model.fit(X, y, weights=weights)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from datawrangling import produce_SPARC_df
    df = produce_SPARC_df("data/SPARC", quality=1)
    fit_vr_2param(
        df,
        output_directory="outputs/SPARC/stage9_velocity_2param",
        iterations=99999,
        n_galaxies=None,
        n_d_grid=15,
        n_d_refine=20,
        n_gamma_grid=8,
        n_gamma_refine=10,
        gamma_range=(0.0, 2.0),
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
        unary_operators=["atan", "log", "sqrt", "log1p"],
        maxsize=22,
    )

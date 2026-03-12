"""
Verification run: velocity2param (with new positivity penalty) on 5-galaxy pISO
noiseless toy dataset.

Checks:
  1. Julia code compiles and runs without errors
  2. pISO velocity form f(x) = 1 - atan(x)/x is rediscovered (or near Pareto front)
  3. Positivity penalty is zero for the true pISO form
  4. Origin penalty is near-zero for pISO
  5. Loss is near-zero for noiseless data
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from models.velocity2param import fit_vr_2param

# ── 1. Pre-check: confirm pISO f(x) = 1 - atan(x)/x has zero penalties ──────
x_test = np.array([0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0])
f_piso = 1.0 - np.arctan(x_test) / x_test

print("=" * 60)
print("pISO physics penalty pre-check (should all be 0):")
print(f"  f at test points: {np.round(f_piso, 5)}")
print(f"  weight_nonneg penalty = sum(max(-f,0)) = {np.sum(np.maximum(-f_piso, 0)):.2e}")
print(f"  origin penalty        = |f(x=0.001)|  = {abs(f_piso[0]):.2e}")
print()

# ── 2. Run SR ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/toydatasets/pISO_noiseless.csv")
print(f"Dataset: {df['galaxy'].nunique()} galaxies loaded.")
print("Starting velocity2param verification (5 galaxies, ~200 iterations) …")
print("=" * 60)

os.makedirs("outputs/verify/velocity2param_toydata", exist_ok=True)

fit_vr_2param(
    df,
    output_directory="outputs/verify/velocity2param_toydata",
    error_weighting=True,
    iterations=200,
    n_galaxies=5,
    n_d_grid=15,
    n_d_refine=10,
    n_gamma_grid=4,
    n_gamma_refine=5,
    gamma_range=(0.0, 2.0),
    populations=8,
    population_size=30,
    ncycles_per_iteration=50,
    weight_optimize=0.1,
    optimizer_iterations=3,
    upsilon_weight=0.0,        # disabled for noiseless: simplifies interpretation
    origin_weight=1.0,
    weight_nonneg=0.01,
    nu_t=3.0,
    n_irls=3,
    min_points=3,
    unary_operators=["atan"],  # only need atan for pISO; keep search space small
    maxsize=18,
)

print()
print("=" * 60)
print("Verification complete.  Check HOF above for 1 - atan(r)/r or equivalent.")
print("Expected loss near zero for noiseless data.")

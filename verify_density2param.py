"""
Verification run: density2param log-slope model on 5-galaxy pISO noiseless toy dataset.

Checks:
  1. Julia code compiles and runs without errors
  2. pISO log-slope s(x) = 2x²/(1+x²) is rediscovered (or near Pareto front)
  3. Physics penalties are zero for the true pISO form
  4. Loss is near-zero for noiseless data
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from models.density2param import fit_density_2param

# ── 1. Pre-check: confirm pISO s(x) = 2x²/(1+x²) has zero physics penalties ──
x_test = np.array([0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0])
s_piso = 2 * x_test**2 / (1 + x_test**2)
print("=" * 60)
print("pISO physics penalty pre-check (should both be 0):")
print(f"  s at test points: {np.round(s_piso, 5)}")
print(f"  weight_spos  penalty = sum(max(-s,0)) = {np.sum(np.maximum(-s_piso, 0)):.2e}")
print(f"  inner_slope  penalty = max(s[0]-3, 0) = {max(s_piso[0]-3, 0):.2e}")
print()

# ── 2. Manual h-recovery sanity check ────────────────────────────────────────
# For pISO h(x)=1/(1+x²), verify that integrating s recovers h up to a scale.
r_check = np.array([1.0, 2.0, 5.0, 10.0, 20.0])  # kpc, d=10 kpc → x=0.1..2
x_check = r_check / 10.0
s_check = 2 * x_check**2 / (1 + x_check**2)
h_true  = 1.0 / (1 + x_check**2)

h_rec = np.ones(len(r_check))
log_r = np.log(r_check)
for i in range(1, len(r_check)):
    dln = log_r[i] - log_r[i-1]
    h_rec[i] = h_rec[i-1] * np.exp(-0.5 * (s_check[i-1] + s_check[i]) * dln)
h_rec_normalised = h_rec / h_rec[0] * h_true[0]  # rescale to match h_true[0]

print("h-recovery sanity check (d=10 kpc, r=[1,2,5,10,20] kpc):")
print(f"  h_true  (pISO): {np.round(h_true, 4)}")
print(f"  h_rec (trapz):  {np.round(h_rec_normalised, 4)}")
print(f"  max rel error:  {np.max(np.abs(h_rec_normalised - h_true) / h_true):.2e}")
print()

# ── 3. Run SR ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/toydatasets/pISO_noiseless.csv")
print(f"Dataset: {df['galaxy'].nunique()} galaxies loaded.")
print("Starting density2param verification (5 galaxies, ~200 iterations) …")
print("=" * 60)

os.makedirs("outputs/verify/density2param_toydata", exist_ok=True)

fit_density_2param(
    df,
    output_directory="outputs/verify/density2param_toydata",
    error_weighting=True,
    iterations=200,
    n_galaxies=5,
    n_d_grid=15,
    n_gamma_grid=4,
    gamma_range=(0.0, 2.0),
    populations=8,
    population_size=30,
    ncycles_per_iteration=50,
    weight_optimize=0.1,
    optimizer_iterations=3,
    upsilon_weight=0.0,        # disabled for noiseless: simplifies interpretation
    nu_t=3.0,
    n_irls=3,
    min_points=3,
    unary_operators=[],        # algebraic only: pISO s = 2x²/(1+x²) needs no unary
    weight_spos=0.1,
    weight_inner_slope=0.1,
    n_gamma_refine=5,
)

print()
print("=" * 60)
print("Verification complete.  Check HOF above for 2*r*r/(1+r*r) or equivalent.")
print("Expected loss near zero for noiseless data if integration is correct.")

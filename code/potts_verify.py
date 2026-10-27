"""
Verification of the mathematical claims in
"Complex by Nature, Momentary by Design" (Sections 6.6, 6.7, 6.8, 8.1, Appendix A).

All computations below are EXACT (full enumeration of the finite state space and
exact transition matrices), except where a Monte Carlo run is explicitly labelled.
Deterministic: no random numbers are used unless stated, and the seed is fixed.

Run:  python3 potts_verify.py      (Python >= 3.9, numpy >= 1.24, scipy >= 1.10)
"""
import itertools, json, sys
import numpy as np
from scipy.optimize import brentq

np.set_printoptions(suppress=True, precision=6)
OUT = {}

# ---------------------------------------------------------------------------
# Generic Potts utilities (Equation 2 with each unordered pair counted once)
# ---------------------------------------------------------------------------
def configs(N, q):
    return list(itertools.product(range(q), repeat=N))

def energy(x, J, a):
    """H(x) = -sum_{i<j} J_ij I(x_i=x_j) - sum_i a_{i,x_i}. J symmetric, zero diagonal."""
    N = len(x); H = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if x[i] == x[j]:
                H -= J[i, j]
        H -= a[i][x[i]]
    return H

def boltzmann(J, a, beta, q):
    N = J.shape[0]; X = configs(N, q)
    w = np.array([np.exp(-beta * energy(x, J, a)) for x in X])
    return X, w / w.sum(), w.sum()

def local_logits(x, i, J, a, beta, q, field=None):
    f = a[i] if field is None else field[i]
    return np.array([beta * (sum(J[i, j] for j in range(len(x)) if j != i and x[j] == c) + f[c])
                     for c in range(q)])

def site_kernel(X, i, J, a, beta, q, rho=0.0):
    """Exact single-site Gibbs kernel K_i (with optional persistence rho, Eq. 6)."""
    idx = {x: k for k, x in enumerate(X)}
    K = np.zeros((len(X), len(X)))
    for k, x in enumerate(X):
        lg = local_logits(x, i, J, a, beta, q)
        lg = lg + rho * (np.arange(q) == x[i])
        p = np.exp(lg - lg.max()); p /= p.sum()
        for c in range(q):
            y = list(x); y[i] = c
            K[k, idx[tuple(y)]] += p[c]
    return K

def random_scan(X, J, a, beta, q, rho=0.0):
    N = J.shape[0]
    return sum(site_kernel(X, i, J, a, beta, q, rho) for i in range(N)) / N

def sweep(X, J, a, beta, q, order):
    K = np.eye(len(X))
    for i in order:
        K = K @ site_kernel(X, i, J, a, beta, q)
    return K

def random_perm_sweep(X, J, a, beta, q):
    N = J.shape[0]; perms = list(itertools.permutations(range(N)))
    return sum(sweep(X, J, a, beta, q, p) for p in perms) / len(perms)

def parallel(X, J, a, beta, q):
    """Synchronous kernel: K(x,y) = prod_i P(y_i | x_{-i})."""
    K = np.zeros((len(X), len(X)))
    cond = []
    for x in X:
        ps = []
        for i in range(len(x)):
            lg = local_logits(x, i, J, a, beta, q); p = np.exp(lg - lg.max()); ps.append(p / p.sum())
        cond.append(ps)
    for k, x in enumerate(X):
        for m, y in enumerate(X):
            K[k, m] = np.prod([cond[k][i][y[i]] for i in range(len(x))])
    return K

def stationary(K):
    w, v = np.linalg.eig(K.T)
    s = np.real(v[:, np.argmin(np.abs(w - 1))]); return s / s.sum()

def tv(p, r):
    return 0.5 * np.abs(np.asarray(p) - np.asarray(r)).sum()

# ---------------------------------------------------------------------------
# 1. Table 11 network: exact enumeration, stationarity, detailed balance
# ---------------------------------------------------------------------------
q = 3
J = np.array([[0, 0.8], [0.8, 0]])
a = [np.array([0.25, -0.10, 0.00]), np.array([-0.15, 0.00, 0.30])]
X, pi, Z = boltzmann(J, a, 1.0, q)
OUT["table11"] = {"Z": round(Z, 4), "pi": {str(x): round(p, 4) for x, p in zip(X, pi)}}
K = random_scan(X, J, a, 1.0, q)
F = pi[:, None] * K
OUT["table11"]["stationarity_max_abs_dev"] = float(np.abs(pi @ K - pi).max())
OUT["table11"]["detailed_balance_max_asym"] = float(np.abs(F - F.T).max())
# persistence on the same network (Eq. 6), rho = 1.5 : exact stationary law
for rho in (0.5, 1.0, 1.5):
    Kr = random_scan(X, J, a, 1.0, q, rho=rho)
    sr = stationary(Kr); Fr = sr[:, None] * Kr
    OUT["table11"][f"persistence_rho{rho}"] = {
        "TV_vs_Boltzmann": round(tv(sr, pi), 5),
        "stationary": {str(x): round(p, 4) for x, p in zip(X, sr)},
        "detailed_balance_max_asym_wrt_own_stationary": float(np.abs(Fr - Fr.T).max())}

# ---------------------------------------------------------------------------
# 2. Schedule comparison: three-element, two-category network, beta=2.2, J=0.9
#    (fields not reported in the manuscript; zero fields assumed and stated)
# ---------------------------------------------------------------------------
q2 = 2; N3 = 3
J3 = 0.9 * (np.ones((N3, N3)) - np.eye(N3)); a3 = [np.zeros(q2)] * N3
X3, pi3, _ = boltzmann(J3, a3, 2.2, q2)
sched = {
    "random_scan": random_scan(X3, J3, a3, 2.2, q2),
    "sequential_sweep": sweep(X3, J3, a3, 2.2, q2, range(N3)),
    "random_permutation_sweep": random_perm_sweep(X3, J3, a3, 2.2, q2),
    "parallel": parallel(X3, J3, a3, 2.2, q2)}
OUT["schedules_zero_field"] = {}
for k, Ks in sched.items():
    s = stationary(Ks); Fs = pi3[:, None] * Ks
    OUT["schedules_zero_field"][k] = {
        "exact_TV_stationary_vs_Boltzmann": round(tv(s, pi3), 6),
        "pi_is_stationary(max|piK-pi|)": float(np.abs(pi3 @ Ks - pi3).max()),
        "detailed_balance_wrt_pi(max asym)": float(np.abs(Fs - Fs.T).max())}
# Parallel kernel on 3 nodes: also report which configurations differ
sp = stationary(sched["parallel"])
OUT["schedules_zero_field"]["parallel_stationary"] = {str(x): round(p, 4) for x, p in zip(X3, sp)}
OUT["schedules_zero_field"]["boltzmann"] = {str(x): round(p, 4) for x, p in zip(X3, pi3)}

# ---------------------------------------------------------------------------
# 3. Mean-field equation (Eq. 8), spinodal, saddle-node, beta_c (Eq. 9)
# ---------------------------------------------------------------------------
def f_mf(z, beta, q):
    return np.exp(beta * z) / (np.exp(beta * z) + (q - 1) * np.exp(beta * (1 - z) / (q - 1)))

def fixed_points(beta, q, grid=20001):
    zs = np.linspace(1e-6, 1 - 1e-6, grid); g = f_mf(zs, beta, q) - zs
    roots = []
    for k in range(grid - 1):
        if g[k] == 0 or g[k] * g[k + 1] < 0:
            roots.append(brentq(lambda z: f_mf(z, beta, q) - z, zs[k], zs[k + 1]))
    out = []
    for r in roots:
        d = beta * q / (q - 1) * r * (1 - r)   # f'(z) at a fixed point
        out.append((round(r, 4), "stable" if d < 1 else "unstable"))
    return out

def free_energy(z, beta, q):
    """Mean-field free energy per site (up to constants) for the symmetric ansatz."""
    zs = np.array([z] + [(1 - z) / (q - 1)] * (q - 1))
    return -0.5 * beta * np.sum(zs ** 2) + np.sum(zs * np.log(zs))

# saddle-node: smallest beta at which a non-trivial fixed point with z > 1/q exists
def n_roots_above(beta, q):
    return sum(1 for r, _ in fixed_points(beta, q) if r > 1 / q + 1e-3)
lo, hi = 2.5, 2.77
for _ in range(60):
    mid = (lo + hi) / 2
    (hi, lo) = (mid, lo) if n_roots_above(mid, q) > 0 else (hi, mid)
beta_sn = hi
z_sn = [r for r, _ in fixed_points(beta_sn + 1e-6, q) if r > 1 / q + 1e-3]
beta_c3 = 2 * (q - 1) * np.log(q - 1) / (q - 2)
zc = max(r for r, s in fixed_points(beta_c3, q))
OUT["mean_field_q3"] = {
    "beta_c_formula": round(beta_c3, 5),
    "beta_saddle_node": round(beta_sn, 4),
    "z_at_saddle_node": z_sn,
    "spinodal_beta_sp=q": q,
    "f_prime_at_1/q_equals_beta/q_check(beta=2.5)": round(2.5 * q / (q - 1) * (1 / q) * (1 - 1 / q), 6),
    "fixed_points_beta2.9": fixed_points(2.9, q),
    "fixed_points_0.8beta_c": fixed_points(0.8 * beta_c3, q),
    "free_energy_equal_at_beta_c(ordered - disordered)": round(free_energy(zc, beta_c3, q) - free_energy(1 / 3, beta_c3, q), 8),
    "z_ordered_at_beta_c": round(zc, 4),
    "fixed_points_q2_beta2.2": fixed_points(2.2, 2)}

# ---------------------------------------------------------------------------
# 4. Identification (A.4): rescaling invariance of all kernels
# ---------------------------------------------------------------------------
lam = 2.7
Ka = random_scan(X, J, a, 1.0, q); Kb = random_scan(X, J / lam, [ai / lam for ai in a], lam, q)
OUT["identification_rescaling_max_kernel_diff"] = float(np.abs(Ka - Kb).max())
Kp1 = parallel(X, J, a, 1.0, q); Kp2 = parallel(X, J / lam, [ai / lam for ai in a], lam, q)
OUT["identification_rescaling_parallel_kernel_diff"] = float(np.abs(Kp1 - Kp2).max())

json.dump(OUT, open(sys.argv[1] if len(sys.argv) > 1 else "verify_output.json", "w"), indent=2, default=str)
print(json.dumps(OUT, indent=2, default=str))

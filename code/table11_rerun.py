"""Seeded rerun of the Table 11 Monte Carlo check: random-order asynchronous Gibbs updating
(Eq. 5) on the two-process, three-category network; 540,000 recorded updates after 10,000
burn-in updates; numpy PCG64, seed 20261026. Usage: python3 table11_rerun.py out.json"""
import json, sys, itertools
import numpy as np
J12 = 0.8; a = np.array([[0.25, -0.10, 0.00], [-0.15, 0.00, 0.30]]); beta = 1.0
rng = np.random.default_rng(20261026)
x = rng.integers(0, 3, 2); counts = np.zeros((3, 3))
burn, n = 10_000, 540_000
for t in range(burn + n):
    i = rng.integers(2); j = 1 - i
    logit = beta * (J12 * (np.arange(3) == x[j]) + a[i]); p = np.exp(logit - logit.max()); p /= p.sum()
    x[i] = rng.choice(3, p=p)
    if t >= burn:
        counts[x[0], x[1]] += 1
freq = counts / n
w = np.array([[np.exp(beta * (J12 * (c == e) + a[0, c] + a[1, e])) for e in range(3)] for c in range(3)])
pi = w / w.sum()
out = {"Z": float(w.sum()), "freq": {f"({c}, {e})": round(float(freq[c, e]), 4) for c in range(3) for e in range(3)},
       "exact": {f"({c}, {e})": round(float(pi[c, e]), 4) for c in range(3) for e in range(3)},
       "TV": round(float(0.5 * np.abs(freq - pi).sum()), 4), "seed": 20261026, "burn_in": burn, "recorded": n}
json.dump(out, open(sys.argv[1], "w"), indent=1); print(out)

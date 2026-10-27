"""
Revised Figure 8 / Section 8.5 analysis (NEW analysis prepared during revision; it does
NOT reproduce the original, unavailable Figure 8 script).

Exact equilibrium (Boltzmann) properties of the six-process, three-mode Potts network of
Section 7.1 / Figure 4, computed by full enumeration of all 3^6 = 729 configurations.
Because only kappa = beta*J is identified (Appendix A.4), couplings are given directly as
kappa (beta = 1). Contextual fields are zero, so no mode is favoured a priori.

Graph (Figure 4): PTSD cluster {ATT, ARO, ACT}; DSO cluster {AFF, SEL, REL}; all three
within-cluster pairs coupled with kappa_w; four bridges ATT-AFF, ARO-REL, SEL-ACT, ATT-SEL
coupled with kappa_b.

Definitions (denominators explicit):
  C_PTSD  = P(ATT = ARO = ACT)                       [denominator: all configurations]
  C_DSO   = P(AFF = SEL = REL)                       [denominator: all configurations]
  P_both  = P(both clusters internally coherent)     [denominator: all configurations]
  A_cond  = P(PTSD mode = DSO mode | both coherent)  [denominator: configurations with both coherent]
  D       = P(both coherent AND modes differ) = P_both * (1 - A_cond)  ("dissociated configurations")
Chance (kappa = 0) reference values: C = 1/9, P_both = 1/81, A_cond = 1/3.

Analytic result (zero fields): conditional on both clusters being coherent, the within-cluster
energy is constant, so A_cond = e^B / (e^B + q - 1), where B is the summed bridge coupling; A_cond
does not depend on kappa_w. kappa_w governs how often both clusters are coherent (P_both) and hence
the prevalence of dissociated configurations D.

A seeded Monte Carlo cross-check (random-scan Gibbs, Eq. 5) is also run to show agreement
with the exact values and to quantify the sampling variability of A_cond when P_both is small.

Run:  python3 fig8_exact.py <outdir>
"""
import itertools, json, os, sys
import numpy as np

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUTDIR, exist_ok=True)

NODES = ["ATT", "ARO", "ACT", "AFF", "SEL", "REL"]
WITHIN = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
BRIDGES = [(0, 3), (1, 5), (4, 2), (0, 4)]
Q = 3
X = np.array(list(itertools.product(range(Q), repeat=6)))
cP = (X[:, 0] == X[:, 1]) & (X[:, 1] == X[:, 2])
cD = (X[:, 3] == X[:, 4]) & (X[:, 4] == X[:, 5])
both = cP & cD
agree = both & (X[:, 0] == X[:, 3])


def coupling_matrix(kw, kb):
    K = np.zeros((6, 6))
    for i, j in WITHIN:
        K[i, j] = K[j, i] = kw
    for i, j in BRIDGES:
        K[i, j] = K[j, i] = kb
    return K


def exact(kw, kb):
    K = coupling_matrix(kw, kb)
    E = np.zeros(len(X))
    for i in range(6):
        for j in range(i + 1, 6):
            if K[i, j]:
                E -= K[i, j] * (X[:, i] == X[:, j])
    p = np.exp(-E); p /= p.sum()
    Pb = p[both].sum(); Ac = p[agree].sum() / Pb
    return {"C_PTSD": p[cP].sum(), "C_DSO": p[cD].sum(), "P_both": Pb,
            "A_cond": Ac, "D_dissociated": Pb * (1 - Ac)}


def gibbs(kw, kb, n_sweeps, burn, seed):
    rng = np.random.default_rng(seed)          # numpy PCG64
    K = coupling_matrix(kw, kb)
    x = rng.integers(0, Q, 6)
    rec = []
    for s in range(burn + n_sweeps):
        for _ in range(6):                     # 6 random-scan updates = 1 sweep-equivalent
            i = rng.integers(6)
            logits = np.array([sum(K[i, j] for j in range(6) if j != i and x[j] == c) for c in range(Q)])
            pr = np.exp(logits - logits.max()); pr /= pr.sum()
            x[i] = rng.choice(Q, p=pr)
        if s >= burn:
            rec.append(x.copy())
    R = np.array(rec)
    rp = (R[:, 0] == R[:, 1]) & (R[:, 1] == R[:, 2]); rd = (R[:, 3] == R[:, 4]) & (R[:, 4] == R[:, 5])
    rb = rp & rd; ra = rb & (R[:, 0] == R[:, 3])
    return {"C_PTSD": rp.mean(), "C_DSO": rd.mean(), "P_both": rb.mean(),
            "A_cond": ra.sum() / rb.sum() if rb.sum() else float("nan"), "n_both": int(rb.sum())}


CONDITIONS = {
    "Weak coupling overall": (0.5, 0.25),
    "Weak bridges": (2.0, 0.25),
    "Strong bridges": (2.0, 1.5),
}
CHANCE = {"C": 1 / 9, "P_both": 1 / 81, "A_cond": 1 / 3}

results = {"definitions": __doc__, "chance": CHANCE, "conditions": {}, "ratio_sweep": [], "monte_carlo": {}}
for name, (kw, kb) in CONDITIONS.items():
    results["conditions"][name] = {"kappa_within": kw, "kappa_bridge": kb, **{k: round(float(v), 4) for k, v in exact(kw, kb).items()}}

# Bridge sweep for two within-cluster couplings; analytic check of A_cond = e^B/(e^B+q-1)
for kw in (0.5, 2.0):
    for kb in np.round(np.linspace(0, 1.0, 51), 4):
        e = exact(kw, kb)
        B = 4 * kb
        analytic = np.exp(B) / (np.exp(B) + Q - 1)
        results["ratio_sweep"].append({"kappa_within": kw, "kappa_bridge": float(kb), "B_total_bridge": float(B),
                                       "A_cond_analytic": round(float(analytic), 6),
                                       **{k: round(float(v), 6) for k, v in e.items()}})
results["max_abs_diff_A_cond_exact_vs_analytic"] = float(max(abs(d["A_cond"] - d["A_cond_analytic"]) for d in results["ratio_sweep"]))

# Monte Carlo cross-check: 10 independent chains per condition, fixed seeds
SEEDS = list(range(20261026, 20261036))
for name, (kw, kb) in CONDITIONS.items():
    runs = [gibbs(kw, kb, n_sweeps=4000, burn=500, seed=s) for s in SEEDS]
    agg = {}
    for k in ["C_PTSD", "C_DSO", "P_both", "A_cond"]:
        v = np.array([r[k] for r in runs], float)
        agg[k] = {"mean": round(float(np.nanmean(v)), 4), "sd_across_chains": round(float(np.nanstd(v, ddof=1)), 4)}
    agg["n_both_per_chain_mean"] = float(np.mean([r["n_both"] for r in runs]))
    agg["settings"] = {"chains": len(SEEDS), "seeds": SEEDS, "burn_in_sweeps": 500, "recorded_sweeps": 4000,
                       "update": "random-scan single-site Gibbs, 6 updates per recorded sweep", "rng": "numpy PCG64"}
    results["monte_carlo"][name] = agg

json.dump(results, open(os.path.join(OUTDIR, "fig8_exact_results.json"), "w"), indent=2)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8})
C_PT, C_DS, C_AG = "#C8582A", "#3A68C6", "#3F8F2A"
INK, MUTED = "#222222", "#6B6B6B"

fig = plt.figure(figsize=(7.2, 4.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.30)

# Panel A: three conditions
ax = fig.add_subplot(gs[0])
names = list(CONDITIONS)
short = {"Weak coupling overall": "Weak overall", "Weak bridges": "Weak bridges", "Strong bridges": "Strong bridges"}
xs = np.arange(len(names)); w = 0.26
vals = [[results["conditions"][n][k] for n in names] for k in ["C_PTSD", "C_DSO", "A_cond"]]
labels = ["PTSD-cluster coherence", "DSO-cluster coherence", "Conditional cross-cluster agreement"]
for k, (v, c, lab) in enumerate(zip(vals, [C_PT, C_DS, C_AG], labels)):
    bars = ax.bar(xs + (k - 1) * w, v, w - 0.04, color=c, label=lab, edgecolor="white", linewidth=0.6,
                  hatch="////" if k == 2 else None)
    for bx, val in zip(bars, v):
        ax.text(bx.get_x() + bx.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", va="bottom",
                fontsize=6.3, color=INK)
ax.axhline(1 / 3, color=MUTED, lw=0.8, ls=":")
ax.axhline(1 / 9, color=MUTED, lw=0.8, ls="--")
ax.set_xticks(xs)
ax.set_xticklabels([f"{short[n]}\n\u03BAw = {CONDITIONS[n][0]:g}\n\u03BAb = {CONDITIONS[n][1]:g}" for n in names], fontsize=7.2)
ax.set_ylim(0, 1.1); ax.set_ylabel("Probability (exact)")
ax.set_title("A. Three coupling conditions", loc="left", fontsize=9, color=INK)
from matplotlib.lines import Line2D
h, l = ax.get_legend_handles_labels()
h += [Line2D([], [], color=MUTED, ls=":", lw=0.8), Line2D([], [], color=MUTED, ls="--", lw=0.8)]
l += ["Chance agreement (1/3)", "Chance coherence (1/9)"]
ax.legend(h, l, frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2)

# Panel B: bridge sweep
ax2 = fig.add_subplot(gs[1])
sw = results["ratio_sweep"]
d2 = [r for r in sw if r["kappa_within"] == 2.0]; d05 = [r for r in sw if r["kappa_within"] == 0.5]
ax2.plot([r["kappa_bridge"] for r in d2], [r["A_cond"] for r in d2], color=C_AG, lw=2,
         label="Conditional agreement, \u03BAw = 2")
ax2.plot([r["kappa_bridge"] for r in d05][::5], [r["A_cond"] for r in d05][::5], ls="none", marker="o", ms=4,
         mfc="white", mec=C_AG, mew=1.2, label="Conditional agreement, \u03BAw = 0.5 (coincides)")
ax2.plot([r["kappa_bridge"] for r in d2], [r["D_dissociated"] for r in d2], color=INK, lw=1.4, ls="--",
         label="Dissociated configurations, \u03BAw = 2")
ax2.plot([r["kappa_bridge"] for r in d05], [r["D_dissociated"] for r in d05], color=INK, lw=1.4, ls="-.",
         label="Dissociated configurations, \u03BAw = 0.5")
ax2.axhline(1 / 3, color=MUTED, lw=0.8, ls=":")
ax2.set_xlabel("Coupling per bridge \u03BAb  (summed bridge coupling B = 4\u03BAb)", fontsize=8)
ax2.set_ylabel("Probability (exact)")
ax2.set_ylim(0, 1.05); ax2.set_xlim(0, 1.0)
ax2.set_title("B. Bridge vs. within-cluster coupling", loc="left", fontsize=9, color=INK)
ax2.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1)

fig.savefig(os.path.join(OUTDIR, "Figure8_revised.png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "Figure8_revised.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "Figure8_revised.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
print(json.dumps({k: results[k] for k in ["conditions", "max_abs_diff_A_cond_exact_vs_analytic"]}, indent=2))

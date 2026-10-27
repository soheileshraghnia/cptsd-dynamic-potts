"""Regenerates Figure 6 exactly from Equation 8 (mean-field, fully connected, zero-field, q = 3).
Same content as the original figure; 600 dpi PNG/TIFF + vector PDF. Usage: python3 fig6_regen.py outdir"""
import sys, os, numpy as np
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = sys.argv[1]; q = 3
f = lambda z, b: np.exp(b*z) / (np.exp(b*z) + (q-1)*np.exp(b*(1-z)/(q-1)))
def stable_ordered(b):
    zs = np.linspace(1/q + 1e-4, 0.999, 4000); g = f(zs, b) - zs
    r = [brentq(lambda z: f(z, b) - z, zs[k], zs[k+1]) for k in range(len(zs)-1) if g[k]*g[k+1] < 0]
    r = [x for x in r if b*q/(q-1)*x*(1-x) < 1]   # keep stable fixed points
    return max(r) if r else np.nan
bsn = 2.7456; bc = 4*np.log(2)
bd = np.linspace(1.5, 3.0, 400); bo = np.linspace(bsn + 1e-4, 3.3, 400)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.plot(bd, np.full_like(bd, 1/3), color="#3A68C6", lw=2, label="Disordered branch (stable up to spinodal)")
ax.plot(bo, [stable_ordered(b) for b in bo], color="#C8582A", lw=2, label="Stable ordered branch")
ax.axvline(bc, color="#444444", ls=":", lw=1, label=f"Mean-field βc = 4 ln 2 ≈ {bc:.3f}")
ax.axvline(3.0, color="#888888", ls="--", lw=1, label="Disordered-branch spinodal βsp = 3")
ax.axvline(bsn, color="#C8582A", ls=(0, (1, 2)), lw=0.8, label=f"Saddle-node β ≈ {bsn:.3f}")
ax.set_xlabel("Alignment β"); ax.set_ylabel("Reference-category occupancy z")
ax.set_xlim(1.45, 3.35); ax.set_ylim(0.3, 0.92)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
for ext in ("png", "tiff"):
    fig.savefig(os.path.join(out, f"Figure6_regenerated.{ext}"), dpi=600, bbox_inches="tight",
                **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}))
fig.savefig(os.path.join(out, "Figure6_regenerated.pdf"), bbox_inches="tight")

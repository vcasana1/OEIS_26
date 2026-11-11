#!/usr/bin/env python3
# Castor and Pollux I: vertical cut Re s = 1/2. G and A (from their definitions, no zeta) on s = 1/2 + it, 0.5 <= t <= 100;
# the values of t where G(1/2+it) = A(1/2+it), located from G and A only, and compared afterwards with the zeros of zeta.
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, mpmath as mp
from scipy.optimize import minimize_scalar
from cp_eval import G_grid, A
t0 = time.time(); out = open('vertical_cut_output.txt', 'w')
def log(s): print(s, flush=True); out.write(s + '\n')
T = np.round(np.arange(0.5, 100.0001, 0.01), 2)
g = G_grid([0.5], T)[:, 0]; a = A(0.5 + 1j * T); d = np.abs(g - a)
cand = [i for i in range(1, len(T) - 1) if d[i] <= d[i - 1] and d[i] <= d[i + 1]]
log('local minima of |G-A| on the grid (step 0.01): %d' % len(cand))
f = lambda t: abs(G_grid([0.5], [t])[0, 0] - A(np.array([0.5 + 1j * t]))[0])
hits = []
for i in cand:
    r = minimize_scalar(lambda t: f(t) ** 2, bounds=(T[i] - 0.01, T[i] + 0.01), method='bounded', options={'xatol': 1e-12})
    hits.append((r.x, f(r.x), d[i]))
zz = [float(mp.im(mp.zetazero(k))) for k in range(1, 30)]
zz = [z for z in zz if z <= 100]
log('zeros of zeta with 0 < gamma <= 100: %d' % len(zz))
log('  t* (G = A)          min |G-A|    nearest gamma_k      |t* - gamma_k|   G(1/2+it*)')
rows = []
for x, v, _ in hits:
    if v > 1e-6:
        log('  local minimum at t=%.4f with |G-A|=%.3e (not a coincidence)' % (x, v)); continue
    k = int(np.argmin([abs(x - z) for z in zz])); gv = G_grid([0.5], [x])[0, 0]
    rows.append([k + 1, x, zz[k], v, gv.real, gv.imag])
    log('  %.10f  %.2e   %.10f   %.1e    %.6f%+.6fi' % (x, v, zz[k], abs(x - zz[k]), gv.real, gv.imag))
log('coincidences found: %d; every zero of zeta up to 100 matched: %s' % (len(rows), sorted(r[0] for r in rows) == list(range(1, len(zz) + 1))))
r4 = lambda v: [float('%.5g' % x) for x in v]
json.dump(dict(t=[float(x) for x in T[::2]], Gr=r4(g.real[::2]), Gi=r4(g.imag[::2]), Ar=r4(a.real[::2]), Ai=r4(a.imag[::2]),
               hits=rows), open('vertical_cut.json', 'w'), separators=(',', ':'))
log('time %.1f s' % (time.time() - t0))

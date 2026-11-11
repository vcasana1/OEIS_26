#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
castor_pollux_I_rouche_certificate_v1.py

Interval (ball) arithmetic certificate for the Rouche test of
"Castor and Pollux I" (file "Castor Pollux I v1.tex",
SHA-256 3bc78229172e068d3f4b5403e7ff808c565b7f6baa8694b8da2f4774f5f7556a),
Proposition prop:rouche, inequality (eq:rouche), Table tab:rouche.

What is certified
-----------------
For each of the six centres s0 of Table tab:rouche, with
    M = 4000 (harmonics), N = 200 (periods), L = 65536 (nodes), radius = 1/100,
the script encloses, in rigorous ball arithmetic (Arb, through python-flint),

    eta0 = H_{M,N,L}(s0)
         = (1/L) sum_{n=1}^{N} sum_{i=0}^{L-1} K_M(x_i) (n+x_i)^(-s0-3),
    eta1 = H'_{M,N,L}(s0)
         = -(1/L) sum_{n=1}^{N} sum_{i=0}^{L-1} K_M(x_i) log(n+x_i) (n+x_i)^(-s0-3),
    K_M(x) = sum_{m=1}^{M} (1-cos(2 pi m x))/m^3,      x_i = (i+1/2)/L,

the bounds E0, E1 of Theorem thm:trunc at s = s0, and both sides of

    |eta0| + E0 + r E1 + r^2 B0/(Re s0 - r + 2)^3  <  r |eta1|.          (eq:rouche)

Every number is a ball [midpoint +/- radius] that provably contains the exact
real number; all rounding errors are inside the radius.  The inequality is
declared CERTIFIED only when
    upper bound of the left side  <  lower bound of the right side.
No floating-point value enters the proof.  The analytic statement
(Proposition prop:rouche) is used as it stands in the manuscript and is not
re-proved here.

Exact inputs
------------
The centres are the decimal numbers printed in Table tab:rouche, taken as
exact rationals.  x_i, n+x_i, 1/m^3 are exact rationals.  Only pi, cos, log
and exp are transcendental; Arb returns rigorous enclosures for them.

Exact identities used to organise the finite sums (no approximation)
--------------------------------------------------------------------
(a) cos(2 pi m x_i) = cos(pi j/L) with j = m(2i+1) mod 2L, so a table of the
    2L numbers cos(pi j/L) suffices.
(b) x_{L-1-i} = 1 - x_i and K_M(1-x) = K_M(x), so K_M is computed at the nodes
    0 <= i < L/2 only.
(c) (n+x_i)^(-s-3) = exp(-(s+3) log(n+x_i)), n+x_i = (2Ln+2i+1)/(2L).

Optional cross-check (not part of the proof)
--------------------------------------------
With numpy, the same two sums are evaluated in IEEE double precision and the
distance to the certified enclosure is printed.  This measures the rounding
error of a double precision evaluation.  Disable with --no-double.

Usage
-----
    python3 castor_pollux_I_rouche_certificate_v1.py [--prec 128] [--procs 2]
            [--out PREFIX] [--no-double]
Outputs PREFIX_output.txt (log) and PREFIX_intervals.json.

Requires: python-flint (>= 0.6; tested with python-flint 0.9.0 / FLINT 3.6.0).
"""
import argparse
import json
import os
import platform
import sys
import time

import flint
from flint import acb, arb, ctx, fmpq

# ----------------------------------------------------------------------------
# Exact data of the manuscript
# ----------------------------------------------------------------------------
M_HARM = 4000          # \mathsf M
N_PER = 200            # \mathsf N
L_NODES = 65536        # \mathsf L
RADIUS = fmpq(1, 100)  # \varsigma

# (Re s0, Im s0) as exact decimals, and the values printed in Table tab:rouche
# (|eta0|, |eta1|, E0, left side, right side) for comparison only.
CENTRES = [
    ("1.127312874", "15.001263432", (4.2e-9, 0.018463, 8.6e-8, 8.3e-6, 1.8e-4)),
    ("0.533572290", "21.763254707", (7.9e-8, 0.011815, 1.5e-6, 1.7e-5, 1.2e-4)),
    ("0.894730810", "25.624836921", (9.2e-9, 0.006881, 2.4e-7, 1.1e-5, 6.9e-5)),
    ("0.227688021", "30.975962723", (2.9e-7, 0.006866, 8.4e-6, 3.2e-5, 6.9e-5)),
    ("0.950506780", "33.599087545", (5.2e-9, 0.003138, 2.0e-7, 1.0e-5, 3.1e-5)),
    ("0.430899874", "38.054886210", (8.0e-8, 0.004536, 2.7e-6, 2.1e-5, 4.5e-5)),
]


def dec_to_fmpq(txt):
    """Exact rational value of a decimal string."""
    sign = -1 if txt.startswith("-") else 1
    txt = txt.lstrip("+-")
    whole, _, frac = txt.partition(".")
    return sign * fmpq(int(whole + frac), 10 ** len(frac))


def ball(x, digits=25):
    """Decimal ball that contains the binary ball x (Arb prints rigorously)."""
    return x.str(digits)


# ----------------------------------------------------------------------------
# Step 1: the truncated kernel at the midpoint nodes
# ----------------------------------------------------------------------------
def kernel_at_nodes(log):
    """Enclosures of K_M(x_i), 0 <= i < L/2 (the other half by symmetry)."""
    L = L_NODES
    two_l = 2 * L
    t0 = time.time()
    # d[j] = 1 - cos(pi j / L), j = 0..2L-1
    d = [1 - arb(fmpq(j, L)).cos_pi() for j in range(two_l)]
    w = [arb(fmpq(1, m ** 3)) for m in range(1, M_HARM + 1)]
    log("  table of 1-cos(pi j/L), j<2L: %d balls, %.1f s" % (two_l, time.time() - t0))
    t0 = time.time()
    kern = []
    for i in range(L // 2):
        q = 2 * i + 1
        j = 0
        acc = arb(0)
        for m in range(M_HARM):
            j += q
            if j >= two_l:
                j -= two_l
            acc += d[j] * w[m]
        kern.append(acc)
    log("  K_M at %d nodes (x %d harmonics): %.1f s" % (L // 2, M_HARM, time.time() - t0))
    return kern


# ----------------------------------------------------------------------------
# Step 2: the two finite sums and the test, for a subset of the centres
# ----------------------------------------------------------------------------
def error_bounds(s, sigma):
    """E0 and E1 of Theorem thm:trunc at s (sigma = Re s), as balls."""
    M, N, L = arb(M_HARM), arb(N_PER), arb(L_NODES)
    pi = arb.pi()
    b0 = arb(fmpq(5, 2))
    b1 = pi ** 3 / 3
    b2 = 4 * pi ** 2 * (1 + M.log())

    def z(theta):
        return 1 + 1 / (theta - 1)

    a3 = abs(s + 3)
    a34 = abs((s + 3) * (s + 4))
    a27 = abs(2 * s + 7)
    np1 = N + 1
    np1pow = (np1.log() * (sigma + 2)).exp()          # (N+1)^(sigma+2)
    quad = 24 * L ** 2
    e0 = (1 / ((sigma + 2) * M ** 2)
          + b0 / ((sigma + 2) * np1pow)
          + (b2 * z(sigma + 3) + 2 * a3 * b1 * z(sigma + 4) + a34 * b0 * z(sigma + 5)) / quad)
    e1 = (1 / ((sigma + 2) ** 2 * M ** 2)
          + b0 / np1pow * (np1.log() / (sigma + 2) + 1 / (sigma + 2) ** 2)
          + (b2 * z(sigma + 2) + 2 * b1 * (z(sigma + 4) + a3 * z(sigma + 3))) / quad
          + b0 * (a34 * z(sigma + 4) + a27 * z(sigma + 5)) / quad)
    return e0, e1, b0


_KERN = None   # kernel enclosures; set in main() before the workers are forked
               # (Arb balls are inherited by fork, they are never pickled)


def certify_subset(args):
    """Worker: full certificate for the centres whose indices are in idx."""
    idx, prec = args
    kern = _KERN
    ctx.prec = prec
    L = L_NODES
    two_l = 2 * L
    last = L - 1
    log_two_l = arb(two_l).log()
    sig = [arb(dec_to_fmpq(CENTRES[c][0])) for c in idx]
    tee = [arb(dec_to_fmpq(CENTRES[c][1])) for c in idx]
    s0 = [acb(sig[k], tee[k]) for k in range(len(idx))]
    expo = [-(s + 3) for s in s0]                      # exponent -(s0+3)
    a0 = [acb(0) for _ in idx]
    a1 = [acb(0) for _ in idx]
    rng = range(len(idx))
    for n in range(1, N_PER + 1):
        base = two_l * n + 1
        for i in range(L):
            lg = arb(base + 2 * i).log() - log_two_l   # log(n + x_i)
            k = kern[i] if i < L // 2 else kern[last - i]
            klg = k * lg
            for c in rng:
                p = (expo[c] * lg).exp()               # (n+x_i)^(-s0-3)
                a0[c] += p * k
                a1[c] += p * klg
    out = []
    r = arb(RADIUS)
    for k, c in enumerate(idx):
        eta0 = a0[k] / L
        eta1 = -a1[k] / L
        e0, e1, b0 = error_bounds(s0[k], sig[k])
        taylor = r ** 2 * b0 / (sig[k] - r + 2) ** 3
        abs0, abs1 = abs(eta0), abs(eta1)
        left = abs0 + e0 + r * e1 + taylor
        right = r * abs1
        margin = right - left
        hyp = bool(sig[k] - r > -1)                    # Re s0 - r > -1, certain
        cert = bool(left.upper() < right.lower()) and bool(margin > 0)
        out.append({
            "index": c + 1,
            "re_s0": CENTRES[c][0], "im_s0": CENTRES[c][1],
            "hypothesis_Re_s0_minus_r_gt_minus1": hyp,
            "eta0_re": ball(eta0.real), "eta0_im": ball(eta0.imag),
            "eta1_re": ball(eta1.real), "eta1_im": ball(eta1.imag),
            "abs_eta0": ball(abs0), "abs_eta1": ball(abs1),
            "E0": ball(e0), "E1": ball(e1), "taylor_term": ball(taylor),
            "left_side": ball(left), "right_side": ball(right),
            "left_upper": ball(left.upper()), "right_lower": ball(right.lower()),
            "margin_right_minus_left": ball(margin),
            "ratio_left_over_right": ball(left / right, 12),
            "max_radius_eta": ball(max(eta0.real.rad(), eta0.imag.rad(),
                                       eta1.real.rad(), eta1.imag.rad()), 5),
            "certified": bool(hyp and cert),
            # midpoints as floats, for the comparisons only (not for the proof)
            "_f": [float(eta0.real.mid()), float(eta0.imag.mid()),
                   float(eta1.real.mid()), float(eta1.imag.mid()),
                   float(abs0.mid()), float(abs1.mid()), float(e0.mid()),
                   float(left.mid()), float(right.mid())],
        })
    return out


# ----------------------------------------------------------------------------
# Optional: IEEE double precision evaluation of the same sums (cross-check)
# ----------------------------------------------------------------------------
def double_precision_sums():
    import numpy as np
    L, M, N = L_NODES, M_HARM, N_PER
    x = (np.arange(L) + 0.5) / L
    m = np.arange(1, M + 1, dtype=np.float64)
    wts = 1.0 / m ** 3
    kern = np.empty(L)
    step = 2048
    for a in range(0, L, step):
        xx = x[a:a + step]
        kern[a:a + step] = (1.0 - np.cos(2.0 * np.pi * np.outer(xx, m))) @ wts
    res = []
    for re, im, _ in CENTRES:
        s = complex(float(re), float(im))
        e0 = 0j
        e1 = 0j
        for n in range(1, N + 1):
            u = n + x
            lg = np.log(u)
            p = np.exp(-(s + 3.0) * lg)
            e0 += np.sum(kern * p)
            e1 += np.sum(kern * lg * p)
        res.append((e0 / L, -e1 / L))
    return res


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prec", type=int, default=128, help="working precision in bits")
    ap.add_argument("--procs", type=int, default=2, help="worker processes")
    ap.add_argument("--out", default="castor_pollux_I_rouche_certificate_v1")
    ap.add_argument("--no-double", action="store_true")
    a = ap.parse_args()
    ctx.prec = a.prec
    logf = open(a.out + "_output.txt", "w")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t_all = time.time()
    log("Castor and Pollux I -- ball arithmetic certificate of the Rouche test (eq:rouche)")
    log("manuscript: Castor Pollux I v1.tex, SHA-256 "
        "3bc78229172e068d3f4b5403e7ff808c565b7f6baa8694b8da2f4774f5f7556a")
    log("method: Arb ball arithmetic; python-flint %s, FLINT %s, Python %s, %s"
        % (flint.__version__, getattr(flint, "__FLINT_VERSION__", "?"),
           platform.python_version(), platform.platform()))
    log("precision: %d bits; M=%d, N=%d, L=%d, radius=1/100" % (a.prec, M_HARM, N_PER, L_NODES))
    log()
    log("Step 1: truncated kernel K_M at the midpoint nodes")
    global _KERN
    kern = kernel_at_nodes(log)
    _KERN = kern
    kmax_rad = max(k.rad() for k in kern)
    log("  largest radius of K_M(x_i): %s" % ball(kmax_rad, 5))
    log("  K_M(x_0)       = %s" % ball(kern[0]))
    log("  K_M(x_{L/2-1}) = %s" % ball(kern[-1]))
    log()
    log("Step 2: eta0, eta1 (%d x %d terms per centre), E0, E1 and the test"
        % (N_PER, L_NODES))
    t0 = time.time()
    procs = max(1, min(a.procs, len(CENTRES)))
    groups = [list(range(len(CENTRES)))[g::procs] for g in range(procs)]
    if procs == 1:
        parts = [certify_subset((groups[0], a.prec))]
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(procs) as pool:
            parts = pool.map(certify_subset, [(g, a.prec) for g in groups])
    rows = sorted((r for part in parts for r in part), key=lambda r: r["index"])
    log("  done in %.1f s" % (time.time() - t0))
    log()

    for r in rows:
        log("Centre %d:  s0 = %s + %s i" % (r["index"], r["re_s0"], r["im_s0"]))
        log("  Re s0 - r > -1 ............ %s" % r["hypothesis_Re_s0_minus_r_gt_minus1"])
        log("  Re eta0 = %s" % r["eta0_re"])
        log("  Im eta0 = %s" % r["eta0_im"])
        log("  Re eta1 = %s" % r["eta1_re"])
        log("  Im eta1 = %s" % r["eta1_im"])
        log("  |eta0|  = %s" % r["abs_eta0"])
        log("  |eta1|  = %s" % r["abs_eta1"])
        log("  E0      = %s" % r["E0"])
        log("  E1      = %s" % r["E1"])
        log("  r^2 B0/(Re s0 - r + 2)^3 = %s" % r["taylor_term"])
        log("  left side  = %s" % r["left_side"])
        log("  right side = %s" % r["right_side"])
        log("  upper bound of left  = %s" % r["left_upper"])
        log("  lower bound of right = %s" % r["right_lower"])
        log("  margin right - left  = %s" % r["margin_right_minus_left"])
        log("  left/right           = %s" % r["ratio_left_over_right"])
        log("  largest radius among Re/Im of eta0, eta1 = %s" % r["max_radius_eta"])
        log("  (eq:rouche) %s" % ("CERTIFIED" if r["certified"] else "NOT CERTIFIED"))
        log()

    log("Comparison with the values printed in Table tab:rouche (2 to 5 significant digits)")
    log("  centre   |eta0| here / table      |eta1| here / table     E0 here / table"
        "       left here / table      right here / table")
    for r, (_, _, tab) in zip(rows, CENTRES):
        f = r["_f"]
        log("  %d   %.3e / %.1e   %.6f / %.6f   %.3e / %.1e   %.3e / %.1e   %.3e / %.1e"
            % (r["index"], f[4], tab[0], f[5], tab[1], f[6], tab[2], f[7], tab[3], f[8], tab[4]))
    log()

    if not a.no_double:
        log("Cross-check (not part of the proof): IEEE double precision evaluation with numpy")
        t0 = time.time()
        try:
            dbl = double_precision_sums()
            import numpy
            log("  numpy %s, %.1f s" % (numpy.__version__, time.time() - t0))
            for r, (d0, d1) in zip(rows, dbl):
                f = r["_f"]
                err0 = abs(d0 - complex(f[0], f[1]))
                err1 = abs(d1 - complex(f[2], f[3]))
                r["double_eta0"] = [d0.real, d0.imag]
                r["double_eta1"] = [d1.real, d1.imag]
                r["double_minus_certified_eta0"] = err0
                r["double_minus_certified_eta1"] = err1
                log("  centre %d: |eta0(double) - eta0| = %.2e   |eta1(double) - eta1| = %.2e"
                    % (r["index"], err0, err1))
        except ImportError:
            log("  numpy not available; skipped")
        log()

    ok = all(r["certified"] for r in rows)
    log("RESULT: (eq:rouche) certified at %d of %d centres%s"
        % (sum(r["certified"] for r in rows), len(rows),
           "  -- ALL CERTIFIED" if ok else "  -- FAILURE"))
    log("total time %.1f s" % (time.time() - t_all))
    for r in rows:
        r.pop("_f", None)
    with open(a.out + "_intervals.json", "w") as fh:
        json.dump({
            "manuscript": "Castor Pollux I v1.tex",
            "sha256": "3bc78229172e068d3f4b5403e7ff808c565b7f6baa8694b8da2f4774f5f7556a",
            "statement": "Proposition prop:rouche, inequality (eq:rouche), Table tab:rouche",
            "method": "Arb ball arithmetic (python-flint %s, FLINT %s), %d bits"
                      % (flint.__version__, getattr(flint, "__FLINT_VERSION__", "?"), a.prec),
            "M": M_HARM, "N": N_PER, "L": L_NODES, "radius": "1/100",
            "all_certified": ok, "centres": rows}, fh, indent=1)
    logf.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

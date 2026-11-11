#!/usr/bin/env python3
# Castor and Pollux I: certificate that A(s) != 0 for 0 <= Re s <= 1, 0 <= Im s <= T0 (T0 = 102), in ball arithmetic.
# Together with the analytic bound for |Im s| >= 102 and the symmetry A(conj s) = conj A(s), this proves
# A(s) != 0 in the whole strip 0 < Re s < 1.
#
#   B(s) = s^2 A(s) = 1 + s log(2 pi) - s (psi(1 - s/2) + gamma) + s^2 E(s),
#   E(s) = sum_{k>=1} c_k / (k (2k - s)),   c_k = Sigma_{2k} - 1 = sum_{m>=2} m^(-2k)   (0 <= c_k <= 3 * 4^(-k)).
# psi is the digamma function (Arb), gamma Euler's constant.  The constants c_k are enclosed from their definition:
#   c_k in [S_k, S_k + M^(1-2k)/(2k-1)],  S_k = sum_{m=2}^{M} m^(-2k)  (exact rationals),
# and the terms k > K of E are bounded by 3 * 4^(-K) / (3 (K+1)(2K+1)) in modulus, since |2k - s| >= 2k - 1.
# No zeta function is used.  A box is accepted when the ball enclosing B over the box does not contain 0.
import sys, time
from flint import arb, acb, fmpq, ctx
import flint

PREC = 128
K = 80              # terms of E computed explicitly
MM = 40             # terms m <= MM of each c_k
T0 = 102.0
ctx.prec = PREC

out = open('A_nonvanishing_check_output.txt', 'w')
def log(s=''):
    print(s, flush=True); out.write(s + '\n')

# ---- enclosures of c_k = Sigma_{2k} - 1
c = []
for k in range(1, K + 1):
    S = sum(fmpq(1, m ** (2 * k)) for m in range(2, MM + 1))
    tail = fmpq(1, (2 * k - 1) * MM ** (2 * k - 1))
    lo, hi = arb(S), arb(S + tail)
    c.append(arb((lo + hi) / 2, ((hi - lo) / 2).upper()))
tailE = arb(fmpq(1, 4 ** K)) / ((K + 1) * (2 * K + 1))           # bound for |sum_{k>K}|
LOG2PI = (2 * arb.pi()).log()
EG = arb.const_euler()


def B_on_box(s0, s1, t0, t1):
    re = arb((arb(s0) + arb(s1)) / 2, ((arb(s1) - arb(s0)) / 2).upper())
    im = arb((arb(t0) + arb(t1)) / 2, ((arb(t1) - arb(t0)) / 2).upper())
    s = acb(re, im)
    E = acb(0)
    for k in range(1, K + 1):
        E += c[k - 1] / (k * (2 * k - s))
    E += acb(arb(0, tailE.upper()), arb(0, tailE.upper()))
    return 1 + s * LOG2PI - s * ((1 - s / 2).digamma() + EG) + s * s * E


def excludes_zero(b):
    return not (b.real.contains(0) and b.imag.contains(0))


t_start = time.time()
log('Castor and Pollux I: ball arithmetic certificate that A(s) != 0 on [0,1] x [0,%g]' % T0)
log('python-flint %s, FLINT %s, %d bits; K = %d terms of E, c_k from m <= %d' % (flint.__version__, getattr(flint, '__FLINT_VERSION__', '?'), PREC, K, MM))
# initial boxes 0.05 x 0.5, adaptive bisection
stack = []
ns, nt = 20, int(T0 / 0.5)
for i in range(ns):
    for j in range(nt):
        stack.append((fmpq(i, ns), fmpq(i + 1, ns), fmpq(j, 2), fmpq(j + 1, 2), 0))
accepted, maxdepth, minB = 0, 0, None
failed = []
while stack:
    s0, s1, t0, t1, d = stack.pop()
    b = B_on_box(s0, s1, t0, t1)
    if excludes_zero(b):
        accepted += 1
        maxdepth = max(maxdepth, d)
        lb = max(abs(b.real).lower(), abs(b.imag).lower())   # positive lower bound for |Re B| or |Im B|: B != 0 on the box
        minB = lb if minB is None or lb < minB else minB
    elif d >= 14:
        failed.append((s0, s1, t0, t1))
    else:
        sm, tm = (s0 + s1) / 2, (t0 + t1) / 2
        stack += [(s0, sm, t0, tm, d + 1), (sm, s1, t0, tm, d + 1), (s0, sm, tm, t1, d + 1), (sm, s1, tm, t1, d + 1)]
log('boxes accepted: %d, largest bisection depth: %d, boxes not decided: %d' % (accepted, maxdepth, len(failed)))
log('smallest certified lower bound of max(|Re s^2 A|, |Im s^2 A|) over the accepted boxes: %s' % minB.str(6))

# ---- the analytic bound at |t| = T0 (exact arithmetic on balls)
t = arb(T0)
lhs = (8 * arb.pi()).log() + 1 / t ** 2 + arb(2).log() * (1 + 1 / t ** 2).sqrt()
rhs = (t / 2).log()
log('analytic part, |t| >= %g:  log(8 pi) + 1/t^2 + log 2 * sqrt(1 + 1/t^2) = %s  <  log(|t|/2) = %s : %s'
    % (T0, lhs.str(8), rhs.str(8), bool(lhs < rhs)))
ok = (len(failed) == 0) and bool(lhs < rhs)
log('RESULT: A(s) != 0 for 0 < Re s < 1 and all real Im s: %s' % ('CERTIFIED' if ok else 'NOT CERTIFIED'))
log('time %.1f s' % (time.time() - t_start))
out.close()
sys.exit(0 if ok else 1)

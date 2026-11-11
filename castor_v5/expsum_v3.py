# Castor-Pollux I v3: exponential sums over the computed zeros of G and of A (all labels in English).
#   E_G(x) = (1/79)  sum_b exp(i x Im z_b),  z_b the zeros of G with 0.2 <= Im s <= 200  (zeros_G.txt),
#   E_A(x) = (1/101) sum_b exp(i x z_b),     z_b the real zeros of A in [-60, 200]      (zeros_A.txt),
# for 0 <= x <= 2 pi.  In the 3D figure the variable x is the vertical axis.
# The zero lists were computed in the branch castor-pollux-zeros of vcasana1/PaperMath and are taken as given.
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

XMAX = 2 * np.pi
X = np.linspace(0, XMAX, 6001)
zg = np.loadtxt('zeros_G.txt'); zG = zg[:, 0] + 1j * zg[:, 1]
zA = np.loadtxt('zeros_A.txt')[:, 0]
freq = {'G': zG.imag, 'A': zA}
Ev = {n: np.exp(1j * np.outer(X, f)).sum(1) / len(f) for n, f in freq.items()}
col = {'G': 'tab:blue', 'A': 'tab:red'}
out = open('expsum_v3_output.txt', 'w')


def log(s=''):
    print(s); out.write(s + '\n')


cen = {}
for n, f in freq.items():
    c = ((np.exp(1j * f * XMAX) - 1) / (1j * f * XMAX)).mean()       # (1/(2 pi)) int_0^{2 pi} E(x) dx, closed form
    cen[n] = c
    log('%s: %d zeros, frequencies in [%.4f, %.4f]' % (n, len(f), f.min(), f.max()))
    log('   mean of the curve E_%s on [0, 2 pi] = %.5f %+.5f i   (trapezoid rule: %.5f %+.5f i)'
        % (n, c.real, c.imag, (np.trapezoid(Ev[n], X) / XMAX).real, (np.trapezoid(Ev[n], X) / XMAX).imag))
    log('   mean of |E_%s(x)| on [0, 2 pi] = %.4f,  max over x >= 0.3: %.4f' % (n, np.abs(Ev[n]).mean(), np.abs(Ev[n])[X >= 0.3].max()))
log('centroid of the zeros of G: %.5f %+.5f i;  Re s in [%.4f, %.4f]' % (zG.mean().real, zG.mean().imag, zG.real.min(), zG.real.max()))
log('centroid of the real zeros of A: %.5f' % zA.mean())


def is_prime(n): return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))
def is_prime_power(n):
    for p in range(2, n + 1):
        if is_prime(p) and n % p == 0:
            while n % p == 0: n //= p
            return n == 1
    return False


log('')
log('E_G(log n) for 2 <= n <= 31:   n   kind   Re   Im   modulus')
vals = {}
for n in range(2, 32):
    v = np.exp(1j * freq['G'] * math.log(n)).mean()
    kind = 'prime' if is_prime(n) else ('prime power' if is_prime_power(n) else 'other')
    vals[n] = (kind, v)
    log('%3d  %-11s  %+.3f  %+.3f  %.3f' % (n, kind, v.real, v.imag, abs(v)))
for kind in ('prime', 'prime power', 'other'):
    m = [abs(v) for k, v in vals.values() if k == kind]
    r = [v.real for k, v in vals.values() if k == kind]
    log('%-11s: modulus between %.3f and %.3f; real part between %+.3f and %+.3f' % (kind, min(m), max(m), min(r), max(r)))
xa = X[(X > 0.3)]
log('E_A: |E_A(pi)| = %.4f, |E_A(2 pi)| = %.4f; max of |E_A| on [0.3, 2 pi] outside |x - pi| < 0.2 and x > 2 pi - 0.2: %.4f'
    % (abs(np.exp(1j * zA * np.pi).mean()), abs(np.exp(1j * zA * 2 * np.pi).mean()),
       np.abs(Ev['A'])[(X > 0.3) & (np.abs(X - np.pi) > 0.2) & (X < 2 * np.pi - 0.2)].max()))
out.close()

# ---- 3D: x is the vertical axis
fig = plt.figure(figsize=(8, 11))
ax = fig.add_subplot(111, projection='3d')
for n in ('G', 'A'):
    ax.plot(Ev[n].real, Ev[n].imag, X, color=col[n], lw=0.7, label=r'$\mathcal{E}_%s(x)$, %d zeros' % (n, len(freq[n])))
    ax.plot([cen[n].real] * 2, [cen[n].imag] * 2, [0, XMAX], color=col[n], ls='--', lw=1.5, label=r'mean of $\mathcal{E}_%s$' % n)
ax.set_xlabel('Re E(x)'); ax.set_ylabel('Im E(x)'); ax.set_zlabel('x')
ax.set_zlim(0, XMAX); ax.set_box_aspect((1, 1, 2.2))
ax.set_title('Exponential sums over the zeros of G and of A; the variable x is vertical', fontsize=10)
ax.legend(loc='upper left', fontsize=8); ax.view_init(elev=14, azim=-58)
fig.savefig('fig_expsum_3d.png', dpi=200, bbox_inches='tight', pad_inches=0.15); plt.close(fig)

# ---- the curves in the complex plane, and the zeros with their centroids
fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5.5))
for n in ('G', 'A'):
    ax2[0].plot(Ev[n].real, Ev[n].imag, color=col[n], lw=0.6, label=r'$\mathcal{E}_%s$' % n)
    ax2[0].plot(cen[n].real, cen[n].imag, 'o', color=col[n], ms=9, mec='k', label=r'mean of $\mathcal{E}_%s$' % n)
ax2[1].plot(zG.real, zG.imag, '.', color=col['G'], ms=5, label='zeros of G')
ax2[1].plot(zG.mean().real, zG.mean().imag, '*', color=col['G'], ms=16, mec='k', label='centroid of the zeros of G')
ax2[1].plot(zA, 0 * zA, '.', color=col['A'], ms=5, label='real zeros of A')
ax2[1].plot(zA.mean(), 0, '*', color=col['A'], ms=16, mec='k', label='centroid of the zeros of A')
ax2[0].set_xlabel('Re E'); ax2[0].set_ylabel('Im E'); ax2[0].set_aspect('equal')
ax2[0].set_title('The curves E(x), 0 <= x <= 2 pi, in the complex plane'); ax2[0].legend(fontsize=8)
ax2[1].set_xlabel('Re s'); ax2[1].set_ylabel('Im s'); ax2[1].set_title('Zeros and their centroids'); ax2[1].legend(fontsize=8)
fig2.tight_layout(); fig2.savefig('fig_expsum_plane.png', dpi=200); plt.close(fig2)

# ---- moduli
fig3, ax3 = plt.subplots(2, 1, figsize=(11, 7.5))
xx = np.linspace(0.3, 4.2, 20001)
mg = np.abs(np.exp(1j * np.outer(xx, freq['G'])).mean(1))
ax3[0].plot(xx, mg, color=col['G'], lw=0.9)
for n in range(2, 67):
    if is_prime_power(n):
        ax3[0].axvline(math.log(n), color='0.6' if not is_prime(n) else 'tab:green', lw=0.6, ls=':' if not is_prime(n) else '-')
        if n <= 31: ax3[0].text(math.log(n), 0.275, str(n), ha='center', fontsize=7)
ax3[0].set_ylim(0, 0.295); ax3[0].set_xlabel('x'); ax3[0].set_ylabel('|E_G(x)|')
ax3[0].set_title('|E_G(x)|; vertical lines at x = log n: primes (green), other prime powers (grey, dotted)', fontsize=10)
for n in ('G', 'A'):
    ax3[1].plot(X, np.abs(Ev[n]), color=col[n], lw=0.7, label=r'$|\mathcal{E}_%s(x)|$' % n)
ax3[1].set_xlabel('x'); ax3[1].set_ylabel('|E(x)|'); ax3[1].set_ylim(0, 1); ax3[1].legend()
ax3[1].set_title('|E_G(x)| and |E_A(x)| on [0, 2 pi]', fontsize=10)
fig3.tight_layout(); fig3.savefig('fig_expsum_modulus.png', dpi=200); plt.close(fig3)

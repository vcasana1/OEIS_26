# Castor-Pollux I: heat maps of the Gram matrix of the factors h(x/k), k <= 10 (labels in English).
# Left: mean over a period, (pi^2/12) gcd(j,k)^2/(jk).  Right: weight x^{-2 sigma-1}, sigma = 3/4, normalised bracket
#   2 sigma max(j,k)^{2 sigma} I_{jk}(sigma) = T(0) + sum_r D(r) kappa(2 pi max(j,k) r/(jk))   (series truncated at r <= R).
import mpmath as mp, numpy as np
from math import gcd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
mp.mp.dps = 15
def T(j,k,r):
    g = gcd(j,k); jp, kp = j//g, k//g
    if r % g: return mp.mpf(0)
    rp = r//g
    if rp == 0: return mp.pi**2/(12*jp*kp)
    m = 1
    while (m*kp - rp) % jp or (m*kp - rp)//jp < 1: m += 1
    n = (m*kp - rp)//jp
    return (mp.digamma(mp.mpf(m)/jp) - mp.digamma(mp.mpf(n)/kp))/(2*rp)
def U(j,k,r):
    s = mp.mpf(0)
    for m in range(1, r//k + 1):
        if (r - m*k) % j == 0 and (r - m*k)//j >= 1: s += mp.mpf(1)/(2*m*((r - m*k)//j))
    return s
def kappa(sig, y): return 2*sig*mp.re(mp.expint(2*sig+1, -1j*y))
K, sig, R = 10, mp.mpf('0.75'), 1500
flat = np.zeros((K,K)); wt = np.zeros((K,K))
for j in range(1, K+1):
    for k in range(j, K+1):
        flat[j-1,k-1] = flat[k-1,j-1] = float(T(j,k,0))
        tot = T(j,k,0); g = gcd(j,k)
        for r in range(g, R+1, g): tot += (T(j,k,r) + T(j,k,-r) + U(j,k,r))*kappa(sig, 2*mp.pi*max(j,k)*r/(j*k))
        wt[j-1,k-1] = wt[k-1,j-1] = float(tot)
np.savetxt('gram_flat.csv', flat, delimiter=',', fmt='%.8f'); np.savetxt('gram_weighted.csv', wt, delimiter=',', fmt='%.8f')
print("eigenvalues of the mean Gram matrix:", np.round(np.linalg.eigvalsh(flat), 5))
print("eigenvalues of the normalised weighted bracket matrix:", np.round(np.linalg.eigvalsh(wt), 5))
print("negative entries of the weighted bracket:", int((wt < 0).sum()), "of", K*K)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6)); v = max(abs(flat).max(), abs(wt).max())
for ax, Mx, title in ((axs[0], flat, 'Mean over a period: (pi^2/12) gcd(j,k)^2/(jk)'), (axs[1], wt, 'Weight x^(-2 sigma - 1), sigma = 0.75 (normalised)')):
    im = ax.imshow(Mx, cmap='RdBu_r', vmin=-v, vmax=v, extent=(0.5, K+0.5, K+0.5, 0.5)); ax.set_xlabel('k'); ax.set_ylabel('j'); ax.set_title(title, fontsize=9); fig.colorbar(im, ax=ax)
fig.suptitle('Gram matrix of the factors h(x/k), k = 1,...,10'); fig.tight_layout(); fig.savefig('fig_gram.png', dpi=200)

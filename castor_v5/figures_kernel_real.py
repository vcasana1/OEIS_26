# Kernel and real-axis plots. G is computed from the positive kernel K:
#   G(s) = -(s+1)(s+2) H(s)/(4 pi^2),  H(s) = int_0^1 K(x) sum_{n>=1} (n+x)^{-s-3} dx  (midpoint rule + Euler-Maclaurin tail).
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
M, N, L = 2000, 24, 8192
x = (np.arange(L) + 0.5)/L
K = np.zeros(L)
for m0 in range(1, M+1, 200):
    m = np.arange(m0, min(m0+200, M+1))[:, None]
    K += ((1 - np.cos(2*np.pi*m*x[None, :]))/m**3).sum(axis=0)
def H(s):
    a = s + 3; tot = 0j
    for n in range(1, N+1): tot += (K*(n + x)**(-a)).sum()
    b = N + 1 + x
    tot += (K*(b**(1-a)/(a-1) + b**(-a)/2 + a*b**(-a-1)/12 - a*(a+1)*(a+2)*b**(-a-3)/720)).sum()
    return tot/L
def G(s): return -(s+1)*(s+2)*H(s)/(4*np.pi**2)
# kernel
u = np.linspace(0.002, 0.998, 2000)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(u, np.log(2*np.sin(np.pi*u))); axs[0].axhline(0, color='k', lw=0.5); axs[0].set_title('h(u) = log|2 sin(pi u)| on one period'); axs[0].set_xlabel('u'); axs[0].set_ylabel('h(u)')
axs[1].plot(x, K); axs[1].set_title('Positive kernel K(x) = sum (1 - cos(2 pi m x))/m^3'); axs[1].set_xlabel('x'); axs[1].set_ylabel('K(x)')
fig.tight_layout(); fig.savefig('fig_kernel.png', dpi=200)
# G on the real axis
sr = np.linspace(-1.9, 3, 99); Gr = np.array([G(complex(v, 0)).real for v in sr])
fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(sr, Gr); ax.axhline(0, color='k', lw=0.5); ax.plot([-1], [0], 'o', color='tab:red', label='simple zero at s = -1')
ax.set_xlabel('s (real)'); ax.set_ylabel('G(s)'); ax.set_title('G on the real axis: positive on (-2,-1), negative for s > -1'); ax.legend()
fig.tight_layout(); fig.savefig('fig_G_real.png', dpi=200)
print("G(1) =", G(1+0j).real, " G(-1) =", G(-1+0j).real)

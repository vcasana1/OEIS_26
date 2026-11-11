# Castor-Pollux I v3: double precision evaluators of G and A (no zeta function is used for G or A).
#
#   G(s) = -(s+1)(s+2) H(s)/(4 pi^2),   H(s) = int_0^1 K(x) sum_{n>=1} (n+x)^(-s-3) dx,
#   K(x) = sum_{m>=1} (1 - cos(2 pi m x))/m^3 = Sigma_3 - Re Li_3(exp(2 pi i x)).
# The integral over x is a Gauss-Legendre rule on [0,1/2] and on [1/2,1]; the sum over n is taken
# directly for n <= N and by the Euler-Maclaurin formula for n > N.
#
#   A(s) = log(2 pi)/s + 1/s^2 + sum_{k>=1} Sigma_{2k}/(k (2k - s)),  Sigma_{2k} = sum_m m^(-2k),
# evaluated as log(2 pi)/s + 1/s^2 - (psi(1 - s/2) + gamma)/s + sum_k (Sigma_{2k} - 1)/(k (2k - s)).
import numpy as np
import mpmath as mp
from scipy.special import digamma

NODES = 120
_g, _w = np.polynomial.legendre.leggauss(NODES)
X = np.concatenate([(_g + 1) / 4, 0.5 + (_g + 1) / 4])
W = np.concatenate([_w / 4, _w / 4])
mp.mp.dps = 25
SIGMA3 = float(mp.zeta(3))
KX = np.array([float(mp.zeta(3) - mp.re(mp.polylog(3, mp.exp(2j * mp.pi * mp.mpf(float(x)))))) for x in X])
C = W * KX                                    # weights of the rule times the kernel
KMAX = 70
KK = np.arange(1, KMAX + 1)
S2K1 = np.array([float(mp.zeta(2 * k) - 1) for k in range(1, KMAX + 1)])   # Sigma_{2k} - 1


def H_grid(sig, t, N=None):
    """H(sig_a + i t_b) on the tensor grid, shape (len(t), len(sig))."""
    sig = np.atleast_1d(np.asarray(sig, float))
    t = np.atleast_1d(np.asarray(t, float))
    if N is None:
        N = int(max(60, 0.6 * (np.abs(t).max() + np.abs(sig).max() + 3)))
    n = np.arange(1, N + 1)
    U = (n[:, None] + X[None, :]).ravel()
    CU = np.tile(C, N)
    LU = np.log(U)
    out = np.zeros((len(t), len(sig)), complex)
    step = max(1, int(1e7 // len(U)))                                      # memory: blocks of about 1e7 entries
    for c in range(0, len(sig), step):
        P = CU[None, :] * np.exp(-(sig[c:c + step, None] + 3.0) * LU[None, :])   # (nsig', J)
        for a in range(0, len(t), step):
            Q = np.exp(-1j * t[a:a + step, None] * LU[None, :])            # (nt', J)
            out[a:a + step, c:c + step] = Q @ P.T
    # Euler-Maclaurin tail from y = N + 1 + x
    y = N + 1 + X
    ly = np.log(y)
    z = (sig[None, :] + 3.0) + 1j * t[:, None]
    E = np.exp(-1j * t[:, None] * ly[None, :])                             # (nt, nodes)

    def T(p):                                                              # sum_j C_j y_j^(-z-p)
        return E @ (C[None, :] * np.exp(-(sig[:, None] + 3.0 + p) * ly[None, :])).T

    out += (T(-1) / (z - 1) + T(0) / 2 + z * T(1) / 12 - z * (z + 1) * (z + 2) * T(3) / 720
            + z * (z + 1) * (z + 2) * (z + 3) * (z + 4) * T(5) / 30240
            - z * (z + 1) * (z + 2) * (z + 3) * (z + 4) * (z + 5) * (z + 6) * T(7) / 1209600)
    return out


def G_grid(sig, t, N=None):
    s = np.asarray(sig, float)[None, :] + 1j * np.asarray(t, float)[:, None]
    return -(s + 1) * (s + 2) * H_grid(sig, t, N) / (4 * np.pi ** 2)


def G(points, N=None):
    """G at arbitrary complex points (one by one)."""
    pts = np.atleast_1d(np.asarray(points, complex))
    return np.array([G_grid([p.real], [p.imag], N)[0, 0] for p in pts])


def A(s):
    s = np.asarray(s, complex)
    v = np.log(2 * np.pi) / s + 1 / s ** 2 - (digamma(1 - s / 2) + np.euler_gamma) / s
    return v + (S2K1 / (KK * (2 * KK - s[..., None]))).sum(-1)


def A_real(x):
    return float(A(np.array([complex(x, 0.0)]))[0].real)

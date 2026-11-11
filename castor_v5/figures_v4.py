# Surfaces on a locally refined grid, with linear inverted vertical axes.
# Each red marker uses the computed modulus at its mesh vertex.
# G and A are evaluated with cp_eval.py (kernel K only; no zeta function).  zeta is used only for its own figure,
# for the six points 1/2 + i t and for the check of the grid relation.
import time
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patheffects import Normal, Stroke
import cp_eval as E

out = open('figures_v4_output.txt', 'w')


def log(s=''):
    print(s, flush=True)
    out.write(s + '\n')


def zeta_grid(sig, t):
    try:
        from flint import acb, ctx
        ctx.prec = 64
        f = lambda a, b: complex(acb(a, b).zeta())
    except ImportError:
        f = lambda a, b: complex(mp.zeta(mp.mpc(a, b)))
    return np.array([[f(float(a), float(b)) for a in sig] for b in t])


# ---------------------------------------------------------------- data
zg = np.loadtxt('zeros_G.txt')
zG = zg[:, 0] + 1j * zg[:, 1]
zG40 = zG[zG.imag <= 40]
mp.mp.dps = 20
zD = np.array([0.5 + 1j * float(mp.zetazero(k).imag) for k in range(1, 7)])      # points 1/2 + i t
log('zeros of G with Im s <= 40: %d;  points 1/2 + i t: %d' % (len(zG40), len(zD)))

# local refinement: grid lines through every zero and at these distances from it, so that each peak is resolved
OFF_S = np.array([-0.018, -0.009, -0.003, -0.001, 0.0, 0.001, 0.003, 0.009, 0.018])
OFF_T = 12 * OFF_S
pts = np.concatenate([zG40, zD])
sig = np.concatenate([np.round(np.arange(0.05, 1.9501, 0.01), 2)] + [z.real + OFF_S for z in pts])
sig = np.unique(sig[(sig >= 0.05) & (sig <= 1.95)])
t_grid = np.concatenate([np.round(np.arange(1.0, 40.0001, 0.05), 2)] + [z.imag + OFF_T for z in pts])
tt = np.unique(t_grid[(t_grid >= 1.0) & (t_grid <= 40.0)])
S, T = np.meshgrid(sig, tt)
t0 = time.time()
Gv = E.G_grid(sig, tt)
Av = E.A(S + 1j * T)
Zv = zeta_grid(sig, tt)
log('fine grid: %d (Re s) x %d (Im s) points, %.0f s' % (len(sig), len(tt), time.time() - t0))
sc = S + 1j * T
chk = np.abs(Gv - Av - (np.pi / sc) / np.tan(np.pi * sc / 2) * Zv).max()
log('max |G - A - (pi/s) cot(pi s/2) zeta| on the fine grid = %.3e' % chk)
k = np.unravel_index(np.abs(Av).argmin(), Av.shape)
log('min |A| on the fine grid = %.6f at s = %.2f + %.2f i;  max |A| = %.4f' % (np.abs(Av).min(), sig[k[1]], tt[k[0]], np.abs(Av).max()))
gd = np.abs(E.G(zD) - E.A(zD))
log('|G - A| at the six points 1/2 + i t: ' + ' '.join('%.1e' % v for v in gd))
log('|G| at the six zeros of G (double precision evaluator): ' + ' '.join('%.1e' % v for v in np.abs(E.G(zG40))))
np.savez_compressed('grid_values_v4.npz', sigma=sig, t=tt, G=Gv, A=Av, zeta=Zv)


def surface(V, title, fname, zlabel, marks, marklabel, Sg=S, Tg=T, elev=52, azim=-66, top=None, figsize=(9, 7.5)):
    """|F| on a linear vertical axis, inverted (0 at the top): a zero of F is a peak whose tip is at 0."""
    Zs = np.abs(V)
    if top is not None:
        Zs = np.minimum(Zs, top)
    columns = np.searchsorted(Sg[0], marks.real)
    rows = np.searchsorted(Tg[:, 0], marks.imag)
    if not (np.array_equal(Sg[rows, columns], marks.real)
            and np.array_equal(Tg[rows, columns], marks.imag)):
        raise ValueError(f'{fname}: each marked zero must be a mesh vertex')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.plot_surface(Sg, Tg, Zs, cmap='viridis_r', zorder=1, linewidth=0, antialiased=False, rcount=Sg.shape[0], ccount=Sg.shape[1])
    if len(marks):
        ax.scatter(marks.real, marks.imag, Zs[rows, columns],
                   color='#d21d2c', edgecolor='white', linewidth=0.5,
                   s=40, depthshade=False, label=marklabel, zorder=10)
        ax.legend(loc='upper left', fontsize=9)
    ax.set_xlabel('Re s'); ax.set_ylabel('Im s'); ax.set_zlabel(zlabel); ax.set_title(title, fontsize=10)
    ax.set_zlim(0, Zs.max()); ax.invert_zaxis()
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
    if len(marks):
        log('%s: %d marked mesh vertices, largest plotted marker height %.3e'
            % (fname, len(marks), Zs[rows, columns].max()))
    else:
        log('wrote %s' % fname)


none = np.array([], complex)
surface(Gv, '|G(s)| on an inverted linear scale', 'fig_G_surface.png', '|G(s)|', zG40, 'computed zeros of G')
surface(Av, '|A(s)| on the same rectangle, vertical axis inverted', 'fig_A_surface.png', '|A(s)|', none, '')
surface(Gv - Av, '|G(s) - A(s)| on an inverted linear scale (clipped at 0.6)', 'fig_GminusA_surface.png', '|G(s) - A(s)|', zD, 'points with G = A', top=0.6)
surface(Zv, '|zeta(s)| on an inverted linear scale (clipped at 1.5)', 'fig_zeta_surface.png', '|zeta(s)|', zD, 'computed zeros of zeta', top=1.5)

# ---------------------------------------------------------------- A near the real axis: real zeros and poles
zA_all = np.loadtxt('zeros_A.txt')[:, 0]
zA = zA_all[(zA_all > -3.5) & (zA_all < 9.5)]
OFF_A = np.array([-0.03, -0.015, -0.006, -0.002, 0.0, 0.002, 0.006, 0.015, 0.03])
sa = np.unique(np.concatenate([np.arange(-3.495, 9.5, 0.01)] + [z + OFF_A for z in zA]))
ta = np.unique(np.concatenate([np.round(np.arange(-2.0, 2.0001, 0.01), 2), OFF_A]))
Sa, Ta = np.meshgrid(sa, ta)
Aa = E.A(Sa + 1j * Ta)
log('A near the real axis: %d x %d points; real zeros shown: %s' % (len(sa), len(ta), ' '.join('%.4f' % v for v in zA)))
surface(Aa, '|A(s)| near the real axis on an inverted scale (clipped at 1.5)',
        'fig_A_real_surface.png', '|A(s)|', zA.astype(complex), 'real zeros of A', Sg=Sa, Tg=Ta, elev=48, azim=-78, top=1.5, figsize=(11, 6.5))

# ---------------------------------------------------------------- zero map on the fine grid
Psi = Gv / Av
fig, ax = plt.subplots(figsize=(6, 9))
background = LinearSegmentedColormap.from_list('light_gray', ['#ffffff', '#d2d9e3'])
log_modulus = np.log10(np.maximum(np.abs(Gv), np.finfo(float).tiny))
pc = ax.pcolormesh(S, T, log_modulus, cmap=background, shading='auto')
fig.colorbar(pc, ax=ax, label='log10 |G(s)|')
arc_color = '#0055ae'
arcs = ax.contour(S, T, np.where((Psi.real > 0) & (Psi.real < 1), Psi.imag, np.nan),
                  levels=[0], colors=[arc_color], linewidths=2.0)
arcs.set_path_effects([Stroke(linewidth=3.6, foreground='white'), Normal()])
ax.plot(zG40.real, zG40.imag, 'o', color='tab:red', label='zeros of G')
ax.plot(zD.real, zD.imag, 'x', color='tab:green', ms=9, mew=2, label='points with G = A')
ax.plot([], [], color=arc_color, linewidth=2.0, label='Im(G/A) = 0 with 0 < Re(G/A) < 1')
ax.set_xlabel('Re s'); ax.set_ylabel('Im s'); ax.set_title('Zeros of G, points with G = A, and computed arcs'); ax.legend(loc='lower left', fontsize=8)
fig.tight_layout(); fig.savefig('fig_zero_map.png', dpi=200); plt.close(fig)
log('wrote fig_zero_map.png')
out.close()

"""
BTP (2D): PINN, VPINN and RVPINN for the 2-D diffusion-advection problem (41) of
Rojas et al., "Robust Variational Physics-Informed Neural Networks" (arXiv:2308.16910),
on the unit square, with
  (a) --case smooth    : eps = 1, beta = 0, u = sin(pi x) sin(pi y)   (paper eq. 58, Figs. 4-5)
  (b) --case interface : piecewise-constant, DISCONTINUOUS advection across x = 1/2

Problem
    -eps Lap(u) + beta . grad(u) = f   in  Omega = (0,1)^2,      u = 0 on the boundary

Interface case:  beta(x,y) = (0, b_minus)  for x < 1/2,   (0, b_plus)  for x > 1/2.
  The flow is parallel to the interface and only its speed jumps.  Since beta_x = 0,
  div(beta) = 0 in the sense of distributions (the normal component beta.n = 0 is
  continuous across x = 1/2), so (beta.grad v, v) = 0 and a(v,v) = ||v||_V^2 exactly:
  the coercivity/inf-sup constant of the paper is alpha = 1 even though beta jumps.
  Interface conditions at x = 1/2:  [u] = 0,  [eps du/dx] = 0  (natural in the weak form).
  Manufactured exact solution (u, grad u continuous; u_xx jumps at x = 1/2):
      u(x,y) = g(x) sin(pi y),   g(x) = sin(pi x) + kappa (x-1/2)_+^2 (1-x)
      [u_xx](1/2, y) = kappa sin(pi y);   f := -eps Lap(u) + beta.grad(u)  (discontinuous)

Weak form:  a(u,v) = eps (grad u, grad v) + (beta.grad u, v),  l(v) = (f, v),
            r(u,v) = l(v) - a(u,v),   ||v||_V^2 = eps ||grad v||^2.

Methods (strong BCs: u_theta = x(1-x) y(1-y) N_theta(x,y)):
  pinn        mean squared strong residual at random collocation points
  vpinn       sum_{m,n} r(u, s_mn)^2,   s_mn = sin(m pi x) sin(n pi y)
  rvpinn_sp   RVPINN, spectral: s_mn are V-orthogonal with ||s_mn||_V^2 = eps pi^2 (m^2+n^2)/4,
              so  L = 4/(eps pi^2) sum r(u,s_mn)^2 / (m^2+n^2)
  rvpinn_fem  RVPINN, tensor-product piecewise-linear (Q1) hats on a uniform mesh,
              L = R^T G^{-1} R,  G = eps (K (x) M + M (x) K).  G^{-1} is applied exactly by
              fast diagonalisation (1-D generalised eigenproblem K v = lambda M v), so the
              (n^2 x n^2) Gram matrix is never formed.

Networks:  standard = 4 hidden layers x 40 tanh neurons (paper, 2-D examples)
           enriched = x(1-x)y(1-y) [N0 + (x-1/2)_+^2 N1]  -> allows the jump of u_xx at x = 1/2

All integrals: tensor-product composite Gauss-Legendre with a breakpoint at x = 1/2.

Usage
    python btp_2d.py --case smooth   --nets standard              # paper benchmark
    python btp_2d.py --case interface                             # discontinuous beta, both nets
    python btp_2d.py --case interface --methods rvpinn_sp --M_spec 30 --epochs 20000
Outputs (in --outdir): per-run figures, comparison figure, exact-solution figure,
training histories (.npz) and results.csv.
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import scipy.linalg as sla
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
PI = math.pi


# ==========================================================================
# Exact solutions
# ==========================================================================
class Smooth2D:
    """eps = 1, beta = 0, u = sin(pi x) sin(pi y)  (paper eq. 58)."""
    name = "smooth"

    def __init__(self, eps=1.0):
        self.eps, self.bm, self.bp = eps, 0.0, 0.0
        self.desc = r"$\varepsilon=1$, $\beta=0$, $u=\sin(\pi x)\sin(\pi y)$"

    def u(self, x, y):
        return torch.sin(PI * x) * torch.sin(PI * y)

    def grad(self, x, y):
        return (PI * torch.cos(PI * x) * torch.sin(PI * y), PI * torch.sin(PI * x) * torch.cos(PI * y))

    def uxx(self, x, y):
        return -PI**2 * self.u(x, y)

    def lap(self, x, y):
        return -2 * PI**2 * self.u(x, y)

    def beta(self, x, y):
        z = torch.zeros_like(x)
        return z, z

    def f(self, x, y):
        bx, by = self.beta(x, y)
        ux, uy = self.grad(x, y)
        return -self.eps * self.lap(x, y) + bx * ux + by * uy


class Interface2D(Smooth2D):
    """beta = (0, bm) for x < 1/2, (0, bp) for x > 1/2;  u = g(x) sin(pi y)."""
    name = "interface"

    def __init__(self, eps=0.1, bm=1.0, bp=3.0, kappa=8.0):
        self.eps, self.bm, self.bp, self.kappa = eps, bm, bp, kappa
        self.desc = (rf"$\varepsilon={eps}$, $\beta=(0,{bm})$ for $x<1/2$, "
                     rf"$(0,{bp})$ for $x>1/2$, $\kappa={kappa}$")

    def g(self, x, k):
        s = torch.relu(x - 0.5)
        H = (x > 0.5).to(x.dtype)
        K = self.kappa
        if k == 0:
            return torch.sin(PI * x) + K * s**2 * (1 - x)
        if k == 1:
            return PI * torch.cos(PI * x) + K * (2 * s * (1 - x) - s**2)
        return -PI**2 * torch.sin(PI * x) + K * (2 * H * (1 - x) - 4 * s)

    def u(self, x, y):
        return self.g(x, 0) * torch.sin(PI * y)

    def grad(self, x, y):
        return self.g(x, 1) * torch.sin(PI * y), PI * self.g(x, 0) * torch.cos(PI * y)

    def uxx(self, x, y):
        return self.g(x, 2) * torch.sin(PI * y)

    def lap(self, x, y):
        return (self.g(x, 2) - PI**2 * self.g(x, 0)) * torch.sin(PI * y)

    def beta(self, x, y):
        by = torch.where(x < 0.5, torch.full_like(x, self.bm), torch.full_like(x, self.bp))
        return torch.zeros_like(x), by


# ==========================================================================
# Quadrature (1-D composite Gauss-Legendre; tensor products are built from it)
# ==========================================================================
def gauss_1d(breaks, n_sub=1, n_gauss=5):
    g, w = np.polynomial.legendre.leggauss(n_gauss)
    nodes = np.unique(np.concatenate([np.linspace(a, b, n_sub + 1)
                                      for a, b in zip(breaks[:-1], breaks[1:])]))
    h = np.diff(nodes)
    x = (nodes[:-1, None] + 0.5 * h[:, None] * (g[None, :] + 1)).ravel()
    wq = (0.5 * h[:, None] * w[None, :]).ravel()
    return torch.tensor(x), torch.tensor(wq)


def grid(x1, y1):
    """Tensor grid; returns column vectors X, Y of length len(x1)*len(y1), x-major."""
    X, Y = torch.meshgrid(x1, y1, indexing="ij")
    return X.reshape(-1, 1), Y.reshape(-1, 1)


def D(out, inp):
    return torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True)[0]


# ==========================================================================
# Networks with strong Dirichlet BCs
# ==========================================================================
class Net(nn.Module):
    def __init__(self, enriched=False, width=40, depth=4):
        super().__init__()
        self.enriched = enriched
        layers, d = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, 2 if enriched else 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x, y):
        n = self.body(torch.cat([x, y], dim=1))
        if self.enriched:
            n = n[:, :1] + torch.relu(x - 0.5) ** 2 * n[:, 1:2]
        return x * (1 - x) * y * (1 - y) * n


def eval_grad(model, X, Y):
    X = X.clone().requires_grad_(True)
    Y = Y.clone().requires_grad_(True)
    u = model(X, Y)
    return u, D(u, X), D(u, Y)


def eval_all(model, X, Y):
    """u, u_x, u_y, u_xx, Laplacian (with graph, for PINN)."""
    X = X.clone().requires_grad_(True)
    Y = Y.clone().requires_grad_(True)
    u = model(X, Y)
    ux, uy = D(u, X), D(u, Y)
    uxx, uyy = D(ux, X), D(uy, Y)
    return u, ux, uy, uxx, uxx + uyy


# ==========================================================================
# Losses
# ==========================================================================
class PINNLoss:
    robust = False

    def __init__(self, ex, n_col=2000):
        self.ex, self.n_col = ex, n_col

    def __call__(self, model):
        X, Y = torch.rand(self.n_col, 1), torch.rand(self.n_col, 1)
        _, ux, uy, _, lap = eval_all(model, X, Y)
        bx, by = self.ex.beta(X, Y)
        res = -self.ex.eps * lap + bx * ux + by * uy - self.ex.f(X, Y)
        return torch.mean(res**2)


class WeakLoss2D:
    """Weak residual against tensor-product test functions phi_m(x) psi_n(y).

    R[m,n] = (f, phi_m psi_n) - eps (u_x, phi_m' psi_n) - eps (u_y, phi_m psi_n')
             - (beta.grad u, phi_m psi_n)
    All integrals are matrix products on the tensor quadrature grid.
    """

    def __init__(self, ex, x1, wx, y1, wy, A, dA, B, dB, mode, extra=None):
        self.ex, self.mode, self.extra = ex, mode, extra
        self.Qx, self.Qy = len(x1), len(y1)
        self.X, self.Y = grid(x1, y1)
        self.W = wx[:, None] * wy[None, :]                       # (Qx, Qy)
        self.A, self.dA, self.B, self.dB = A, dA, B, dB          # (Qx, M), (Qy, N)
        F = ex.f(self.X, self.Y).view(self.Qx, self.Qy)
        self.Fm = A.T @ (self.W * F) @ B                         # load matrix (M, N)
        bx, by = ex.beta(self.X, self.Y)
        self.bx, self.by = bx.view(self.Qx, self.Qy), by.view(self.Qx, self.Qy)
        self.robust = mode != "vpinn"

    def residual(self, model):
        _, ux, uy = eval_grad(model, self.X, self.Y)
        ux, uy = ux.view(self.Qx, self.Qy), uy.view(self.Qx, self.Qy)
        W, A, dA, B, dB = self.W, self.A, self.dA, self.B, self.dB
        diff = self.ex.eps * (dA.T @ (W * ux) @ B + A.T @ (W * uy) @ dB)
        adv = A.T @ (W * (self.bx * ux + self.by * uy)) @ B
        return self.Fm - diff - adv

    def __call__(self, model):
        R = self.residual(model)
        if self.mode == "vpinn":
            return torch.sum(R**2)
        if self.mode == "spec":
            return torch.sum(self.extra * R**2)                  # extra = G^{-1} (diagonal)
        V, lam_sum = self.extra                                  # fast diagonalisation
        T = V.T @ R @ V
        return torch.sum(T**2 / lam_sum)


def spectral_1d(x, M):
    m = torch.arange(1, M + 1, dtype=torch.float64)[None, :]
    return torch.sin(m * PI * x[:, None]), (m * PI) * torch.cos(m * PI * x[:, None])


def make_spectral(ex, M, robust, n_sub=12, n_gauss=5):
    x1, w1 = gauss_1d([0.0, 0.5, 1.0], n_sub, n_gauss)
    A, dA = spectral_1d(x1, M)
    m = torch.arange(1, M + 1, dtype=torch.float64)
    Ginv = 4.0 / (ex.eps * PI**2 * (m[:, None] ** 2 + m[None, :] ** 2))
    return WeakLoss2D(ex, x1, w1, x1, w1, A, dA, A, dA, "spec" if robust else "vpinn", Ginv)


def hats_1d(x, n_el):
    h = 1.0 / n_el
    nodes = torch.linspace(0, 1, n_el + 1)[1:-1][None, :]
    dist = (x[:, None] - nodes) / h
    Phi = torch.clamp(1 - dist.abs(), min=0.0)
    dPhi = torch.where(dist.abs() < 1, -torch.sign(dist) / h, torch.zeros_like(dist))
    return Phi, dPhi


def make_fem(ex, n_el, n_gauss=2):
    assert n_el % 2 == 0, "use an even number of elements so that x = 1/2 is a mesh line"
    h = 1.0 / n_el
    x1, w1 = gauss_1d(np.linspace(0, 1, n_el + 1), 1, n_gauss)
    A, dA = hats_1d(x1, n_el)
    n = n_el - 1
    K = (np.diag(2 * np.ones(n)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)) / h
    Mm = (np.diag(4 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)) * h / 6
    lam, V = sla.eigh(K, Mm)                 # V^T K V = diag(lam), V^T M V = I
    lam = torch.tensor(lam)
    lam_sum = ex.eps * (lam[:, None] + lam[None, :])
    return WeakLoss2D(ex, x1, w1, x1, w1, A, dA, A, dA, "fem", (torch.tensor(V), lam_sum))


def make_loss(method, ex, args):
    if method == "pinn":
        return PINNLoss(ex, args.n_col)
    if method == "vpinn":
        return make_spectral(ex, args.M_spec, robust=False)
    if method == "rvpinn_sp":
        return make_spectral(ex, args.M_spec, robust=True)
    if method == "rvpinn_fem":
        return make_fem(ex, args.fe_elems)
    raise ValueError(method)


# ==========================================================================
# Error measures
# ==========================================================================
class Metrics:
    def __init__(self, ex, n_sub_fast=25, n_sub_full=50, n_gauss=4):
        self.ex = ex
        self.fast_grid = self._prep(n_sub_fast, n_gauss)
        self.full_grid = self._prep(n_sub_full, n_gauss)
        g = self.full_grid
        self.nH1 = math.sqrt(ex.eps * float(torch.sum(g["W"] * (g["ux"] ** 2 + g["uy"] ** 2))))
        self.nL2 = math.sqrt(float(torch.sum(g["W"] * g["u"] ** 2)))
        self.nLap = math.sqrt(float(torch.sum(g["W"] * g["lap"] ** 2)))

    def _prep(self, n_sub, n_gauss):
        x1, w1 = gauss_1d([0.0, 0.5, 1.0], n_sub, n_gauss)
        X, Y = grid(x1, x1)
        W = (w1[:, None] * w1[None, :]).reshape(-1, 1)
        ux, uy = self.ex.grad(X, Y)
        return dict(x1=x1, X=X, Y=Y, W=W, u=self.ex.u(X, Y), ux=ux, uy=uy,
                    lap=self.ex.lap(X, Y), uxx=self.ex.uxx(X, Y))

    def _h1_l2(self, model, g, chunk=8000):
        e1 = e2 = 0.0
        for i in range(0, len(g["X"]), chunk):
            s = slice(i, i + chunk)
            u, ux, uy = eval_grad(model, g["X"][s], g["Y"][s])
            W = g["W"][s]
            e1 += float(torch.sum(W * ((ux - g["ux"][s]) ** 2 + (uy - g["uy"][s]) ** 2)))
            e2 += float(torch.sum(W * (u - g["u"][s]) ** 2))
        return math.sqrt(self.ex.eps * e1) / self.nH1, math.sqrt(e2) / self.nL2

    def fast(self, model):
        g = self.fast_grid
        e1 = e2 = 0.0
        u, ux, uy = eval_grad(model, g["X"], g["Y"])
        W = g["W"]
        nH1 = math.sqrt(self.ex.eps * float(torch.sum(W * (g["ux"] ** 2 + g["uy"] ** 2))))
        nL2 = math.sqrt(float(torch.sum(W * g["u"] ** 2)))
        e1 = math.sqrt(self.ex.eps * float(torch.sum(W * ((ux - g["ux"]) ** 2 + (uy - g["uy"]) ** 2))))
        e2 = math.sqrt(float(torch.sum(W * (u - g["u"]) ** 2)))
        return e1 / nH1, e2 / nL2

    def full(self, model, chunk=8000):
        g, ex = self.full_grid, self.ex
        H1, L2 = self._h1_l2(model, g)
        outs = {k: [] for k in ["u", "uxx", "lap", "res"]}
        for i in range(0, len(g["X"]), chunk):
            s = slice(i, i + chunk)
            X, Y = g["X"][s], g["Y"][s]
            u, ux, uy, uxx, lap = eval_all(model, X, Y)
            bx, by = ex.beta(X, Y)
            res = -ex.eps * lap + bx * ux + by * uy - ex.f(X, Y)
            for k, v in zip(outs, [u, uxx, lap, res]):
                outs[k].append(v.detach())
        F = {k: torch.cat(v) for k, v in outs.items()}
        W, X, Y = g["W"], g["X"], g["Y"]
        r = dict(H1=H1, L2=L2,
                 LAP=math.sqrt(float(torch.sum(W * (F["lap"] - g["lap"]) ** 2))) / self.nLap,
                 res_rms=math.sqrt(float(torch.sum(W * F["res"] ** 2))))
        dist_b = torch.minimum(torch.minimum(X, 1 - X), torch.minimum(Y, 1 - Y))
        masks = {"res_bdry": dist_b < 0.05,
                 "res_intf": ((X - 0.5).abs() < 0.05) & (dist_b >= 0.05),
                 "res_bulk": ((X - 0.5).abs() >= 0.05) & (dist_b >= 0.05)}
        for k, msk in masks.items():
            r[k] = math.sqrt(float(torch.sum(W[msk] * F["res"][msk] ** 2) / torch.sum(W[msk])))
        n = len(g["x1"])
        fields = {k: v.view(n, n).numpy() for k, v in F.items()}
        fields.update(x1=g["x1"].numpy(), u_ex=g["u"].view(n, n).numpy(),
                      uxx_ex=g["uxx"].view(n, n).numpy(), lap_ex=g["lap"].view(n, n).numpy())
        return r, fields


# ==========================================================================
# Training
# ==========================================================================
def train(method, net, ex, metrics, args, eval_every=50, log_every=1000):
    torch.manual_seed(args.seed)
    model = Net(enriched=(net == "enriched"))
    loss_fn = make_loss(method, ex, args)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist = dict(it=[], loss=[], H1=[], L2=[])
    best_loss, best_state, early = float("inf"), None, None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model)
        lv = loss.item()
        if lv < best_loss:                       # selection by loss only (state BEFORE the step)
            best_loss = lv
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % eval_every == 0 or ep == 1:
            eH1, eL2 = metrics.fast(model)
            hist["it"].append(ep); hist["loss"].append(lv); hist["H1"].append(eH1); hist["L2"].append(eL2)
            if ep % log_every == 0 or ep == 1:
                print(f"  [{method:10s}|{net:8s}] ep {ep:6d} loss {lv:.3e} relH1 {eH1:.3e} "
                      f"relL2 {eL2:.3e}  ({time.time() - t0:.0f}s)", flush=True)
        loss.backward()
        opt.step()
        if args.report_at and ep == args.report_at and ep < args.epochs:
            tmp = Net(enriched=(net == "enriched")); tmp.load_state_dict(best_state)
            early, _ = metrics.full(tmp)
            early["best_loss"] = best_loss
    model.load_state_dict(best_state)
    res, fields = metrics.full(model)
    res.update(time=time.time() - t0, best_loss=best_loss, early=early)
    print(f"  [{method:10s}|{net:8s}] done {res['time']:.0f}s | relH1 {res['H1']:.3e} relL2 "
          f"{res['L2']:.3e} rel Lap {res['LAP']:.3e} res_rms {res['res_rms']:.3e}\n", flush=True)
    return model, hist, res, fields, loss_fn.robust


# ==========================================================================
# Plotting
# ==========================================================================
LABEL = {"pinn": "PINN", "vpinn": "VPINN (sines)",
         "rvpinn_fem": "RVPINN (FE hats)", "rvpinn_sp": "RVPINN (spectral)"}


def _img(ax, Z, x1, title, log=False, cmap="viridis"):
    ext = [0, 1, 0, 1]
    Zp = np.log10(np.abs(Z) + 1e-16) if log else Z
    im = ax.imshow(Zp.T, origin="lower", extent=ext, cmap=cmap, aspect="equal")
    ax.set(xlabel="x", ylabel="y", title=title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_run(tag, method, net, hist, F, metrics, robust, ex, outdir):
    x1 = F["x1"]
    j = np.argmin(np.abs(x1 - 0.5))                    # slice y ~ 1/2
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    _img(ax[0, 0], F["u"], x1, r"(a) $u_\theta$")
    _img(ax[0, 1], F["u"] - F["u_ex"], x1, r"(b) error $u_\theta-u$", cmap="RdBu_r")
    _img(ax[0, 2], F["res"], x1, r"(c) $\log_{10}|$strong residual$|$", log=True, cmap="magma")
    a = ax[1, 0]
    a.plot(x1, F["u_ex"][:, j], "r-", lw=2, label="exact"); a.plot(x1, F["u"][:, j], "b--", label=LABEL[method])
    a.axvline(0.5, color="gray", lw=0.5); a.set(xlabel="x", title=rf"(d) $u$ along $y={x1[j]:.3f}$"); a.legend()
    a = ax[1, 1]
    a.plot(x1, F["uxx_ex"][:, j], "r-", lw=2, label="exact"); a.plot(x1, F["uxx"][:, j], "b--", label=LABEL[method])
    a.axvline(0.5, color="gray", lw=0.5); a.set(xlabel="x", title=rf"(e) $u_{{xx}}$ along $y={x1[j]:.3f}$"); a.legend()
    a = ax[1, 2]
    a.loglog(hist["it"], hist["H1"], "g-", label=r"$\|u-u_\theta\|_U/\|u\|_U$")
    if robust:
        a.loglog(hist["it"], np.sqrt(hist["loss"]) / metrics.nH1, "--", color="orange", lw=0.9,
                 label=r"$\sqrt{\mathcal{L}}/\|u\|_U$")
    else:
        a.loglog(hist["it"], hist["loss"], "--", color="orange", lw=0.9, label="loss")
    a.set(xlabel="Iterations", title="(f) Training history"); a.legend(fontsize=9)
    fig.suptitle(f"{LABEL[method]}, {net} network: " + ex.desc)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{tag}.png"), dpi=130); plt.close(fig)


def plot_compare(runs, net, metrics, ex, outdir):
    sel = {k: v for k, v in runs.items() if k[1] == net}
    if not sel:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for (m, _), r in sel.items():
        ax[0].loglog(r["hist"]["it"], r["hist"]["H1"], lw=1, label=LABEL[m])
        if m.startswith("rvpinn"):
            est = np.sqrt(r["hist"]["loss"]) / metrics.nH1
            ax[1].loglog(est, r["hist"]["H1"], ".", ms=2, label=LABEL[m])
    lo, hi = 1e-4, 3
    ax[1].loglog([lo, hi], [lo, hi], "k--", lw=1, label="error = estimate")
    ax[0].set(xlabel="Iterations", ylabel="relative energy error", title="Convergence"); ax[0].legend(fontsize=8)
    ax[1].set(xlabel=r"$\sqrt{\mathcal{L}}/\|u\|_U$", ylabel=r"$\|u-u_\theta\|_U/\|u\|_U$",
              title="RVPINN: true error vs loss-based estimate"); ax[1].legend(fontsize=8, markerscale=4)
    fig.suptitle(f"Comparison ({net} network): " + ex.desc)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"comparison_{net}.png"), dpi=130); plt.close(fig)


def plot_exact(ex, metrics, outdir):
    g = metrics.full_grid
    n = len(g["x1"]); x1 = g["x1"].numpy()
    f = ex.f(g["X"], g["Y"]).view(n, n).numpy()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    _img(ax[0], g["u"].view(n, n).numpy(), x1, r"exact $u$")
    _img(ax[1], g["uxx"].view(n, n).numpy(), x1, r"exact $u_{xx}$", cmap="RdBu_r")
    _img(ax[2], f, x1, r"source $f$", cmap="RdBu_r")
    fig.suptitle("Exact solution: " + ex.desc)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "exact_solution.png"), dpi=130); plt.close(fig)


# ==========================================================================
# Main
# ==========================================================================
def self_check(ex):
    """Check the exact solution: BCs, interface conditions and the formula for the Laplacian."""
    t = torch.linspace(0, 1, 201).view(-1, 1)
    z, o = torch.zeros_like(t), torch.ones_like(t)
    bc = max(float(ex.u(z, t).abs().max()), float(ex.u(o, t).abs().max()),
             float(ex.u(t, z).abs().max()), float(ex.u(t, o).abs().max()))
    xm, xp = torch.full_like(t, 0.5 - 1e-9), torch.full_like(t, 0.5 + 1e-9)
    ju = float((ex.u(xp, t) - ex.u(xm, t)).abs().max())
    jux = float((ex.grad(xp, t)[0] - ex.grad(xm, t)[0]).abs().max())
    juxx = float((ex.uxx(xp, t) - ex.uxx(xm, t)).abs().max())
    # compare analytic Laplacian with autograd of the analytic u (away from x = 1/2)
    X = torch.rand(500, 1); Y = torch.rand(500, 1)
    X = torch.where((X - 0.5).abs() < 1e-3, X + 2e-3, X).requires_grad_(True); Y.requires_grad_(True)
    u = ex.u(X, Y); ux, uy = D(u, X), D(u, Y); lap = D(ux, X) + D(uy, Y)
    gx, gy = ex.grad(X, Y)
    e_grad = max(float((ux - gx).abs().max().detach()), float((uy - gy).abs().max().detach()))
    e_lap = float((lap - ex.lap(X, Y)).abs().max().detach())
    print(f"Exact-solution check: max|u| on boundary {bc:.1e}, [u] {ju:.1e}, [u_x] {jux:.1e}, "
          f"max|[u_xx]| {juxx:.3f}, grad formula err {e_grad:.1e}, Laplacian formula err {e_lap:.1e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="interface", choices=["smooth", "interface"])
    p.add_argument("--eps", type=float, default=0.1, help="interface case only")
    p.add_argument("--beta_minus", type=float, default=1.0)
    p.add_argument("--beta_plus", type=float, default=3.0)
    p.add_argument("--kappa", type=float, default=8.0, help="size of the u_xx jump")
    p.add_argument("--methods", default="pinn,vpinn,rvpinn_fem,rvpinn_sp")
    p.add_argument("--nets", default="standard,enriched")
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--M_spec", type=int, default=20, help="sines per direction (M x M test functions)")
    p.add_argument("--fe_elems", type=int, default=50, help="FE elements per direction (even)")
    p.add_argument("--n_col", type=int, default=2000, help="PINN collocation points")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report_at", type=int, default=0, help="also report best model after this many its")
    p.add_argument("--outdir", default="results_2d")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ex = Smooth2D(1.0) if args.case == "smooth" else \
        Interface2D(args.eps, args.beta_minus, args.beta_plus, args.kappa)
    self_check(ex)
    metrics = Metrics(ex)
    plot_exact(ex, metrics, args.outdir)

    runs = {}
    for net in args.nets.split(","):
        for m in args.methods.split(","):
            model, hist, res, F, robust = train(m, net, ex, metrics, args)
            tag = f"{m}_{net}"
            plot_run(tag, m, net, hist, F, metrics, robust, ex, args.outdir)
            runs[(m, net)] = dict(hist=hist, res=res)
            np.savez(os.path.join(args.outdir, f"{tag}.npz"), nH1=metrics.nH1,
                     **{f"hist_{k}": np.array(v) for k, v in hist.items()},
                     **{f"field_{k}": v for k, v in F.items()})
        plot_compare(runs, net, metrics, ex, args.outdir)

    cols = ["H1", "L2", "LAP", "res_rms", "res_bdry", "res_intf", "res_bulk", "best_loss"]
    rows = []
    for (m, net), r in runs.items():
        if r["res"]["early"] is not None:
            rows.append((m, net, args.report_at, r["res"]["early"]))
        rows.append((m, net, args.epochs, r["res"]))
    with open(os.path.join(args.outdir, "results.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["method", "net", "iterations"] + cols + ["effectivity"])
        for m, net, it, q in rows:
            eff = math.sqrt(q["best_loss"]) / metrics.nH1 / q["H1"]
            wr.writerow([m, net, it] + [f"{q[c]:.4e}" for c in cols] + [f"{eff:.3f}"])
    print(f"\n{'method':11s} {'net':9s} {'its':>6s} {'rel H1':>9s} {'rel L2':>9s} {'rel Lap':>9s} "
          f"{'res bdry':>9s} {'res intf':>9s} {'res bulk':>9s} {'eff':>6s}")
    for m, net, it, q in rows:
        eff = math.sqrt(q["best_loss"]) / metrics.nH1 / q["H1"]
        print(f"{m:11s} {net:9s} {it:6d} {q['H1']:9.2e} {q['L2']:9.2e} {q['LAP']:9.2e} "
              f"{q['res_bdry']:9.2e} {q['res_intf']:9.2e} {q['res_bulk']:9.2e} {eff:6.2f}")


if __name__ == "__main__":
    main()

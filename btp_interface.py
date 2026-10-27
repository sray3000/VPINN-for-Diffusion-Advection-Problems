"""
BTP: 1-D diffusion-advection with a DISCONTINUOUS (piecewise-constant) advection
coefficient, solved with PINN, VPINN and RVPINN (FE-hat and spectral test spaces).
Reference: Rojas et al., "Robust Variational Physics-Informed Neural Networks"
(arXiv:2308.16910), problem (41)-(43), discrete setting of Sec. 5.3.

Problem
    -eps u''(x) + beta(x) u'(x) = f(x),   x in (-1,0) U (0,1)
    u(-1) = u(1) = 0
    beta(x) = beta_minus (x < 0),  beta_plus (x > 0)
    interface x = 0:  u(0-) = u(0+),   eps u'(0-) = eps u'(0+)

Weak form (v in H^1_0(-1,1)); the interface conditions are natural here:
    a(u,v) = eps (u', v') + (beta u', v),   l(v) = (f, v)
    r(u,v) = l(v) - a(u,v)
Norm on V:  ||v||_V^2 = eps (v', v')  (paper eq. 44)

NOTE on coercivity: since beta is not divergence-free across x = 0,
(beta v', v) = (beta_minus - beta_plus) v(0)^2 / 2.  a(v,v) >= ||v||_V^2 holds
when beta_minus >= beta_plus (default here).  The problem is well-posed either
way, but the RVPINN loss is then only equivalent to (not equal to) the error.

Exact solution (f = f0 constant): on each side u = A + B*E(x) + f0*x/beta,
E(x) = exp(beta (x - x_ref)/eps); the 4 constants come from the 2 BCs and the
2 interface conditions.  u, u' are continuous at 0 but u'' JUMPS:
    eps [u''] = (beta_plus - beta_minus) u'(0).

Methods (all use strong Dirichlet BCs  u = (x^2 - 1) * N(x)):
    pinn        mean squared strong residual at random collocation points
    vpinn       classical VPINN loss  sum_m r(u, s_m)^2,  s_m = sin(m pi (x+1)/2)
    rvpinn_fem  RVPINN, M piecewise-linear hats,  loss = R^T G^{-1} R   (eq. 23)
    rvpinn_sp   RVPINN, M sines (V-orthogonal),   loss = 4/(eps pi^2) sum r_m^2/m^2  (eq. 54)

Networks:
    standard    5 x 25 tanh MLP (paper Sec. 5.3) -> C-infinity, cannot represent the u'' jump
    enriched    u = (x^2-1) [N_0(x) + max(x,0)^2 N_1(x)]  -> u, u' continuous, u'' may jump at 0

All integrals use composite Gauss-Legendre quadrature whose breakpoints include
x = 0 (and the FE nodes), so the discontinuity in beta is integrated exactly.

Usage
    python btp_interface.py                                  # everything, default data
    python btp_interface.py --eps 0.05 --beta_minus 3 --beta_plus 1
    python btp_interface.py --methods rvpinn_sp --nets standard --M_spec 50
    python btp_interface.py --case smooth --nets standard     # beta = 0 benchmark (Sec. 5.4)
Outputs (in --outdir): per-run figures, comparison figures, results.csv
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
PI = math.pi


# ==========================================================================
# Exact solution
# ==========================================================================
class Exact:
    def __init__(self, eps, bm, bp, f0):
        self.eps, self.bm, self.bp, self.f0 = eps, bm, bp, f0
        self.sides = [self._side(bm, -1.0, 0.0), self._side(bp, 0.0, 1.0)]
        # unknowns c = [A-, B-, A+, B+]
        L, R = self.sides

        def row(side, x, k, sgn, offset):
            # contribution of side's (A,B) to k-th derivative at x
            r = np.zeros(4)
            r[offset] = sgn * (1.0 if k == 0 else 0.0)
            r[offset + 1] = sgn * side["E"](x, k)
            return r, sgn * side["p"](x, k)

        A = np.zeros((4, 4)); b = np.zeros(4)
        r, p = row(L, -1.0, 0, 1, 0); A[0] = r; b[0] = -p          # u(-1) = 0
        r, p = row(R, 1.0, 0, 1, 2); A[1] = r; b[1] = -p           # u(1) = 0
        for i, k in [(2, 0), (3, 1)]:                               # [u]=0, [u']=0
            r1, p1 = row(L, 0.0, k, 1, 0)
            r2, p2 = row(R, 0.0, k, -1, 2)
            A[i] = r1 + r2; b[i] = -(p1 + p2)
        self.c = np.linalg.solve(A, b)

    def _side(self, beta, a, b):
        eps, f0 = self.eps, self.f0
        if abs(beta) > 1e-14:
            xr = b if beta > 0 else a
            lam = beta / eps

            def E(x, k, mod=np):
                arg = lam * (x - xr)
                arg = mod.clip(arg, None, 0.0) if mod is np else torch.clamp(arg, max=0.0)
                return lam**k * mod.exp(arg)

            def p(x, k, mod=np):
                return [f0 * x / beta, f0 / beta + 0 * x, 0 * x][k]
        else:
            def E(x, k, mod=np):
                return [x, 1.0 + 0 * x, 0 * x][k]

            def p(x, k, mod=np):
                return [-f0 * x**2 / (2 * eps), -f0 * x / eps, -f0 / eps + 0 * x][k]
        return {"E": E, "p": p}

    def deriv(self, x, k):
        """k-th derivative (k = 0,1,2) of the exact solution, torch tensor in/out."""
        out = []
        for s, (side, off) in enumerate(zip(self.sides, (0, 2))):
            A, B = self.c[off], self.c[off + 1]
            v = (A if k == 0 else 0.0) + B * side["E"](x, k, torch) + side["p"](x, k, torch)
            out.append(v)
        return torch.where(x < 0, out[0], out[1])

    def beta(self, x):
        return torch.where(x < 0, torch.full_like(x, self.bm), torch.full_like(x, self.bp))

    def f(self, x):
        return torch.full_like(x, self.f0)

    @property
    def desc(self):
        return rf"$\varepsilon={self.eps}$, $\beta_-={self.bm}$, $\beta_+={self.bp}$, $f={self.f0}$"


class SmoothExact:
    """beta = 0 benchmark of paper Sec. 5.4: -eps u'' = f, u = x sin(pi(x+1))."""

    def __init__(self, eps=1.0):
        self.eps, self.bm, self.bp = eps, 0.0, 0.0

    def deriv(self, x, k):
        s, c = torch.sin(PI * (x + 1)), torch.cos(PI * (x + 1))
        return [x * s, s + PI * x * c, 2 * PI * c - PI**2 * x * s][k]

    def beta(self, x):
        return torch.zeros_like(x)

    def f(self, x):
        return -self.eps * self.deriv(x, 2)

    desc = r"$\varepsilon=1$, $\beta=0$, $u=x\sin(\pi(x+1))$"


# ==========================================================================
# Quadrature: composite Gauss-Legendre on given breakpoints
# ==========================================================================
def gauss_on_breaks(breaks, n_sub=1, n_gauss=5):
    g, w = np.polynomial.legendre.leggauss(n_gauss)
    pts = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        pts.append(np.linspace(a, b, n_sub + 1))
    nodes = np.unique(np.concatenate(pts))
    h = np.diff(nodes)
    xq = (nodes[:-1, None] + 0.5 * h[:, None] * (g[None, :] + 1)).ravel()
    wq = (0.5 * h[:, None] * w[None, :]).ravel()
    return torch.tensor(xq).view(-1, 1), torch.tensor(wq).view(-1, 1)


def D(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


# ==========================================================================
# Networks (strong BCs)
# ==========================================================================
class Net(nn.Module):
    def __init__(self, enriched=False, width=25, depth=5):
        super().__init__()
        self.enriched = enriched
        layers, d = [], 1
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, 2 if enriched else 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        n = self.body(x)
        if self.enriched:
            n = n[:, :1] + torch.relu(x) ** 2 * n[:, 1:2]
        return (x + 1) * (x - 1) * n


# ==========================================================================
# Losses
# ==========================================================================
class PINNLoss:
    robust = False

    def __init__(self, ex, n_col=1000):
        self.ex, self.n_col = ex, n_col

    def __call__(self, model):
        x = (2 * torch.rand(self.n_col, 1) - 1).requires_grad_(True)
        u = model(x); du = D(u, x); d2u = D(du, x)
        res = -self.ex.eps * d2u + self.ex.beta(x) * du - self.ex.f(x)
        return torch.mean(res**2)


class WeakLoss:
    """Weak residual r(u, v_m) = (f,v_m) - eps(u',v_m') - (beta u', v_m) for a test set."""

    def __init__(self, ex, xq, wq, V, dV, Ginv=None):
        self.ex, self.xq, self.wq, self.V, self.dV = ex, xq, wq, V, dV
        self.F = (wq * ex.f(xq) * V).sum(0)
        self.beta_q = ex.beta(xq)
        self.Ginv = Ginv            # None -> classical VPINN loss
        self.robust = Ginv is not None

    def residual(self, model):
        x = self.xq.clone().requires_grad_(True)
        du = D(model(x), x)
        a = self.ex.eps * (self.wq * du * self.dV).sum(0) + (self.wq * self.beta_q * du * self.V).sum(0)
        return self.F - a

    def __call__(self, model):
        R = self.residual(model)
        if self.Ginv is None:
            return torch.sum(R**2)
        if self.Ginv.dim() == 1:
            return torch.sum(self.Ginv * R**2)
        return R @ self.Ginv @ R


def spectral_loss(ex, M, robust, n_sub=400):
    xq, wq = gauss_on_breaks([-1.0, 0.0, 1.0], n_sub=n_sub)
    m = torch.arange(1, M + 1, dtype=torch.float64).view(1, -1)
    arg = m * PI * (xq + 1) / 2
    V, dV = torch.sin(arg), (m * PI / 2) * torch.cos(arg)
    Ginv = 4.0 / (ex.eps * PI**2 * m.view(-1) ** 2) if robust else None
    return WeakLoss(ex, xq, wq, V, dV, Ginv)


def fem_loss(ex, M):
    n_el = M + 1
    h = 2.0 / n_el
    nodes = np.linspace(-1, 1, n_el + 1)
    breaks = np.unique(np.concatenate([nodes, [0.0]]))      # split element containing x=0
    xq, wq = gauss_on_breaks(breaks, n_sub=1, n_gauss=5)
    xi = torch.tensor(nodes[1:-1]).view(1, -1)
    dist = (xq - xi) / h
    V = torch.clamp(1 - dist.abs(), min=0.0)
    dV = torch.where(dist.abs() < 1, -torch.sign(dist) / h, torch.zeros_like(dist))
    G = (ex.eps / h) * (2 * torch.eye(M) - torch.diag(torch.ones(M - 1), 1)
                        - torch.diag(torch.ones(M - 1), -1))
    return WeakLoss(ex, xq, wq, V, dV, torch.linalg.inv(G))


# ==========================================================================
# Error measures
# ==========================================================================
class Metrics:
    def __init__(self, ex, n_sub=1000):
        self.ex = ex
        self.x, self.w = gauss_on_breaks([-1.0, 0.0, 1.0], n_sub=n_sub)
        self.u, self.du, self.d2u = (ex.deriv(self.x, k) for k in range(3))
        self.nH1 = torch.sqrt(ex.eps * torch.sum(self.w * self.du**2)).item()
        self.nL2 = torch.sqrt(torch.sum(self.w * self.u**2)).item()
        self.nD2 = torch.sqrt(torch.sum(self.w * self.d2u**2)).item()

    def fast(self, model):
        x = self.x.clone().requires_grad_(True)
        u = model(x); du = D(u, x)
        eH1 = torch.sqrt(self.ex.eps * torch.sum(self.w * (du - self.du) ** 2)).item() / self.nH1
        eL2 = torch.sqrt(torch.sum(self.w * (u - self.u) ** 2)).item() / self.nL2
        return eH1, eL2

    def full(self, model):
        x = self.x.clone().requires_grad_(True)
        u = model(x); du = D(u, x); d2u = D(du, x)
        res = -self.ex.eps * d2u + self.ex.beta(x) * du - self.ex.f(x)
        w = self.w
        out = dict(
            H1=torch.sqrt(self.ex.eps * torch.sum(w * (du - self.du) ** 2)).item() / self.nH1,
            L2=torch.sqrt(torch.sum(w * (u - self.u) ** 2)).item() / self.nL2,
            D2=torch.sqrt(torch.sum(w * (d2u - self.d2u) ** 2)).item() / self.nD2,
            res_rms=torch.sqrt(torch.sum(w * res**2) / 2).item(),
        )
        # where is the strong residual concentrated?
        xs = x.detach()
        for name, mask in [("res_bdry", (xs.abs() > 0.9)),
                           ("res_intf", (xs.abs() < 0.1)),
                           ("res_bulk", (xs.abs() >= 0.1) & (xs.abs() <= 0.9))]:
            out[name] = torch.sqrt(torch.sum(w[mask] * res[mask] ** 2) / torch.sum(w[mask])).item()
        return out, dict(x=xs.numpy().ravel(), u=u.detach().numpy().ravel(),
                         d2u=d2u.detach().numpy().ravel(), res=res.detach().numpy().ravel())


# ==========================================================================
# Training
# ==========================================================================
def make_loss(method, ex, args):
    if method == "pinn":
        return PINNLoss(ex, args.n_col)
    if method == "vpinn":
        return spectral_loss(ex, args.M_spec, robust=False)
    if method == "rvpinn_sp":
        return spectral_loss(ex, args.M_spec, robust=True)
    if method == "rvpinn_fem":
        return fem_loss(ex, args.M_fem)
    raise ValueError(method)


def train(method, net, ex, metrics, args, eval_every=10, log_every=1000):
    torch.manual_seed(args.seed)
    model = Net(enriched=(net == "enriched"))
    loss_fn = make_loss(method, ex, args)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist = dict(it=[], loss=[], H1=[], L2=[])
    best_loss, best_state = float("inf"), None
    t0 = time.time()
    early = None
    for ep in range(1, args.epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model)
        lv = loss.item()
        # Model selection uses the LOSS only (no access to the exact solution).
        # The state is stored BEFORE the optimiser step, i.e. the parameters that produced lv.
        if lv < best_loss:
            best_loss, best_state = lv, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % eval_every == 0 or ep == 1:
            eH1, eL2 = metrics.fast(model)          # same parameters as lv
            hist["it"].append(ep); hist["loss"].append(lv); hist["H1"].append(eH1); hist["L2"].append(eL2)
            if ep % log_every == 0 or ep == 1:
                print(f"  [{method:10s}|{net:8s}] ep {ep:5d}  loss {lv:.3e}  "
                      f"relH1 {eH1:.3e}  relL2 {eL2:.3e}", flush=True)
        loss.backward()
        opt.step()
        if args.report_at and ep == args.report_at and ep < args.epochs:
            # metrics of the best-loss model seen in the first `report_at` iterations
            tmp = Net(enriched=(net == "enriched"))
            tmp.load_state_dict(best_state)
            early, _ = metrics.full(tmp)
            early["best_loss"] = best_loss
            print(f"  [{method:10s}|{net:8s}] best model after {ep} its: relH1 {early['H1']:.3e} "
                  f"relL2 {early['L2']:.3e} rel u'' {early['D2']:.3e}", flush=True)
    model.load_state_dict(best_state)
    res, fields = metrics.full(model)
    res.update(time=time.time() - t0, best_loss=best_loss, early=early)
    print(f"  [{method:10s}|{net:8s}] done {res['time']:.0f}s | relH1 {res['H1']:.3e} "
          f"relL2 {res['L2']:.3e} rel u'' {res['D2']:.3e} res_rms {res['res_rms']:.3e}\n", flush=True)
    return model, hist, res, fields, loss_fn.robust


# ==========================================================================
# Plotting
# ==========================================================================
LABEL = {"pinn": "PINN", "vpinn": "VPINN (sines)",
         "rvpinn_fem": "RVPINN (FE hats)", "rvpinn_sp": "RVPINN (spectral)"}


def plot_run(tag, method, net, hist, fields, metrics, robust, ex, outdir):
    x = fields["x"]
    ue, d2e = metrics.u.numpy().ravel(), metrics.d2u.numpy().ravel()
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    ax[0].plot(x, ue, "r-", lw=2, label="Exact")
    ax[0].plot(x, fields["u"], "b--", lw=1.5, label=LABEL[method])
    ax[0].axvline(0, color="gray", lw=0.5)
    ax[0].set(xlabel="x", ylabel="u", title="(a) Solution"); ax[0].legend()

    ax[1].semilogy(x, np.abs(fields["u"] - ue) + 1e-16, "k-", lw=1)
    ax[1].set(xlabel="x", ylabel=r"$|u-u_\theta|$", title="(b) Pointwise error")

    ax[2].semilogy(x, np.abs(fields["d2u"] - d2e) + 1e-16, "m-", lw=1, label=r"$|u''-u_\theta''|$")
    ax[2].semilogy(x, np.abs(fields["res"]) + 1e-16, "c-", lw=1, alpha=0.8, label="|strong residual|")
    ax[2].axvline(0, color="gray", lw=0.5)
    ax[2].set(xlabel="x", title="(c) 2nd derivative error & strong residual"); ax[2].legend(fontsize=8)

    it = np.array(hist["it"]); L = np.array(hist["loss"])
    ax[3].loglog(it, hist["H1"], "g-", label=r"$\|u-u_\theta\|_U/\|u\|_U$")
    if robust:
        ax[3].loglog(it, np.sqrt(L) / metrics.nH1, "--", color="orange",
                     label=r"$\sqrt{\mathcal{L}}/\|u\|_U$ (estimate)")
    else:
        ax[3].loglog(it, L, "--", color="orange", label="loss")
    ax[3].set(xlabel="Iterations", title="(d) Training history"); ax[3].legend(fontsize=8)

    fig.suptitle(rf"{LABEL[method]}, {net} net:  $-\varepsilon u''+\beta(x)u'=f$,  "
                 + ex.desc)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{tag}.png"), dpi=130)
    plt.close(fig)


def plot_compare(runs, net, metrics, ex, outdir):
    sel = {k: v for k, v in runs.items() if k[1] == net}
    if not sel:
        return
    x = metrics.x.numpy().ravel()
    ue, d2e = metrics.u.numpy().ravel(), metrics.d2u.numpy().ravel()
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(x, ue, "k-", lw=2.5, label="Exact")
    for (m, _), r in sel.items():
        ax[0, 0].plot(x, r["fields"]["u"], "--", lw=1.3, label=LABEL[m])
        ax[0, 1].semilogy(x, np.abs(r["fields"]["u"] - ue) + 1e-16, lw=1, label=LABEL[m])
        ax[1, 0].semilogy(x, np.abs(r["fields"]["d2u"] - d2e) + 1e-16, lw=1, label=LABEL[m])
        ax[1, 1].loglog(r["hist"]["it"], r["hist"]["H1"], lw=1, label=LABEL[m])
    ax[0, 0].set(xlabel="x", ylabel="u", title="Solutions"); ax[0, 0].legend()
    ax[0, 1].set(xlabel="x", ylabel=r"$|u-u_\theta|$", title="Pointwise error")
    ax[1, 0].set(xlabel="x", ylabel=r"$|u''-u_\theta''|$", title="Second-derivative error")
    ax[1, 1].set(xlabel="Iterations", ylabel="Relative energy-norm error", title="Convergence")
    for a in ax.ravel()[1:]:
        a.legend(fontsize=8)
    for a in ax.ravel()[:3]:
        a.axvline(0, color="gray", lw=0.5)
    fig.suptitle(f"Comparison ({net} network): " + ex.desc)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"comparison_{net}.png"), dpi=130)
    plt.close(fig)


def plot_exact(ex, metrics, outdir):
    x = metrics.x.numpy().ravel()
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a, k, name in zip(ax, range(3), ["u", "u'", "u''"]):
        a.plot(x, ex.deriv(metrics.x, k).numpy().ravel(), "k-")
        a.axvline(0, color="gray", lw=0.5); a.set(xlabel="x", title=f"Exact {name}")
    fig.suptitle("Exact solution: " + ex.desc)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "exact_solution.png"), dpi=130)
    plt.close(fig)


# ==========================================================================
# Main
# ==========================================================================
def self_check(ex):
    x = torch.linspace(-1, 1, 2001).view(-1, 1)
    x = x[x.abs().view(-1) > 1e-9].view(-1, 1)
    res = -ex.eps * ex.deriv(x, 2) + ex.beta(x) * ex.deriv(x, 1) - ex.f(x)
    z = torch.tensor([[-1e-12], [1e-12]])
    u0, du0, d20 = (ex.deriv(z, k).view(-1) for k in range(3))
    bc = ex.deriv(torch.tensor([[-1.0], [1.0]]), 0).view(-1)
    print(f"Exact-solution check: max|PDE residual| = {res.abs().max():.2e}, "
          f"u(+-1) = {bc[0]:.1e}, {bc[1]:.1e}, [u](0) = {(u0[1]-u0[0]).item():.1e}, "
          f"[u'](0) = {(du0[1]-du0[0]).item():.1e}, [u''](0) = {(d20[1]-d20[0]).item():.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="interface", choices=["interface", "smooth"],
                   help="interface: discontinuous beta (default); smooth: beta = 0 benchmark of Sec. 5.4")
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--beta_minus", type=float, default=2.0)
    p.add_argument("--beta_plus", type=float, default=1.0)
    p.add_argument("--f0", type=float, default=1.0, help="constant source term")
    p.add_argument("--methods", default="pinn,vpinn,rvpinn_fem,rvpinn_sp")
    p.add_argument("--nets", default="standard,enriched")
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--M_spec", type=int, default=50, help="# sine test functions")
    p.add_argument("--M_fem", type=int, default=100, help="# FE hat test functions")
    p.add_argument("--n_col", type=int, default=1000, help="# PINN collocation points")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report_at", type=int, default=6000,
                   help="also report the best-loss model after this many iterations (if < epochs)")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ex = Exact(args.eps, args.beta_minus, args.beta_plus, args.f0) if args.case == "interface" \
        else SmoothExact(1.0)
    self_check(ex)
    if args.case == "interface" and args.beta_minus < args.beta_plus:
        print("WARNING: beta_minus < beta_plus -> a(v,v) >= ||v||_V^2 not guaranteed; "
              "RVPINN loss is then not an exact error estimate.")
    metrics = Metrics(ex)
    plot_exact(ex, metrics, args.outdir)

    runs = {}
    for net in args.nets.split(","):
        for m in args.methods.split(","):
            model, hist, res, fields, robust = train(m, net, ex, metrics, args)
            tag = f"{m}_{net}"
            plot_run(tag, m, net, hist, fields, metrics, robust, ex, args.outdir)
            runs[(m, net)] = dict(hist=hist, res=res, fields=fields)
            np.savez(os.path.join(args.outdir, f"{tag}.npz"), nH1=metrics.nH1,
                     **{f"hist_{k}": np.array(v) for k, v in hist.items()},
                     **{f"field_{k}": v for k, v in fields.items()})
        plot_compare(runs, net, metrics, ex, args.outdir)

    cols = ["H1", "L2", "D2", "res_rms", "res_bdry", "res_intf", "res_bulk", "best_loss"]
    rows = []
    for (m, net), r in runs.items():
        if r["res"]["early"] is not None:
            rows.append((m, net, args.report_at, r["res"]["early"]))
        rows.append((m, net, args.epochs, r["res"]))
    with open(os.path.join(args.outdir, "results.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["method", "net", "iterations"] + cols)
        for m, net, it, q in rows:
            wr.writerow([m, net, it] + [f"{q[c]:.4e}" for c in cols])
    print(f"\n{'method':12s} {'net':9s} {'its':>6s} {'rel H1':>9s} {'rel L2':>9s} {'rel u\"':>9s} "
          f"{'res rms':>9s} {'res|x|>.9':>10s} {'res|x|<.1':>10s} {'res bulk':>9s}")
    for m, net, it, q in rows:
        print(f"{m:12s} {net:9s} {it:6d} {q['H1']:9.2e} {q['L2']:9.2e} {q['D2']:9.2e} {q['res_rms']:9.2e} "
              f"{q['res_bdry']:10.2e} {q['res_intf']:10.2e} {q['res_bulk']:9.2e}")


if __name__ == "__main__":
    main()

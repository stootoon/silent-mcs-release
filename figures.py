"""One builder per model figure.

Each function takes a computed ``model.SilentMcs`` instance and returns a
Matplotlib ``Figure``; ``driver.py`` calls them and saves the results. These are
the model-data panels only -- no experimental comparison panels.
"""
import numpy as np
import matplotlib.pyplot as plt

import plotting as P


def _label(ax, s):
    ax.text(-0.04, 1.0, s, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom", ha="right")


def responses(d, num_odours=10):
    """Baseline and odour-evoked voltages for the last ``num_odours``
    presentations, flattened over the cells (the cell-odour pairs)."""
    bl = d.bl_resps[-num_odours:, :].flatten()
    od = d.od_resps[-num_odours:, :].flatten()
    return bl, od


# --- predicted slopes (closed form, no fitting) ----------------------------
def _bg_slope(d):
    """Predicted slope of the pre-odour baseline against the common input:
        mu_pre = -e^{-n_air |log a|} * ka*o_bar*(g c + b) / (c (g S + |a|^2)) * Ac,
        S = sigma^2_eff + ka*o_bar.
    The panel divides both axes by sigma^2_eff, so the plotted slope is mu_pre/Ac
    (sigma^2_eff cancels). The common odour is one-hot (bg = 0.9 A x0), so its
    concentration c = 0.9 and its unit-norm receptive field gives |a|^2 = 1."""
    c = 0.9
    a2 = float(np.sum(d.bg_odour ** 2)) / c ** 2       # ||a||^2 (=1 for unit-norm columns)
    kaob = d.ka * d.o_bar
    S = d.sd2_eff + kaob
    nbar = (d.min_air_trials + d.max_air_trials) / 2.0
    rho = np.exp(-nbar * abs(np.log(d.al)))            # pre-odour baseline decay
    return -rho * kaob * (d.ga * c + d.be) / (c * (d.ga * S + a2))


def _resp_slope(d):
    """Predicted slope of the odour-averaged response against the pre-odour baseline
    voltage:
        <lambda>_od = -sigma^2_eff / (ka * o_bar * rho) * lambda_pre,
        rho = e^{-n_air |log alpha|}  (the pre-odour baseline decay).
    In air the glomerular input is ~0 and the granule cells are sub-threshold, so
    lambda_pre = mu_pre / sigma^2_eff; with mu_pre = rho*mu and mu = -ka*o_bar*<lambda>_od.
    Both scatter axes are raw voltages (no sigma^2_eff normalisation), so this slope is
    plotted directly. Depends only on the noise variances, learning gain and odour
    statistics -- not on the granule-cell parameters beta, gamma."""
    nbar = (d.min_air_trials + d.max_air_trials) / 2.0
    rho = np.exp(-nbar * abs(np.log(d.al)))
    return -d.sd2_eff / (d.ka * d.o_bar * rho)


# --- figures ---------------------------------------------------------------
def fig_bg(d, num_odours=10):
    """Common glomerular input (/sigma^2_eff) vs. mitral-cell baseline for the
    default model (A) and five one-parameter manipulations (B-F). The line in each
    panel is the slope predicted by ``_bg_slope``, with no fitting; all panels share
    the same axes for comparison."""
    from model import SilentMcs
    base = dict(M=d.M, N=d.N, n_odours=d.n_odours, k=d.k, sd=d.sd, eta=d.eta,
                al=d.al, ga=d.ga, be=d.be, tau_la=d.tau_la, tau_v=d.tau_v,
                t_whiff=d.t_whiff, n_trials=d.n_trials, first_trial=d.first_trial,
                min_air_trials=d.min_air_trials, max_air_trials=d.max_air_trials, seed=d.seed)
    panels = [
        ("default",                     {}),
        (r"$\beta=1.5$",                dict(be=1.5)),
        (r"$\alpha=0.96$",              dict(al=0.96)),
        (r"$\bar n_\mathrm{air}=5$",    dict(min_air_trials=5, max_air_trials=5)),
        (r"$\eta=0.05$",                dict(eta=0.05)),
        (r"$\sigma=0.18$",              dict(sd=0.18)),
    ]
    results = []
    for lab, over in panels:
        dm = d if not over else SilentMcs(**{**base, **over}).compute(dt=d.dt,
                                                                      randomize_trials=False)
        bl = dm.bl_resps[-num_odours:, :]
        bg = np.outer(np.ones(bl.shape[0]), dm.bg_odour).flatten() / dm.sd2_eff
        y = bl.flatten()
        slope = _bg_slope(dm)
        results.append((lab, bg, y, slope))
        print(f"  {lab:18s} pred slope={slope:+.3f}  emp={np.polyfit(bg, y, 1)[0]:+.3f}")
    XL = 1.05 * max(np.abs(bg).max() for _, bg, _, _ in results)
    YL = 1.05 * max(np.abs(y).max() for _, _, y, _ in results)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)
    for ax, (lab, bg, y, slope), letter in zip(axes.flat, results, "ABCDEF"):
        ax.plot([-XL, XL], [0, 0], "-", color="gray", lw=0.8)
        ax.plot([0, 0], [-YL, YL], "-", color="gray", lw=0.8)
        ax.plot(bg, y, "o", color="dodgerblue", markersize=2.5)
        ax.plot([-XL, XL], [-XL * slope, XL * slope], "-", color="crimson", lw=1.8,
                label="prediction")
        ax.set_xlim(-XL, XL); ax.set_ylim(-YL, YL)
        ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-2, 0, 2])
        P.spines_off(ax); ax.tick_params(labelsize=11)
        ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="left")
        ax.text(0.04, 0.05, lab, transform=ax.transAxes, fontsize=12, va="bottom", ha="left")
        ax.text(0.97, 0.90, f"slope\n{slope:.2f}", transform=ax.transAxes, fontsize=9,
                color="crimson", va="top", ha="right", linespacing=0.9)
    axes.flat[0].legend(loc="lower left", bbox_to_anchor=(0.03, 0.13), fontsize=9,
                        frameon=False, handlelength=1.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Baseline activity (V)", fontsize=12)
    for ax in axes[1, :]:
        ax.set_xlabel(r"Common input $/\;\sigma^2_\mathrm{eff}$ (V)", fontsize=12)
    fig.tight_layout()
    return fig


def fig_resp_dists(d):
    """Model odour-response distribution (A) and odour response vs. baseline (B).
    The red line in B is the slope predicted by ``_resp_slope`` (no fitting); open
    circles are the per-cell means, which it passes through."""
    bl, od = responses(d)
    fig = plt.figure(figsize=(7.5, 3.2))
    axA = fig.add_axes([0.09, 0.17, 0.33, 0.74])         # model histogram
    axA.hist(od, np.linspace(-2, 2, 31), color="w", ec="C0"); P.spines_off(axA)
    axA.set_yticks([p / 100. * len(od) for p in (0, 5, 10)])
    axA.set_yticklabels(["0%", "5%", "10%"])
    axA.set_xlabel("Odour-evoked response (V)"); axA.set_ylabel("# observed", labelpad=-2)
    _label(axA, "A")
    axB = fig.add_axes([0.57, 0.17, 0.38, 0.74])         # response vs. baseline
    axB.plot(bl, od, "o", markersize=3, zorder=1); P.spines_off(axB)
    blc, odc = d.bl_resps[-10:, :].mean(0), d.od_resps[-10:, :].mean(0)  # per-cell means
    axB.plot(blc, odc, "o", mfc="none", mec="k", ms=5, zorder=2)
    s = _resp_slope(d)                                   # predicted slope, no fitting
    xl = axB.get_xlim()
    axB.plot(list(xl), [s * xl[0], s * xl[1]], "-", color="crimson", lw=1.8, zorder=3)
    axB.set_xlim(xl)
    axB.text(0.96, 0.96, f"slope\n{s:.2f}", transform=axB.transAxes, fontsize=9,
             color="crimson", va="top", ha="right", linespacing=0.9)
    axB.set_xlabel("Baseline (V)"); axB.set_ylabel("Odour-evoked response (V)")
    _label(axB, "B")
    print(f"  rho(baseline, odour-evoked) = {np.corrcoef(bl, od)[0, 1]:+.3f}")
    return fig


def fig_whiskers(d):
    """Model odour responses grouped by pre-odour (baseline) voltage."""
    bl, od = responses(d)
    fig = plt.figure(figsize=(2.8, 2.9))
    ax = fig.add_axes([0.28, 0.16, 0.66, 0.78])
    P.plot_resp_voltage_by_pre_voltage(bl, od, ax=ax)
    return fig


def whiskers_stats(d, num_odours=10):
    """Statistics for the ``fig_whiskers`` pre-odour/odour-response split. Compares
    the pooled cell-odour pairs with a per-cell analysis that averages each cell over
    its presentations first (addressing the cells x presentations pseudoreplication).
    Needs scipy (lazy import)."""
    from scipy.stats import mannwhitneyu, spearmanr
    bl = d.bl_resps[-num_odours:, :]          # (presentations, cells)
    od = d.od_resps[-num_odours:, :]
    blf, odf = bl.flatten(), od.flatten()     # pooled cell-odour pairs
    th = np.median(blf)
    pair_p = mannwhitneyu(odf[blf < th], odf[blf >= th], alternative="two-sided").pvalue
    bl_c, od_c = bl.mean(axis=0), od.mean(axis=0)   # per-cell means
    thc = np.median(bl_c)
    lo, hi = od_c[bl_c < thc], od_c[bl_c >= thc]
    cell_p = mannwhitneyu(lo, hi, alternative="two-sided").pvalue
    rho, rho_p = spearmanr(bl_c, od_c)
    return dict(n_pairs=int(blf.size), pair_p=float(pair_p),
                n_cells=int(bl_c.size), n_lo=int(lo.size), n_hi=int(hi.size),
                cell_p=float(cell_p), spearman_rho=float(rho), spearman_p=float(rho_p))


def fig_cdfs(d):
    """Model cumulative distributions of odour responses, grouped by baseline."""
    bl, od = responses(d)
    fig = plt.figure(figsize=(3.6, 3.0))
    ax = fig.add_axes([0.15, 0.17, 0.82, 0.78])
    P.plot_cdf(bl, od, ax=ax)
    ax.spines["left"].set_bounds(0, 1)               # y spans 0%-100% (not down to spine)
    return fig


def fig_spikes(d, rate_mul=0.062, seed=1):
    """Model single-cell spiking of four cells ordered by baseline. A wider-gap run
    (odour roughly every 12 s) gives a long baseline with a brief odour. Cells show
    the baseline gradient: a *silent* low-baseline cell that responds, plus three
    higher-baseline cells (decreasing baseline spiking) that are suppressed."""
    from model import SilentMcs
    ds = SilentMcs(M=d.M, N=d.N, n_odours=d.n_odours, k=d.k, sd=d.sd, eta=d.eta,
                   al=d.al, ga=d.ga, be=d.be, tau_la=d.tau_la, tau_v=d.tau_v,
                   t_whiff=d.t_whiff, n_trials=d.n_trials, first_trial=d.first_trial,
                   min_air_trials=11, max_air_trials=11, seed=d.seed
                   ).compute(dt=d.dt, randomize_trials=False)
    T_RNG = (-5.5, -1.0, 6.5)                         # 12 s window; odour [0,1] = 8.3%

    def windows(ws):
        t0 = ds.t_full[ds.stim_inds[ws]]
        bl = ds.la[(ds.t_full > t0 + T_RNG[0]) & (ds.t_full < t0 + T_RNG[1])].mean(0)
        od = ds.la[(ds.t_full > t0) & (ds.t_full < t0 + 1.0)].mean(0)
        return t0, bl, od

    best = None
    for ws in range(-1, -min(12, len(ds.stim_inds)) - 1, -1):
        t0, bl, od = windows(ws)
        if t0 + T_RNG[0] < ds.t_full[0] or t0 + T_RNG[2] > ds.t_full[-1]:
            continue
        resp = int(np.argmax(od - bl))
        score = od[resp] - bl[resp]
        if best is None or score > best[0]:
            best = (score, ws, resp, bl, od)
    _, which_stim, responder, bl, od = best
    t0 = ds.t_full[ds.stim_inds[which_stim]]

    # Pick four cells with a spread of baselines: a silent strong responder, a mild
    # responder, and two higher-baseline cells that are suppressed by the odour.
    def spk(c, a, b):
        m = (ds.t_full > t0 + a) & (ds.t_full < t0 + b)
        return float(np.clip(ds.la[m, c] * rate_mul, 0, 1).sum())
    M = len(od)
    pre  = np.array([spk(c, T_RNG[0], 0.0) for c in range(M)])   # ~5.5 s baseline
    odc  = np.array([spk(c, 0.0, 1.0)      for c in range(M)])   # 1 s odour
    post = np.array([spk(c, 1.0, T_RNG[2]) for c in range(M)])   # ~5.5 s after
    respL = od - bl
    strong = min([c for c in range(M) if pre[c] < 1 and respL[c] > 0.8]
                 or [int(np.argmax(respL))], key=lambda c: bl[c])
    mild = min([c for c in range(M) if c != strong and respL[c] > 0.2
                and bl[c] < 0.35 and odc[c] >= 2] or [c for c in range(M) if c != strong],
               key=lambda c: abs(pre[c] - 1) + abs(odc[c] - 4))
    used = {strong, mild}
    row1 = max((c for c in range(M) if c not in used), key=lambda c: pre[c])
    used.add(row1)
    row2 = min([c for c in range(M) if c not in used and respL[c] < 0 and 3 <= pre[c] <= 9]
               or [c for c in range(M) if c not in used],
               key=lambda c: abs(pre[c] - 6) + abs(post[c] - 5))
    cells = sorted([strong, mild, row1, row2], key=lambda c: bl[c])

    fig = plt.figure(figsize=(3.8, 4.8))
    ax = fig.add_axes([0.10, 0.06, 0.88, 0.90])
    P.plot_spikes(ds.t_full, ds.la, ds.which_pat, ds.stim_inds, which_stim, cells=cells,
                  t_rng=T_RNG, rate_mul=rate_mul, seed=seed, spike_h=0.55,
                  set_ylim=True, ax=ax)
    ax.text(0.5, -0.02, "Odor", transform=ax.transAxes, ha="center", va="top", fontsize=11)
    return fig


def _dynamics(d=None):
    """Early/late granule- and mitral-cell dynamics. Runs its own four-odour model,
    so it ignores the shared ``d``. Imported lazily to avoid a driver import cycle."""
    from dynamics import fig_dynamics
    return fig_dynamics()


# name -> (builder, output filename)
FIGURES = {
    "bg":         (fig_bg,         "bg.pdf"),
    "resp_dists": (fig_resp_dists, "resp_dists.pdf"),
    "whiskers":   (fig_whiskers,   "whiskers.pdf"),
    "cdfs":       (fig_cdfs,       "cdfs.pdf"),
    "spikes":     (fig_spikes,     "spikes.pdf"),
    "dynamics":   (_dynamics,      "dynamics.pdf"),
}

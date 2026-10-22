"""Plotting primitives for the silent-mitral-cell figures.

These operate on the arrays produced by ``model.SilentMcs`` (mitral-cell
voltages ``la``, granule-cell rates ``x``, the odour label sequence
``which_pat``, etc.) and are model-agnostic. Depends only on NumPy and
Matplotlib.
"""
from numpy import *                     # noqa: F401,F403  (scientific-plotting idiom)
from matplotlib.pyplot import *         # noqa: F401,F403
from matplotlib import cm
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.transforms import TransformedBbox
from cycler import cycler


# --- odour patterns (kept here so plotting is self-contained) --------------
def get_x_true(N, P):
    spacing = N / (P + 1)
    return array([arange(N) == int(spacing * (i + 1)) for i in range(P)])


# --- colours ---------------------------------------------------------------
_cols = concatenate([reshape([0, 0, 0, 1], (1, 4)),
                     linspace([0, 0, 0.85, 1], [0.95, 0.95, 0, 1], 201),
                     linspace([0.95, 0.95, 0, 1], [1.0, 0.35, 0, 1], 201)], axis=0)
new_cmap = ListedColormap(_cols)


def set_alpha(c, a):
    rgb = c[:3] if isinstance(c, (list, tuple)) else to_rgb(c)
    return list(rgb) + [a]


def spines_off(ax=None, which=("top", "right")):
    ax = gca() if ax is None else ax
    for w in which:
        ax.spines[w].set_visible(False)
    return ax


def pat_col(i, P):
    if i == 0:
        return [0, 0, 0, 0.5]                       # common odour: gray
    if i == (P - 1):
        return [1, 0, 0.5, 1]
    return cm.rainbow((i - 1) / (P - 1))


# --- dynamics traces (odour / granule / mitral) ----------------------------
def plot_mcs(t, la, which_la, lw=2):
    for w in which_la:
        plot(t, la[:, w], color=cm.cool(0.1 + 0.8 * float(w) / la.shape[1]),
             label=str(w), lw=lw)


def plot_gcs(t, x, P, tol=1e-2, lw_active=2):
    N = x.shape[1]
    x_true = get_x_true(N, P)
    ind_true = where(sum(x_true, axis=0) > 1e-2)[0]
    ind_active = where(max(x, axis=0) > tol)[0]
    active_true = list(set(ind_active).intersection(ind_true))
    active_other = list(set(ind_active) - set(active_true))
    plot_order = active_other + active_true
    col_cycle = cycler(color=[[0, 0, 0, 1]] * len(active_other) +
                       [pat_col(list(ind_true).index(i), P) for i in active_true])
    lw_cycle = cycler(lw=[1] * len(active_other) +
                      [1 if i == ind_true[0] else lw_active for i in active_true])
    gca().set_prop_cycle(col_cycle + lw_cycle)
    gca().plot(t, x[:, list(plot_order)])


def plot_all(t, pats, la, x, which_la, t_rng, figsz=(9, 5), gc_yl=None, mc_yl=None,
             mc_lw=1, gc_lw=1, ax=None, plot_every=1):
    """One column of (odour band, granule rates, mitral voltages) over t_rng."""
    t, pats, la, x = t[::plot_every], pats[::plot_every], la[::plot_every], x[::plot_every]
    ind_t = where((t >= t_rng[0]) * (t <= t_rng[-1]))[0]
    bl_mean = mean(la[pats < 0, :], axis=0)
    la_order = argsort(bl_mean)
    P = int(max(pats)) + 1

    if ax is None:
        figure(figsize=figsz)
    ax_od = subplot(5, 1, 1) if ax is None else ax[0]
    sca(ax_od)
    for p in range(P):
        fill_between(t[ind_t], -0.5 + 0 * t[ind_t], pats[ind_t] == p, color=pat_col(p, P))
    ylim(0, 1); spines_off(); gca().set_xticklabels([]); ylabel("Odour")
    gca().set_yticks([0, 1]); xlim(t[ind_t[0]], t[ind_t[-1]])

    ax_gc = subplot(5, 1, (2, 3)) if ax is None else ax[1]
    sca(ax_gc)
    plot_gcs(t[ind_t], x[ind_t], P, lw_active=gc_lw)
    spines_off(); ylabel("GC firing rate"); gca().set_xticklabels([])
    xlim(t[ind_t[0]], t[ind_t[-1]])
    gc_yl and gca().set_ylim(gc_yl)

    ax_mc = subplot(5, 1, (4, 5)) if ax is None else ax[2]
    sca(ax_mc)
    plot_mcs(t[ind_t], la[ind_t][:, la_order], which_la, lw=mc_lw)
    spines_off(); ylabel("MC voltage"); xlim(t[ind_t[0]], t[ind_t[-1]])
    mc_yl and gca().set_ylim(mc_yl)
    xlabel("Time (sec.)")
    return ax_od, ax_gc, ax_mc


# --- response-distribution panels ------------------------------------------
def plot_bl_vs_od(bl_resps, od_resps, num_odours=10, fig_size=(8, 3.5), pcs=(0, 5, 10), ax=None):
    bl_mean = bl_resps[-num_odours:, :].flatten()
    od_mean = od_resps[-num_odours:, :].flatten()
    if ax is None:
        figure(figsize=fig_size)
    subplot(2, 2, 3)
    hist(od_mean, linspace(-2, 2, 31), color="w", ec="C0"); spines_off()
    gca().set_yticks([pci / 100. * len(od_mean) for pci in pcs])
    gca().set_yticklabels([f"{pci}%" for pci in pcs])
    xlabel("Odour-evoked response (a.u.)"); ylabel("# observed", labelpad=-2)
    subplot(2, 2, (2, 4)); plot(bl_mean, od_mean, "o", markersize=3)
    xlabel("Baseline (a.u.)"); ylabel("Odour-evoked response (a.u.)")
    ylim(-2, 2); gca().set_yticks([-2, -1, 0, 1, 2]); spines_off()
    tight_layout(w_pad=0, h_pad=0)
    print(f"  rho(baseline, odour-evoked) = {corrcoef(bl_mean, od_mean)[0, 1]:+.3f}")
    return bl_mean, od_mean


def plot_resp_voltage_by_pre_voltage(bl_mean, od_mean, th=None, fig_size=(1.5, 2.5),
                                     lab_font_size=12, ax=None, seed=1):
    random.seed(seed)
    th = median(bl_mean) if th is None else th
    i_below, i_above = where(bl_mean < th)[0], where(bl_mean >= th)[0]
    if ax is None:
        figure(figsize=fig_size)
    else:
        sca(ax)
    plot(-1 + random.randn(*i_below.shape) * 3e-1, od_mean[i_below], "o", markersize=1.5, color="lightgray")
    plot(1 + random.randn(*i_above.shape) * 3e-1, od_mean[i_above], "o", markersize=1.5, color="lightgray")
    pc_below, pc_above = percentile(od_mean[i_below], [25, 75]), percentile(od_mean[i_above], [25, 75])
    xlim(-2, 2); gca().set_xticks([-1, 1])
    gca().set_xticklabels([f"<${th:1.1f}$", f"≥${th:1.1f}$"], fontsize=6)
    plot(xlim(), [0, 0], "--", linewidth=1, color="lightgray")
    plot([-1, -1], pc_below, "k-", linewidth=1)
    [plot([-1.1, -0.9], [p] * 2, "k-", linewidth=1) for p in pc_below]
    plot(-1, mean(od_mean[i_below]), "wo", markeredgecolor="k", markersize=4)
    plot([1, 1], pc_above, "k-", linewidth=1)
    [plot([0.9, 1.1], [p] * 2, "k-", linewidth=1) for p in pc_above]
    plot(1, mean(od_mean[i_above]), "wo", markeredgecolor="k", markersize=4)
    # significance bar + stars between the two groups (as in the original figure)
    yl = gca().get_ylim()
    yb = yl[1] * 0.96
    plot([-1, 1], [yb, yb], "k-", linewidth=1)
    text(0, yb, "***", ha="center", va="bottom", fontsize=11)
    ylim(yl[0], yb * 1.18)
    spines_off(which=["top", "bottom", "right"])
    xlabel("Pre-odour voltage", fontsize=lab_font_size)
    ylabel("Odour response (a.u.)", fontsize=lab_font_size)
    gca().xaxis.set_tick_params(width=0)
    gca().tick_params(axis="both", which="major", labelsize=12)


def plot_cdf(bl_mean, od_mean, pcs=None, fig_size=(2.8, 3), ax=None):
    pcs = arange(10, 100, 10) if pcs is None else pcs
    bl_pc = percentile(bl_mean, pcs)
    if ax is None:
        figure(figsize=fig_size)
    else:
        sca(ax)
    iflag = bl_mean >= bl_pc[-1]
    vals = od_mean[iflag]
    plot(sorted(vals), arange(len(vals)) / len(vals), linewidth=2, color="k",
         label=f"$\\geq {bl_pc[-1]:2.1f}$ V")
    for i, blm in enumerate(reversed(bl_pc)):
        frac = mean(bl_mean < blm)
        col = set_alpha(new_cmap(1 - frac), 1)
        label = f"$< {blm:2.1f}$ V" if (not mod(i, 2) and abs(blm - 0.2) > 0.3) else ""
        vals = od_mean[bl_mean < blm].flatten()
        plot(sorted(vals), arange(len(vals)) / len(vals), linewidth=2, color=col, label=label)
    legend(loc=(0.65, 0.05), frameon=False, labelspacing=0, fontsize=10,
           borderpad=0, handlelength=1, handletextpad=0)
    xlim(-2.5, 2.5)
    yt = arange(0.25, 1.1, 0.25); gca().set_yticks(yt)
    gca().set_yticklabels([f"{int(y * 100)}%" if 0 < y < 1 else "" for y in yt])
    gca().set_xticks([-2, -1, 0, 1, 2])
    [plot(xlim(), [y, y], "--", color="gray", linewidth=0.5) for y in [0, 1]]
    plot([0, 0], [-0.06, 1], "--", color="gray", linewidth=0.5)
    ylim(-0.06, 1.0)                               # x-axis spine sits below the 0% line
    xlabel("Odour-evoked response\n(a.u.)", fontsize=12); spines_off()
    gca().tick_params(axis="both", which="major", labelsize=12)


def plot_spikes(t_full, la, which_pat, stim_inds, which_stim, which_sorted_ind=None,
                cells=None, t_rng=(-3, -0.5, 6), rate_mul=10, y_spacing=1, seed=0,
                spike_h=0.9, set_ylim=True, ax=None):
    """Convert MC voltages to a Poisson spike raster for a few cells, ordered by
    baseline. Added noise is purely cosmetic (as stated in the paper).

    Either give ``cells`` (explicit MC indices, bottom-to-top) or
    ``which_sorted_ind`` (indices into the baseline-sorted order)."""
    random.seed(seed)
    stim_start = t_full[stim_inds[which_stim]]
    t_bl = [stim_start + t_rng[0], stim_start + t_rng[1]]
    la_bl = mean(la[(t_full > t_bl[0]) & (t_full < t_bl[1]), :], axis=0)
    if cells is None:
        order = argsort(la_bl)
        cells = [order[i] for i in which_sorted_ind]
    if ax is None:
        figure(figsize=(3, 8))
    else:
        sca(ax)
    t_flag = (t_full > t_bl[0]) * (t_full < stim_start + t_rng[2])
    tt = t_full[t_flag]
    ot = (which_pat[t_flag] > 0).astype(int)                 # odour-present mask
    edges = diff(concatenate([[0], ot, [0]]))                # run boundaries
    runs = list(zip(where(edges == 1)[0], where(edges == -1)[0]))
    for i, cell in enumerate(cells):
        y = y_spacing * i
        r = la[t_flag, cell]
        s = random.rand(sum(t_flag),) <= (r * rate_mul)
        noise = convolve(random.randn(len(s),) * 0.01, ones(60,) / 5, "same")
        for a, b in runs:                                    # one grey bar per odour
            fill_between(tt[a:b], y - 0.12, y + spike_h + 0.12, color="lightgray")
        plot(t_full[t_flag], y + spike_h * s + noise, "k", linewidth=0.5)
        plot(t_full[t_flag], y - 0.05 * s + noise, "k", linewidth=0.5)
        spines_off(which=["top", "left", "right", "bottom"])
        text(t_full[t_flag][0], y, f"{la_bl[cell]:1.2f} ",
             horizontalalignment="right", verticalalignment="center")
    if set_ylim:
        gca().set_ylim(-0.2, y + 1.2)
    gca().set_xticks([]); gca().set_yticks([])


def label_axes(ax_list, labs, dx=0, dy=0, **kwargs):
    fig = gcf()
    fig.canvas.draw()                              # ensure a renderer exists
    renderer = fig.canvas.get_renderer()
    trans = fig.transFigure
    itrans = trans.inverted()
    for i, (ax, lab) in enumerate(zip(ax_list, labs)):
        bb = TransformedBbox(ax.get_tightbbox(renderer), itrans)
        dxi = dx[i] if hasattr(dx, "__len__") else dx
        dyi = dy[i] if hasattr(dy, "__len__") else dy
        fig.text(bb.x0 + dxi, bb.y1 + dyi, lab, transform=trans, **kwargs)

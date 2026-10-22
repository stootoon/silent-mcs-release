"""Dynamics figure: common feature subtraction, early vs late.

The same four odours are shown before CFS has acted (early: the priming phase,
baseline mu = 0) and after it has converged (late), as rows

    Odour / GC (early) / MC (early) / GC (late) / MC (late)

with the four odours in columns. Early: granule cells decode common + unique,
mitral cells respond to both. Late: the learned baseline cancels the common
drive, so the common granule cell falls silent and mitral baselines are
suppressed -- only the unique feature survives.

    python dynamics.py            # -> figs/dynamics.pdf (+ .png)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import plotting as P
from model import SilentMcs, get_x_true
from driver import MODEL_PARAMS, DT

N_ODOURS = 5                       # 1 common (index 0) + 4 unique odours
ODOURS = [1, 2, 3, 4]
T_PRE, T_POST = 0.5, 2.0           # snippet: 0.5 s pre-air, 1 s odour, then post
GC_YL = (-0.1, 1.6)
MC_YL = (-2.6, 2.6)
COMMON_CONC = 0.9                  # concentration of the common component (bg = 0.9 A x0)
UNIQUE_CONC = 1.0                  # concentration of each unique odour (x_true = 1)

# Per-odour colour palette: common feature (tan) + four unique features.
ODOUR_COLOURS = ["#ce1e25", "#b560b7", "#9c2060", "#f2c7ce"]   # unique-feature fills
C_COMMON = "#d2b06a"                                          # common feature (tan)
# Line traces need a darker pink than the pale fill (which vanishes as a thin line).
TRACE_COLOURS = list(ODOUR_COLOURS[:-1]) + ["#e07a9a"]


def _odour_pat_col(i, Pn):
    """Palette override for plot_gcs: common = tan, odours = unique-feature colours."""
    if i == 0:
        return C_COMMON
    return TRACE_COLOURS[(i - 1) % len(TRACE_COLOURS)]


def _xticks(ax, labels, axislabel=None):
    """Ticks at odour onset (0) and offset (1). ``labels`` shows the 0/1 numbers;
    ``axislabel`` (defaults to ``labels``) adds the 'Time (sec.)' caption."""
    if axislabel is None:
        axislabel = labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"] if labels else [])
    ax.tick_params(axis="x", labelsize=8, length=3)
    if axislabel:
        ax.set_xlabel("Time (sec.)", fontsize=9, labelpad=1)


def run_model():
    p = dict(MODEL_PARAMS)
    p.update(n_odours=N_ODOURS, min_air_trials=2, max_air_trials=2, n_trials=90)
    return SilentMcs(**p).compute(dt=DT, randomize_trials=False)


def fig_dynamics(d=None):
    if d is None:
        d = run_model()
    P.pat_col = _odour_pat_col                      # colour granule cells by odour
    t = d.t_full
    sid = [(i, int(d.which_pat[i])) for i in d.stim_inds]
    early, late = {}, {}
    t_conv = d.activation_time + 3 * d.tau_b       # mu is settled (|b| plateaued)
    for i, o in sid:
        if i < d.activation_index and o not in early:
            early[o] = i                           # priming: mu = 0
        if t[i] > t_conv and o not in late:
            late[o] = i                            # first converged presentation

    n_pre, n_post = int(T_PRE / d.dt), int(T_POST / d.dt)
    # granule indices: the common feature and each odour's unique feature
    xt = get_x_true(d.N, N_ODOURS)
    jc = int(np.argmax(xt[0]))
    juni = {o: int(np.argmax(xt[o])) for o in ODOURS}
    # example mitral cells: strong / mild / opposite common drive
    order = np.argsort(d.bg_odour)
    which_la = [order[-1], order[-1 - len(order) // 3], order[0]]
    # shades of green: MC identity is deliberately unrelated to the odour palette
    mc_colours = ["#1b5e20", "#43a047", "#81c784"]

    fig = plt.figure(figsize=(6.6, 5.75))
    # plot rows at 0,2,4,6,8; spacer rows 1,3,5,7 set uneven gaps -- small within an
    # early/late block (GC<->MC, so they read as a pair), larger between blocks
    # (room for the Time (sec.) labels). The odour row (0) is compressed.
    gs = GridSpec(9, 4, height_ratios=[0.4, 0.41, 1, 0.22, 1, 0.48, 1, 0.22, 1],
                  hspace=0.0, wspace=0.16, left=0.115, right=0.98, top=0.96, bottom=0.08)
    block_ax = {}       # left-column axis of each plot row, for EARLY/LATE block labels
    row_lab = ["Odour", "GC", "MC", "GC", "MC"]      # early/late conveyed by block labels

    def snippet(on):
        sl = slice(on - n_pre, on + n_post)
        return t[sl] - t[on], sl

    def ylab(ax, r, j):
        if j == 0:
            ax.set_ylabel(row_lab[r], fontsize=10)

    gc_left = None      # leftmost GC axis (alignment reference)
    mc_left = []        # leftmost MC axes (to move in)
    for j, o in enumerate(ODOURS):
        # row 0: odour during [0, 1] s = common (tan) + unique (colour), stacked so
        # the reader can see the common component is present in every odour.
        ax = fig.add_subplot(gs[0, j])
        ax.fill_between([0, 1], 0, COMMON_CONC, color=C_COMMON)
        ax.fill_between([0, 1], COMMON_CONC, COMMON_CONC + UNIQUE_CONC,
                        color=ODOUR_COLOURS[o - 1])
        ax.set_xlim(-T_PRE, T_POST); ax.set_ylim(0, COMMON_CONC + UNIQUE_CONC + 0.15)
        P.spines_off(ax); ax.set_yticks([0, 2]); _xticks(ax, True, axislabel=False)
        if j: ax.set_yticklabels([])
        ax.set_title(f"odour {o}", fontsize=9); ylab(ax, 0, j)
        if j == 0:
            block_ax[0] = ax

        for src, r_gc, r_mc, lg, lm in [(early, 2, 4, 1, 2), (late, 6, 8, 3, 4)]:
            tr, sl = snippet(src[o])
            # granule cells: always draw the common (tan) and this odour's unique
            # granule -- a fully-subtracted common then shows a line ON zero (GC
            # rates are >= 0), rather than disappearing.
            ax = fig.add_subplot(gs[r_gc, j])
            ax.axhline(0, color="gray", lw=0.4, zorder=1)
            ax.plot(tr, d.x[sl][:, jc], color=C_COMMON, lw=2, zorder=2)          # common
            ax.plot(tr, d.x[sl][:, juni[o]], color=TRACE_COLOURS[o - 1], lw=2, zorder=3)  # unique
            ax.set_xlim(-T_PRE, T_POST); ax.set_ylim(GC_YL)
            P.spines_off(ax); _xticks(ax, True, axislabel=False); ax.set_yticks([0, 1])
            if j: ax.set_yticklabels([])
            if j == 0:
                if gc_left is None:
                    gc_left = ax
                block_ax[r_gc] = ax
            ylab(ax, lg, j)
            # mitral cells
            ax = fig.add_subplot(gs[r_mc, j])
            ax.axhline(0, color="gray", lw=0.5)
            for w, c in zip(which_la, mc_colours):
                ax.plot(tr, d.la[sl][:, w], lw=1.2, color=c)
            ax.set_xlim(-T_PRE, T_POST); ax.set_ylim(MC_YL)
            P.spines_off(ax); _xticks(ax, True, axislabel=(r_mc == 8)); ax.set_yticks([-2, 0, 2])
            if j: ax.set_yticklabels([])
            if j == 0:
                mc_left.append(ax)
                block_ax[r_mc] = ax
                ax.tick_params(axis="y", pad=1.5)   # shift numbers in; -2 clears the label
            ylab(ax, lm, j)
    # move only the MC row labels in to line up with the GC labels (MC tick labels
    # are wider because of the minus sign).
    fig.canvas.draw()
    xg_disp = gc_left.yaxis.label.get_position()[0]
    xg_axes = gc_left.transAxes.inverted().transform((xg_disp, 0.0))[0]
    for ax in mc_left:
        ax.yaxis.set_label_coords(xg_axes, 0.5)
    # bold EARLY / LATE block labels in the left margin, rotated, each spanning its
    # GC+MC pair (rows 2-4 and 6-8).
    for (r_top, r_bot), name in [((2, 4), "EARLY"), ((6, 8), "LATE")]:
        y_top = block_ax[r_top].get_position().y1
        y_bot = block_ax[r_bot].get_position().y0
        fig.text(0.050, 0.5 * (y_top + y_bot), name, rotation=90, fontsize=10,
                 fontweight="bold", fontstyle="italic", va="center", ha="center")
    # panel letters: A = Odour row, B = Early block, C = Late block. A sits above
    # the odour row (clear of its label); B and C are centred on the tops of their
    # GC plots.
    for r0, letter, va, dy in [(0, "A", "bottom", 0.013), (2, "B", "center", 0.0),
                               (6, "C", "center", 0.0)]:
        fig.text(0.038, block_ax[r0].get_position().y1 + dy, letter, fontsize=10.5,
                 fontweight="bold", va=va, ha="left")
    return fig


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    fig = fig_dynamics()
    for ext in ("pdf", "png"):
        out = os.path.join("figs", f"dynamics.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  -> {out}")

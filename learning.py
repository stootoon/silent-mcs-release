"""Learning figure: one mitral cell followed through learning.

Companion to dynamics.py (the early/late figure). Same simulation run
(dynamics.run_model), same styling. We follow ONE mitral cell -- the one that
ends up silent -- and the granule cells through four presentations of the SAME
odour:

    presentation (early) / mid-1 / mid-2 / (late)

across columns, with rows

    Odour (compressed) / GC / MC

As the cell is driven on each presentation its baseline falls (Eq. dmudt), the
common-feature granule cell is progressively cancelled, and the cell settles into
the 'silent' profile: quiet at rest, large odour-evoked depolarization.

    python learning.py            # -> figs/learning.pdf (+ .png)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import plotting as P
from model import get_x_true
from dynamics import (run_model, ODOUR_COLOURS, C_COMMON, TRACE_COLOURS,
                      _xticks, T_PRE, T_POST, GC_YL, MC_YL,
                      COMMON_CONC, UNIQUE_CONC, N_ODOURS, ODOURS)

FOLLOW_COLOUR = "#1b5e20"          # strong dark green = followed cell
COMPANION_COLOUR = "#c4c4c4"       # light grey = the other two dynamics cells, behind


def _which_la(d):
    """The three example mitral cells of the dynamics figure, same selection."""
    order = np.argsort(d.bg_odour)
    return [order[-1], order[-1 - len(order) // 3], order[0]]


def _early_late(d):
    """Replicate the dynamics figure's early (priming) and late (converged) onset per odour."""
    sid = [(i, int(d.which_pat[i])) for i in d.stim_inds]
    early, late = {}, {}
    t_conv = d.activation_time + 3 * d.tau_b
    for i, o in sid:
        if i < d.activation_index and o not in early:
            early[o] = i
        if d.t_full[i] > t_conv and o not in late:
            late[o] = i
    return early, late


def _presentations(d, o):
    """Onset indices of every presentation of odour o, in time order (k is 1-based)."""
    return [si for si in d.stim_inds if int(d.which_pat[si]) == o]


def select(d, verbose=True):
    """Pick followed cell, odour, and the four presentation columns. Prints the
    per-presentation baseline table and returns everything the figure needs."""
    n_pre = int(T_PRE / d.dt)

    def bl(cell, onset):                      # pre-odour baseline = mean la over 0.5 s pre
        return float(np.mean(d.la[onset - n_pre:onset, cell]))

    def resp(cell, onset):                    # odour-evoked depolarization from baseline
        n_od = int(1.0 / d.dt)
        return float(np.mean(d.la[onset:onset + n_od, cell])) - bl(cell, onset)

    early, late = _early_late(d)
    cells = _which_la(d)

    # Followed cell: of the three, the most negative converged (late) pre-odour baseline
    # (averaged over odours, since the converged baseline is ~odour-independent).
    late_bl = {c: np.mean([bl(c, late[o]) for o in late]) for c in cells}
    follow = min(cells, key=lambda c: late_bl[c])
    companions = [c for c in cells if c != follow]

    # Odour: the one giving the followed cell the largest early (priming) response.
    odour = max(ODOURS, key=lambda o: resp(follow, early[o]))

    onsets = _presentations(d, odour)
    k_of = {si: k + 1 for k, si in enumerate(onsets)}
    bl_late = bl(follow, late[odour])

    # mid-1 / mid-2: presentations (strictly between early and late occurrences) whose
    # followed-cell pre-odour baseline is closest to 1/3 and 2/3 of the late value.
    k_early, k_late = k_of[early[odour]], k_of[late[odour]]
    between = [si for si in onsets if k_early < k_of[si] < k_late]
    mid1 = min(between, key=lambda si: abs(bl(follow, si) - bl_late / 3.0))
    mid2 = min(between, key=lambda si: abs(bl(follow, si) - 2.0 * bl_late / 3.0))
    columns = [early[odour], mid1, mid2, late[odour]]

    if verbose:
        print(f"\n  Followed cell = {follow} (late baseline {late_bl[follow]:+.3f}); "
              f"companions {companions}")
        print(f"  Odour = {odour} (early response {resp(follow, early[odour]):+.3f}); "
              f"late baseline target bl_late = {bl_late:+.3f}")
        print(f"  {'k':>3} {'t (s)':>8} {'pre-odour baseline':>20}   role")
        roles = {early[odour]: "EARLY", mid1: "mid-1", mid2: "mid-2", late[odour]: "LATE"}
        for si in onsets:
            print(f"  {k_of[si]:>3} {d.t_full[si]:>8.2f} {bl(follow, si):>20.3f}   "
                  f"{roles.get(si, '')}")

    return dict(follow=follow, companions=companions, odour=odour, columns=columns,
                k_of=k_of)


def fig_learning(d=None):
    if d is None:
        d = run_model()
    sel = select(d)
    follow, companions, odour = sel["follow"], sel["companions"], sel["odour"]
    columns, k_of = sel["columns"], sel["k_of"]

    t = d.t_full
    n_pre, n_post = int(T_PRE / d.dt), int(T_POST / d.dt)
    xt = get_x_true(d.N, N_ODOURS)
    jc = int(np.argmax(xt[0]))                      # common-feature granule
    juni = int(np.argmax(xt[odour]))                # unique-feature granule
    band_col = ODOUR_COLOURS[odour - 1]
    uni_trace = TRACE_COLOURS[odour - 1]

    def snippet(on):
        sl = slice(on - n_pre, on + n_post)
        return t[sl] - t[on], sl

    fig = plt.figure(figsize=(6.6, 4.4))
    # three rows: Odour compressed (~40%), GC, MC; spacer rows hold the per-panel
    # time ticks.
    gs = GridSpec(5, 4, height_ratios=[0.4, 0.22, 1, 0.22, 1],
                  hspace=0.0, wspace=0.16, left=0.115, right=0.98, top=0.92, bottom=0.12)
    row_lab = {0: "Odour", 2: "GC (a.u.)", 4: "MC (a.u.)"}
    block_ax = {}       # left-column axis of each plot row, for A/B/C labels

    def ylab(ax, r, j):
        if j == 0:
            ax.set_ylabel(row_lab[r], fontsize=10)

    gc_left, mc_left = None, []
    for j, on in enumerate(columns):
        tr, sl = snippet(on)
        # row 0: odour (common tan + unique colour), compressed; title = presentation k
        ax = fig.add_subplot(gs[0, j])
        ax.fill_between([0, 1], 0, COMMON_CONC, color=C_COMMON)
        ax.fill_between([0, 1], COMMON_CONC, COMMON_CONC + UNIQUE_CONC, color=band_col)
        ax.set_xlim(-T_PRE, T_POST); ax.set_ylim(0, COMMON_CONC + UNIQUE_CONC + 0.15)
        P.spines_off(ax); ax.set_yticks([0, 2]); _xticks(ax, True, axislabel=False)
        if j: ax.set_yticklabels([])
        ax.set_title(f"presentation {k_of[on]}", fontsize=9); ylab(ax, 0, j)
        if j == 0:
            block_ax[0] = ax

        # row 2: granule cells (common tan + this odour's unique), same as dynamics
        ax = fig.add_subplot(gs[2, j])
        ax.axhline(0, color="gray", lw=0.4, zorder=1)
        ax.plot(tr, d.x[sl][:, jc], color=C_COMMON, lw=2, zorder=2)        # common
        ax.plot(tr, d.x[sl][:, juni], color=uni_trace, lw=2, zorder=3)     # unique
        ax.set_xlim(-T_PRE, T_POST); ax.set_ylim(GC_YL)
        P.spines_off(ax); _xticks(ax, True, axislabel=False); ax.set_yticks([0, 1])
        if j: ax.set_yticklabels([])
        if j == 0:
            if gc_left is None:
                gc_left = ax
            block_ax[2] = ax
        ylab(ax, 2, j)

        # row 4: mitral cells -- companions grey/thin behind, followed cell dark green/thick
        ax = fig.add_subplot(gs[4, j])
        ax.axhline(0, color="k", lw=0.5, ls=":", zorder=0)   # zero reference
        for c in companions:
            ax.plot(tr, d.la[sl][:, c], lw=0.9, color=COMPANION_COLOUR, zorder=2)
        ax.plot(tr, d.la[sl][:, follow], lw=2.0, color=FOLLOW_COLOUR, zorder=3)
        ax.set_xlim(-T_PRE, T_POST); ax.set_ylim(MC_YL)
        P.spines_off(ax); _xticks(ax, True); ax.set_yticks([-2, 0, 2])
        if j: ax.set_yticklabels([])
        if j == 0:
            mc_left.append(ax); block_ax[4] = ax; ax.tick_params(axis="y", pad=1.5)
        ylab(ax, 4, j)

    # align MC row label with the GC label (as the dynamics figure does)
    fig.canvas.draw()
    xg_disp = gc_left.yaxis.label.get_position()[0]
    xg_axes = gc_left.transAxes.inverted().transform((xg_disp, 0.0))[0]
    for ax in mc_left:
        ax.yaxis.set_label_coords(xg_axes, 0.5)
    # panel letters: A = Odour, B = GC, C = MC. A sits above the odour row (clear of
    # its label); B and C are centred on the tops of their plots.
    for r0, letter, va, dy in [(0, "A", "bottom", 0.013), (2, "B", "center", 0.0),
                               (4, "C", "center", 0.0)]:
        fig.text(0.030, block_ax[r0].get_position().y1 + dy, letter, fontsize=10.5,
                 fontweight="bold", va=va, ha="left")
    return fig


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    fig = fig_learning()
    for ext in ("pdf", "png"):
        out = os.path.join("figs", f"learning.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  -> {out}")

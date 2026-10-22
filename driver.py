#!/usr/bin/env python3
"""Generate every model figure of the common-feature-subtraction model.

    python driver.py                 # all figures -> ./figs as PDF
    python driver.py --format png    # ... as PNG instead
    python driver.py --only bg cdfs  # just those figures
    python driver.py --outdir /tmp/f --seed 1

Everything is simulated from a random seed, so the output is fully reproducible
and no data files are needed.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")                 # headless: render straight to file
import matplotlib.pyplot as plt

from model import SilentMcs
import figures


# Parameters for the figures: the canonical CFS model. The derived quantities
# sigma^2_eff, eta^2_inf, kappa, tau_mu and o_bar are printed when the model runs.
MODEL_PARAMS = dict(N=200, n_odours=12, M=50, sd=0.24, eta=0.08, al=0.951,
                    ga=1.0, be=3.0, tau_la=0.1, tau_v=0.05, t_whiff=1,
                    n_trials=110, first_trial=15,
                    min_air_trials=2, max_air_trials=2, seed=0)
DT = 5e-3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="figs", help="output directory (default: figs)")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"], help="file format")
    ap.add_argument("--seed", type=int, default=None, help="override model seed")
    ap.add_argument("--only", nargs="+", metavar="NAME", choices=list(figures.FIGURES),
                    help="generate only these figures (default: all)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    params = dict(MODEL_PARAMS)
    if args.seed is not None:
        params["seed"] = args.seed

    print(f"Running model {params} dt={DT} ...")
    d = SilentMcs(**params).compute(dt=DT, randomize_trials=False)

    names = args.only or list(figures.FIGURES)
    for name in names:
        builder, fname = figures.FIGURES[name]
        print(f"[{name}]")
        fig = builder(d)
        out = os.path.join(args.outdir, os.path.splitext(fname)[0] + "." + args.format)
        fig.savefig(out, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"  -> {out}")

    print("Done.")


if __name__ == "__main__":
    main()

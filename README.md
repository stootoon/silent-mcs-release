# Silent mitral cells from common feature subtraction

Code to reproduce the modelling figures of

> Tootoonian, Kollo & Schaefer, *"'Silent' mitral cells emerge from common
> feature subtraction."* (code: https://github.com/stootoon/silent-mcs-release)

The model explains the inverse relationship between mitral/tufted-cell baseline
activity and odour responses as a consequence of **common feature subtraction**
(CFS): the olfactory bulb learns a mitral-cell baseline that cancels the receptor
drive from features common to recently sampled odours.

Everything here is a **simulation** — the mixing matrix and odours are generated
from a random seed — so the figures are fully reproducible and **no data files are
required**.

## Install

```bash
pip install -r requirements.txt        # numpy, matplotlib
```

## Reproduce the figures

```bash
python driver.py                       # all figures -> ./figs (PDF)
python driver.py --format png          # ... as PNG
python driver.py --only bg cdfs        # just some figures
python driver.py --outdir /tmp/figs --seed 1
```

This runs the model once and writes:

| name (`--only`) | file | contents |
|---|---|---|
| `dynamics`   | `dynamics.pdf`   | granule- and mitral-cell responses to four odours, early vs. late |
| `bg`         | `bg.pdf`         | common input vs. mitral-cell baseline, with the predicted slope |
| `resp_dists` | `resp_dists.pdf` | odour-response distribution, and response vs. baseline with the predicted slope |
| `whiskers`   | `whiskers.pdf`   | odour responses grouped by pre-odour voltage |
| `cdfs`       | `cdfs.pdf`       | response CDFs grouped by baseline |
| `spikes`     | `spikes.pdf`     | single-cell spiking of four cells, ordered by baseline |

The red lines in `bg` and `resp_dists` are **closed-form predictions** computed from
the model parameters with no fitting (`figures._bg_slope`, `figures._resp_slope`).

## Layout

| file | contents |
|---|---|
| `model.py`    | `SilentMcs` — the CFS model (mitral/granule dynamics + baseline learning). |
| `plotting.py` | model-agnostic plotting primitives (traces, distributions, rasters). |
| `figures.py`  | one builder per figure; `FIGURES` maps names → files. |
| `dynamics.py` | the early/late granule/mitral dynamics figure (`dynamics`). |
| `driver.py`   | runs the model and saves every figure. |

## Notes

- The model (`model.py`) is the canonical CFS model: mitral-cell effective variance
  `sd2_eff`, learning gain `ka`, steady-state `eta2_inf`, following the paper's
  Methods. `MODEL_PARAMS` in `driver.py` are the values used for the figures; the
  model prints the derived quantities (`sigma^2_eff`, `eta^2_inf`, `kappa`, `tau_mu`,
  `o_bar`) when it runs.
- Figures are deterministic given the seed. They reproduce the *phenomena* in the
  paper; exact pixel values depend on parameters and seed.
- `figures.whiskers_stats(d)` reports the per-cell statistics for the `whiskers`
  split (pooled vs. per-cell Mann–Whitney U, and the per-cell Spearman correlation).
  It needs `scipy`, which is not otherwise required.
- `tqdm` is optional (a progress bar only); the code runs without it.

## Licence

Released under the MIT Licence (see `LICENSE`).

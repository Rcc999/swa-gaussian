"""Temperature x decision-threshold grid per configuration, averaged over seeds (no fitting involved).

For each (T, threshold), prints the mean total cost on the validation set and on the
"stress" set (every ambiguous sample counted 3x, ~1/3 ambiguous), each as mean ± std over seeds.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/Users/rayanecharif/master/MA2/PAI/projects/swa-gaussian")
from util import calculate_calibration_curve  # noqa: E402

from analyze import bma, load_runs, stress  # noqa: E402

RUNS = pathlib.Path(__file__).parent / "runs"
TEMPERATURES = [1.0, 1.25, 1.5, 2.0]
THRESHOLDS = [2 / 3, 0.75, 0.8, 0.85]


def total(probs, labels, threshold):
    conf, pred = probs.max(-1), probs.argmax(-1)
    decided = np.where(conf >= threshold, pred, -1)
    cost = np.where(decided == -1, 1.0, np.where(decided == labels, 0.0, 3.0)).mean()
    ece = calculate_calibration_curve(probs, labels, num_bins=20)["ece"]
    return cost + max(ece - 0.1, 0.0)


def main():
    runs = load_runs(sys.argv[1:])

    header = "".join(f"{f'thr={thr:.2f}':>22}" for thr in THRESHOLDS)
    for config, dumps in runs.items():
        print(f"{config} ({len(dumps)} seeds)    entries: val total / stress total")
        print(f"{'':>8}{header}")
        for t in TEMPERATURES:
            cells = []
            for thr in THRESHOLDS:
                vals = []
                for d in dumps:
                    probs, labels = bma(d["logits"], t), d["labels"]
                    s = stress(np.arange(len(labels)), labels)
                    vals.append((total(probs, labels, thr), total(probs[s], labels[s], thr)))
                v = np.array(vals)
                cells.append(f"{v[:, 0].mean():.3f} / {v[:, 1].mean():.3f}±{v[:, 1].std():.2f}")
            print(f"{f'T={t}':>8}" + "".join(f"{c:>22}" for c in cells))
        print()


if __name__ == "__main__":
    main()

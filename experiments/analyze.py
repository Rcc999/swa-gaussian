"""Aggregate run dumps: compare SWAG configurations and calibration methods across seeds.

Fitted calibrators are scored with repeated 2-fold cross-validation on the validation set
(fit on one half, score on the other), so they are not evaluated on their own fitting data.
"stress" duplicates every ambiguous sample 3x (~1/3 ambiguous) to mimic a harder test set.
"""
import collections
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/Users/rayanecharif/master/MA2/PAI/projects/swa-gaussian")
from util import calculate_calibration_curve  # noqa: E402

RUNS = pathlib.Path(__file__).parent / "runs"
THRESHOLD = 2.0 / 3.0
T_GRID = np.round(np.arange(0.5, 5.01, 0.05), 2)


def bma(logits, t):
    z = logits / t
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    return p.mean(0)


def metrics(probs, labels):
    conf, pred = probs.max(-1), probs.argmax(-1)
    decided = np.where(conf >= THRESHOLD, pred, -1)
    cost = np.where(decided == -1, 1.0, np.where(decided == labels, 0.0, 3.0)).mean()
    ece = calculate_calibration_curve(probs, labels, num_bins=20)["ece"]
    return cost, ece, cost + max(ece - 0.1, 0.0)


def stress(idx, labels):
    amb = idx[labels[idx] == -1]
    return np.concatenate([idx, amb, amb])


def fit_ece(logits, labels, idx, t_min):
    grid = T_GRID[(T_GRID >= t_min) & (T_GRID <= 3.0)]
    return min(grid, key=lambda t: calculate_calibration_curve(bma(logits[:, idx], t), labels[idx], 20)["ece"])


def fit_nll_uniform(logits, labels, idx, t_min, ambiguous_weight=1.0):
    """NLL on well-defined samples + (weighted) cross-entropy to the uniform distribution on ambiguous samples."""
    def loss(t):
        p = np.clip(bma(logits[:, idx], t), 1e-12, 1)
        y = labels[idx]
        known = y != -1
        nll = -np.log(p[known, y[known]]).sum()
        uniform_ce = -np.log(p[~known]).mean(-1).sum()
        return (nll + ambiguous_weight * uniform_ce) / len(idx)
    return min(T_GRID[T_GRID >= t_min], key=loss)


CALIBRATORS = {
    "T=1": lambda lg, lb, idx: 1.0,
    "T=1.25": lambda lg, lb, idx: 1.25,
    "T=1.5": lambda lg, lb, idx: 1.5,
    "T=2": lambda lg, lb, idx: 2.0,
    "ECE grid T>=0.5 (old)": lambda lg, lb, idx: fit_ece(lg, lb, idx, 0.5),
    "ECE grid T>=1": lambda lg, lb, idx: fit_ece(lg, lb, idx, 1.0),
    "NLL+uniform T>=0.5": lambda lg, lb, idx: fit_nll_uniform(lg, lb, idx, 0.5),
    "NLL+uniform T>=1": lambda lg, lb, idx: fit_nll_uniform(lg, lb, idx, 1.0),
    "NLL+uniform amb x3 T>=1": lambda lg, lb, idx: fit_nll_uniform(lg, lb, idx, 1.0, ambiguous_weight=3.0),
}


def evaluate(logits, labels, calibrator, rng, num_splits=10):
    """
    Cross-validated (cost, ece, total, stress_total, abstain rate on ambiguous, mean fitted T).
    Each split fits the calibrator on one stratified half and predicts the other half, and vice versa;
    the out-of-fold predictions are joined back into the full set before computing metrics,
    so ECE is always computed on all 140 samples (fixed temperatures reduce to a plain full-set evaluation).
    """
    n = len(labels)
    amb = labels == -1
    results, ts = [], []
    for _ in range(num_splits):
        fold = np.zeros(n, dtype=bool)
        for cls in np.unique(labels):
            members = rng.permutation(np.flatnonzero(labels == cls))
            fold[members[: len(members) // 2]] = True
        probs = np.zeros((n, logits.shape[-1]))
        for fit_idx, eval_idx in ((np.flatnonzero(fold), np.flatnonzero(~fold)),
                                  (np.flatnonzero(~fold), np.flatnonzero(fold))):
            t = calibrator(logits, labels, fit_idx)
            ts.append(t)
            probs[eval_idx] = bma(logits[:, eval_idx], t)
        c, e, tot = metrics(probs, labels)
        s = stress(np.arange(n), labels)
        abstain_amb = (probs[amb].max(-1) < THRESHOLD).mean()
        results.append((c, e, tot, metrics(probs[s], labels[s])[2], abstain_amb))
    return (*np.mean(results, axis=0), float(np.mean(ts)))


def load_runs(prefixes=()):
    """
    Map config name -> list of {"logits": S x N x C, "labels": N} dicts, one per seed.
    Dumps with all 8 dihedral views (S x 8 x N x C) expand into three variants:
    no TTA (view 0 only), "+tta_cycle" (sample s uses view s mod 8) and "+tta_full" (all 8 views).
    """
    runs = collections.defaultdict(list)
    for path in sorted(RUNS.glob("*.npz")):
        config = path.stem.rsplit("_s", 1)[0]
        if prefixes and not config.startswith(tuple(prefixes)):
            continue
        d = np.load(path)
        logits, labels = d["logits"], d["labels"]
        if logits.ndim == 3:
            runs[config].append({"logits": logits, "labels": labels})
            continue
        num_samples, num_views = logits.shape[:2]
        runs[config].append({"logits": logits[:, 0], "labels": labels})
        runs[config + "+tta_cycle"].append(
            {"logits": logits[np.arange(num_samples), np.arange(num_samples) % num_views], "labels": labels})
        runs[config + "+tta_full"].append({"logits": logits.reshape(-1, *logits.shape[2:]), "labels": labels})
    return runs


def main():
    runs = load_runs(sys.argv[1:])

    rng = np.random.default_rng(0)
    print(f"{'config':<22}{'calibration':<24}{'seeds':>5}{'acc':>7}{'cost':>8}{'ECE':>8}{'total':>14}{'stress':>8}{'T':>6}")
    for config, dumps in runs.items():
        for cal_name, calibrator in CALIBRATORS.items():
            rows = []
            for d in dumps:
                logits, labels = d["logits"], d["labels"].copy()
                known = labels != -1
                acc = (bma(logits, 1.0).argmax(-1)[known] == labels[known]).mean()
                rows.append((acc, *evaluate(logits, labels, calibrator, rng)))
            r = np.array(rows)
            m, sd = r.mean(0), r.std(0)
            print(f"{config:<22}{cal_name:<24}{len(dumps):>5}{m[0]:>7.3f}{m[1]:>8.3f}{m[2]:>8.3f}"
                  f"{m[3]:>8.3f} ±{sd[3]:.3f}{m[4]:>8.3f}{m[5]:>9.2f}{m[6]:>6.2f}")
        print()


if __name__ == "__main__":
    main()

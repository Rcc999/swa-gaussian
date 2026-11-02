"""Compare post-hoc recalibration maps (histogram binning, isotonic regression, Platt scaling)
against temperature scaling on the saved validation logits.

Each map g is fit on the BMA top-class confidence -> "is correct" (ambiguous samples count as wrong),
using repeated stratified 2-fold cross-validation exactly like analyze.py. To apply g, the top-class
probability becomes g(conf) and the other classes are rescaled to keep each row summing to 1;
g is clamped so the argmax never changes.

Usage: recalibrate.py [CONFIG_PREFIX ...]
"""
import sys

import numpy as np

from analyze import bma, fit_nll_uniform, load_runs, metrics, stress

EPS = 1e-6


def histogram_binning(num_bins, equal_mass):
    def fit(conf, correct):
        if equal_mass:
            edges = np.quantile(conf, np.linspace(0, 1, num_bins + 1))
        else:
            edges = np.linspace(0, 1, num_bins + 1)
        edges[0], edges[-1] = 0.0, 1.0 + EPS
        bins = np.clip(np.digitize(conf, edges[1:-1]), 0, num_bins - 1)
        # Empty bins fall back to the identity (bin centre)
        values = np.array([correct[bins == b].mean() if np.any(bins == b) else (edges[b] + edges[b + 1]) / 2
                           for b in range(num_bins)])
        return lambda c: values[np.clip(np.digitize(c, edges[1:-1]), 0, num_bins - 1)]
    return fit


def isotonic(conf, correct):
    """Pool-adjacent-violators: non-decreasing step function conf -> P(correct), linearly interpolated."""
    order = np.argsort(conf)
    x, y = conf[order], correct[order].astype(float)
    blocks = [[y[i], 1.0, x[i], x[i]] for i in range(len(x))]  # [mean, weight, x_min, x_max]
    merged = []
    for block in blocks:
        merged.append(block)
        while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
            b = merged.pop()
            a = merged.pop()
            w = a[1] + b[1]
            merged.append([(a[0] * a[1] + b[0] * b[1]) / w, w, a[2], b[3]])
    knots_x = np.array([(b[2] + b[3]) / 2 for b in merged])
    knots_y = np.array([b[0] for b in merged])
    return lambda c: np.interp(c, knots_x, knots_y)


def platt(conf, correct):
    """Logistic regression of correctness on logit(conf), fit by Newton's method (2 parameters)."""
    z = np.log(np.clip(conf, EPS, 1 - EPS) / np.clip(1 - conf, EPS, 1))
    X = np.stack([z, np.ones_like(z)], axis=1)
    w = np.array([1.0, 0.0])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (p - correct)
        hess = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        w -= np.linalg.solve(hess, grad)
    return lambda c: 1 / (1 + np.exp(-(w[0] * np.log(np.clip(c, EPS, 1 - EPS) / np.clip(1 - c, EPS, 1)) + w[1])))


def apply_map(probs, g):
    """Replace the top-class probability by g(conf), rescale the rest, never change the argmax."""
    top = probs.argmax(-1)
    rows = np.arange(len(probs))
    conf = probs[rows, top]
    second = np.sort(probs, axis=-1)[:, -2]
    # Smallest new top probability that keeps it the argmax: g >= second * (1 - g) / (1 - conf)
    lower = second / np.maximum(1 - conf + second, EPS)
    new_conf = np.clip(g(conf), lower, 1 - EPS)
    scale = (1 - new_conf) / np.maximum(1 - conf, EPS)
    out = probs * scale[:, None]
    out[rows, top] = new_conf
    return out


MAPS = {
    "hist 5 equal-width": histogram_binning(5, False),
    "hist 10 equal-width": histogram_binning(10, False),
    "hist 5 equal-mass": histogram_binning(5, True),
    "isotonic": isotonic,
    "Platt (a, b)": platt,
}


def evaluate(logits, labels, method, rng, num_splits=10):
    n = len(labels)
    results = []
    for _ in range(num_splits):
        fold = np.zeros(n, dtype=bool)
        for cls in np.unique(labels):
            members = rng.permutation(np.flatnonzero(labels == cls))
            fold[members[: len(members) // 2]] = True
        probs = np.zeros((n, logits.shape[-1]))
        for fit_idx, eval_idx in ((np.flatnonzero(fold), np.flatnonzero(~fold)),
                                  (np.flatnonzero(~fold), np.flatnonzero(fold))):
            if method == "temperature (NLL+uniform, T>=1)":
                probs[eval_idx] = bma(logits[:, eval_idx], fit_nll_uniform(logits, labels, fit_idx, 1.0))
                continue
            base_fit, base_eval = bma(logits[:, fit_idx], 1.0), bma(logits[:, eval_idx], 1.0)
            correct = (base_fit.argmax(-1) == labels[fit_idx]).astype(float)
            g = MAPS[method](base_fit.max(-1), correct)
            probs[eval_idx] = apply_map(base_eval, g)
        c, e, tot = metrics(probs, labels)
        s = stress(np.arange(n), labels)
        results.append((c, e, tot, metrics(probs[s], labels[s])[2]))
    return np.mean(results, axis=0)


def main():
    runs = load_runs(sys.argv[1:])
    rng = np.random.default_rng(0)
    methods = ["temperature (NLL+uniform, T>=1)"] + list(MAPS)
    print(f"{'config':<26}{'method':<34}{'cost':>7}{'ECE':>7}{'total':>14}{'stress':>8}")
    for config, dumps in runs.items():
        for method in methods:
            r = np.array([evaluate(d["logits"], d["labels"].copy(), method, rng) for d in dumps])
            m, sd = r.mean(0), r.std(0)
            print(f"{config:<26}{method:<34}{m[0]:>7.3f}{m[1]:>7.3f}{m[2]:>8.3f} ±{sd[2]:.3f}{m[3]:>8.3f}")
        print()


if __name__ == "__main__":
    main()

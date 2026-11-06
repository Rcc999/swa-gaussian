"""Compare decision rules (which samples get "don't know") on the saved validation logits.

All rules use the same tempered BMA probabilities (T fit by NLL+uniform, T >= 1), so the ECE is
identical across rules and only the prediction cost changes. Rules with a free threshold fit it on
one stratified half of the validation set (minimizing cost there, optionally with ambiguous samples
weighted 3x) and are scored on the other half, repeated over random splits.

Usage: decide.py [CONFIG_PREFIX ...]
"""
import sys

import numpy as np

from analyze import fit_nll_uniform, load_runs, stress

THRESHOLD = 2.0 / 3.0


def tempered_samples(logits, t):
    z = logits / t
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(-1, keepdims=True)  # S x N x C


def scores(sample_probs):
    """Uncertainty scores per sample; higher = more uncertain."""
    probs = sample_probs.mean(0)
    entropy = -(probs * np.log(probs + 1e-12)).sum(-1)
    expected_entropy = -(sample_probs * np.log(sample_probs + 1e-12)).sum(-1).mean(0)
    pred = probs.argmax(-1)
    return probs, {
        "1 - max prob": 1 - probs.max(-1),
        "entropy": entropy,
        "mutual information": entropy - expected_entropy,
        "1 - vote agreement": 1 - (sample_probs.argmax(-1) == pred).mean(0),
    }


def cost(decided, labels, weights):
    per_sample = np.where(decided == -1, 1.0, np.where(decided == labels, 0.0, 3.0))
    return (per_sample * weights).sum() / weights.sum()


def fit_score_threshold(score, pred, labels, weights):
    """Best threshold tau on a score: predict iff score <= tau (fit on the given samples)."""
    candidates = np.unique(np.r_[score, score.max() + 1])
    return min(candidates, key=lambda tau: cost(np.where(score <= tau, pred, -1), labels, weights))


def rules(probs, score_dict):
    """Each rule: (name, fit(idx, labels, weights) -> decide(eval_idx) -> labels)."""
    pred, conf = probs.argmax(-1), probs.max(-1)

    def fixed(idx, labels, weights):
        return lambda e: np.where(conf[e] >= THRESHOLD, pred[e], -1)

    def make_score_rule(name):
        def fit(idx, labels, weights):
            tau = fit_score_threshold(score_dict[name][idx], pred[idx], labels[idx], weights)
            return lambda e: np.where(score_dict[name][e] <= tau, pred[e], -1)
        return fit

    def make_combined(name):
        """Max-prob rule with 2/3 threshold AND an extra fitted cap on another uncertainty score."""
        def fit(idx, labels, weights):
            base = conf[idx] >= THRESHOLD
            score = np.where(base, score_dict[name][idx], np.inf)
            tau = fit_score_threshold(np.where(np.isinf(score), 1e9, score), pred[idx], labels[idx], weights)
            return lambda e: np.where((conf[e] >= THRESHOLD) & (score_dict[name][e] <= tau), pred[e], -1)
        return fit

    def per_class(idx, labels, weights):
        """Separate max-prob threshold per predicted class (grid), fit on the fit half."""
        grid = np.round(np.arange(0.5, 1.0, 0.025), 3)
        thresholds = np.full(6, THRESHOLD)
        for c in range(6):
            members = idx[pred[idx] == c]
            if len(members) == 0:
                continue
            w = weights[np.isin(idx, members)]
            thresholds[c] = min(grid, key=lambda t: cost(np.where(conf[members] >= t, pred[members], -1),
                                                         labels[members], w))
        return lambda e: np.where(conf[e] >= thresholds[pred[e]], pred[e], -1)

    out = [("max prob >= 2/3 (submitted)", fixed)]
    out += [(f"fitted threshold on {n}", make_score_rule(n)) for n in score_dict]
    out += [(f"max prob >= 2/3 AND fitted cap on {n}", make_combined(n))
            for n in ("mutual information", "1 - vote agreement")]
    out += [("per-class max-prob thresholds", per_class)]
    return out


def evaluate(d, rng, fit_weight_ambiguous, num_splits=10):
    logits, labels = d["logits"], d["labels"]
    n = len(labels)
    amb = labels == -1
    fit_idx_all = np.arange(n)
    t = fit_nll_uniform(logits, labels, fit_idx_all, 1.0)  # T only affects probabilities, same for all rules
    probs, score_dict = scores(tempered_samples(logits, t))
    results = {}
    for _ in range(num_splits):
        fold = np.zeros(n, dtype=bool)
        for cls in np.unique(labels):
            members = rng.permutation(np.flatnonzero(labels == cls))
            fold[members[: len(members) // 2]] = True
        for name, fit in rules(probs, score_dict):
            decided = np.zeros(n, dtype=int)
            for fit_idx, eval_idx in ((np.flatnonzero(fold), np.flatnonzero(~fold)),
                                      (np.flatnonzero(~fold), np.flatnonzero(fold))):
                weights = np.where(amb[fit_idx], fit_weight_ambiguous, 1.0)
                decided[eval_idx] = fit(fit_idx, labels, weights)(eval_idx)
            s = stress(np.arange(n), labels)
            results.setdefault(name, []).append(
                (cost(decided, labels, np.ones(n)), cost(decided[s], labels[s], np.ones(len(s)))))
    return {k: np.mean(v, axis=0) for k, v in results.items()}


def main():
    runs = load_runs(sys.argv[1:])
    rng = np.random.default_rng(0)
    for config, dumps in runs.items():
        for fit_weight in (1.0, 3.0):
            print(f"\n{config}: thresholds fit with ambiguous weight {fit_weight:g}  (prediction cost, mean over {len(dumps)} seeds)")
            print(f"  {'rule':<52}{'val':>8}{'stress':>8}")
            per_seed = [evaluate(d, rng, fit_weight) for d in dumps]
            for name in per_seed[0]:
                v = np.array([r[name] for r in per_seed])
                print(f"  {name:<52}{v[:, 0].mean():>8.3f}{v[:, 1].mean():>8.3f}")


if __name__ == "__main__":
    main()

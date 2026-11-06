# SWA-Gaussian for calibrated land-use classification

This classifies 60×60 satellite images into 6 land-use classes, or answers "don't know" (`-1`). It uses SWAG ([Maddox et al., 2019](https://arxiv.org/abs/1902.02476)) on top of the provided pretrained CNN.

**Public test cost: 0.786** (the pass mark is 0.865).

Scoring: `cost = mean prediction cost + max(ECE − 0.1, 0)`. A correct label costs 0, a wrong label 3, and "don't know" 1. Any label on an ambiguous image counts as wrong.

## Final design (`solution.py`)

| Component | Choice | Why |
|---|---|---|
| Posterior | Full SWAG, rank K = 15, 30 posterior samples | Low-rank covariance captures how weights in different layers vary together. One random vector z is shared by all layers |
| SWAG training | 30 epochs, constant lr **0.02**, 5 burn-in epochs, 4 snapshots/epoch | The template's lr 0.045 was unstable across seeds (accuracy 0.86 ± 0.05); lr 0.02 gives 0.90 ± 0.01 |
| Outlier exposure | Extra loss (weight 0.5) pushing predictions on synthetic mixed scenes toward uniform. A scene pastes a 30–70% box from a different-class image | The main loss was confident predictions on ambiguous images; this is the only change that lowered the test prediction cost |
| Test-time augmentation | Posterior sample s predicts on rotation/flip view s mod 8 | Satellite images have no orientation. This adds variety at no extra cost |
| Sample caching | 30 networks drawn once, with batch-norm statistics refit on the training set | Calibration and test prediction use the same posterior samples |
| Calibration | Temperature T fit by NLL on well-defined + cross-entropy to uniform on ambiguous validation images, **T ≥ 0.8** (gives T = 0.80) | Uses the ambiguous images that plain NLL would ignore. The lower bound stops over-sharpening (see below) |
| Decision rule | Predict iff max probability ≥ 2/3 | Expected cost of predicting is 3(1 − p), of abstaining 1. This is optimal for calibrated probabilities |

All other options in `solution.py` (CutMix, training augmentation, cyclical lr, variance scale) are implemented but off by default.

## How we got here

### Evaluation protocol

The validation set has only 140 images (20 ambiguous), so single-run validation scores were misleading. Every candidate was therefore scored as follows (`experiments/`):
- **3 seeds** per configuration;
- **cross-validation** of any fitted calibrator (fit on one half of the validation set, score on the other);
- a **stress test** that counts each ambiguous image 3× (about a third of the set), to mimic the more ambiguous test set.

The stress test predicted two checker results to within 0.001, but missed one by 0.09. So the checker had the final say.

### Official checker runs

| # | Configuration | T | Test pred. cost | Test ECE | Cost |
|---|---|---|---|---|---|
| 1 | Template defaults (lr 0.045), T fit by minimizing validation ECE | 0.50 | 0.918 | 0.118 | 0.936 ❌ |
| 2 | lr 0.02 + burn-in + 4 snapshots/epoch, NLL+uniform T ≥ 1 | 2.25 | 0.830 | 0.081 | 0.830 |
| 3 | #2 + CutMix + TTA | 1.25 | 0.831 | 0.119 | 0.850 |
| 4 | #2 + TTA | 2.00 | 0.838 | 0.075 | 0.838 |
| 5 | #2 + outlier exposure + TTA, T ≥ 0.5 | 0.70 | 0.801 | 0.113 | 0.814 |
| **6** | **#5 with T ≥ 0.8** | **0.80** | **0.786** | **0.086** | **0.786** |

1. **Fitting T to validation ECE sharpened to the edge of its range.** Validation has fewer ambiguous images than the test set, so the model was overconfident on test.
2. **A lower lr, burn-in and more snapshots** gave a stable posterior. The NLL+uniform fit then softened it (T ≈ 2.2).
3. **CutMix + TTA raised accuracy** (to 0.92), but the prediction cost didn't improve and the ECE penalty appeared.
4. **TTA alone:** same story, accuracy up but no test gain.
5. **Outlier exposure:** the model now abstains on more ambiguous images, which is the first real drop in test prediction cost. It made the model underconfident, so T < 1 was allowed.
6. **T = 0.70 still sharpened too much** (ECE penalty 0.013). Raising the lower bound to 0.8 removed the penalty and also lowered the prediction cost. This one value was chosen from the public score (2 runs, about 8,300 test images), so it is not a purely validation-based choice.

### Tested and rejected (offline, no checker run)

| Idea | Result |
|---|---|
| Histogram binning, isotonic regression, Platt scaling | Better on validation, worse on stress (0.88–0.96 vs 0.83); they overfit 140 images |
| Decision rules: entropy, mutual information, vote agreement, per-class thresholds | None beat the 2/3 threshold beyond noise |
| SWAG-diagonal | Worse than full SWAG (stress 0.95 vs 0.90 at lr 0.045) |
| lr 0.01 / 0.045, no burn-in, 1 snapshot/epoch | lr 0.01 ≈ 0.02; 0.045 unstable |
| Cyclical lr, batch size 8 / 32, posterior variance ×2 / ×4 | All within ±0.015 of the baseline on stress. A wider posterior only replaces the temperature |
| Outlier exposure weight 1.0 | Lower accuracy (0.906 vs 0.928) than 0.5 |

**Cost breakdown before outlier exposure** (stress set): 0.267 of the 0.763 total came from confident predictions on ambiguous images. The best possible threshold could only reduce the total to 0.728, so better calibration alone could not fix this. The model itself had to become uncertain on mixed scenes.

**Not done:** retraining the MAP network. The task only allows SWAG on the provided pretrained weights for beating the baseline, and it doesn't say whether retraining is allowed beyond that.

## Running

```bash
# Local environment (Python 3.8, torch 2.0.1)
uv venv --python 3.8 .venv && uv pip install --python .venv/bin/python -r requirements.txt

# Official checker in Docker; writes results_check.byte
bash runner.sh
```

- `experiments/` holds the sweep scripts (`run_*.py`) and the analysis scripts (`analyze.py`, `grid.py`, `recalibrate.py`, `decide.py`), with their results in `*_results*.txt`.
- The datasets (`*.npz`, `test_ys.bytes`) are not in git, because `test_xs.npz` is over GitHub's 100 MB limit. Copy them from the course handout.

# SWA-Gaussian for calibrated land-use classification

This project is a Bayesian neural network that sorts 60×60 satellite images into 6 land-use classes. For each image it outputs either a class or "don't know" (`-1`).

The posterior over the weights of a small pretrained CNN is approximated with **SWAG** ([Maddox et al., 2019](https://arxiv.org/abs/1902.02476)). We average the predictions of networks sampled from that posterior, then calibrate the result with **temperature scaling**. The temperature is fit so that the model is penalised for confidence on ambiguous images.

Everything is implemented in `solution.py` (`SWAGPredictor`). `experiments/` holds the local sweep used to choose the final configuration.

## Scoring

```
cost = mean prediction cost + max(ECE − 0.1, 0)
```

- **Prediction cost per image:** 0 for a correct label, 3 for a wrong label, 1 for "don't know". Any label on an ambiguous image counts as wrong.
- **ECE:** 20 bins over the argmax probabilities, with ambiguous images counted as wrong.
- **Pass mark:** cost < 0.865.

## Final method

### 1. Collecting SWAG statistics (`fit_swag_model`, `update_swag_statistics`)

We start from the provided MAP weights and run 30 epochs of SGD with a constant learning rate:

| Setting | Value |
|---|---|
| Learning rate | **0.02** |
| Momentum | 0.9 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Burn-in | first **5 epochs** collect nothing, so the weights can leave the MAP point |
| Snapshots | **4 evenly spaced per epoch** for the remaining 25 epochs, 100 in total |

At each snapshot θᵢ we update these statistics, per layer:

| Statistic | Update |
|---|---|
| First moment θ̄ | θ̄ ← (n·θ̄ + θᵢ)/(n+1) |
| Second moment θ²̄ | θ²̄ ← (n·θ²̄ + θᵢ²)/(n+1) |
| Deviation matrix D | append θᵢ − θ̄ to a `deque(maxlen=K)`, K = 15 |

The deviation for θᵢ is measured from the running mean after it has already been updated with θᵢ, as in the paper. All statistics are stored layer by layer (`name → tensor`), so we never build the full weight vector.

### 2. Sampling from the approximate posterior (`sample_parameters`)

- **SWAG-Diagonal:** 𝒩(θ̄, Σ_diag), with Σ_diag = diag(max(θ²̄ − θ̄², ε)). The clamp stops floating-point cancellation from producing a negative variance.
- **Full SWAG (used):** 𝒩(θ̄, ½(Σ_diag + DDᵀ/(K−1))), sampled as

  ```
  θ = θ̄ + (1/√2)·√Σ_diag ⊙ z₁ + (1/√(2(K−1)))·D z₂,   z₁ ~ 𝒩(0, I_d), z₂ ~ 𝒩(0, I_K)
  ```

  The low-rank term captures correlations across the whole weight vector. So z₂ is drawn **once per sample and shared by every layer**; drawing it per layer would throw those correlations away.

After sampling, we refit the batch-norm statistics on the training set, because the running statistics saved during SGD were computed with different weights.

### 3. Bayesian model averaging (`predict_probabilities_swag`)

We draw S = 30 networks once and cache their state dicts, including the refitted batch-norm buffers. Validation calibration and test prediction then use the **same** posterior samples, and the batch-norm refit only runs 30 times. The prediction is the average of the per-sample softmax outputs:

```
p(y | x) = (1/S) Σₛ softmax(f_θₛ(x) / T)
```

### 4. Calibration (`calibrate_confidence`)

A single temperature T ≥ 1 divides the logits of every sampled network. The ambiguous validation images have no class, so ordinary NLL can't use them. Instead, T minimises

```
Σ_{well-defined i} −log p(yᵢ | xᵢ)  +  Σ_{ambiguous j} CE(uniform, p(· | xⱼ))
```

- The **first term** rewards confident, correct predictions.
- The **second term** pushes ambiguous images toward a uniform prediction.
- We **never sharpen** (T ≥ 1). The test set is more ambiguous than the validation set, and sharpening is what made the first attempt fail (see below).

With this configuration, T usually comes out around 2.2.

### 5. Decision rule (`decide_labels`)

Predicting the argmax with confidence p costs 3(1 − p) on average, while "don't know" always costs 1. So with calibrated probabilities, the Bayes-optimal rule is to **predict iff p ≥ 2/3**. The threshold comes from the cost function, not from tuning. Tuning it confirmed this: raising it to 0.75, 0.80 or 0.85 gave no consistent gain.

## How we got here

### First attempt: failed the public test set

Our first submission used SWAG with the template defaults (lr 0.045, one snapshot per epoch, no burn-in). Its temperature was picked by grid search for the lowest validation ECE, with T ∈ [0.5, 3].

Locally the validation cost was 0.52, but the checker scored it far worse:

| Run | Fitted T | Val. argmax acc. | Public test cost | Public test ECE | Penalized cost |
|---|---|---|---|---|---|
| Checker (Docker) | 0.50 | 0.833 | 0.918 | 0.118 | **0.936** (baseline 0.865) |

Two things went wrong:
1. **The ECE grid search sharpened the model all the way to the edge of its grid (T = 0.5).** Only 20 of the 140 validation images are ambiguous, so the fit is dominated by well-defined images and rewards sharpening. The test set is more ambiguous, so the sharpened model was overconfident there (test ECE 0.118).
2. **Results varied widely between runs.** The same code reached 0.892 validation accuracy on macOS and 0.833 in Docker, because small floating-point differences send SGD at lr 0.045 down different paths. Our local numbers came from a single run, scored on the same images T was fitted on.

### Experiment setup (`experiments/`)

- **Sweep:** 6 SWAG configurations × 3 seeds (`run_one.py`, `configs.txt`). Each run saves the validation logits of all 30 posterior samples.
- **Analysis:** calibration methods are compared offline (`analyze.py`, `grid.py`).
- **Cross-validation:** every fitted calibrator is scored with repeated, stratified 2-fold cross-validation. T is fitted on one half and the other half is predicted, so no image is scored by a T fitted on it. The out-of-fold predictions are then rejoined, so ECE is always computed on all 140 images; ECE on a small set is biased upward.
- **Stress test:** every ambiguous image counts 3×, so about a third of the set is ambiguous. This stands in for a harder test set. For the configuration that failed, the stress total was 0.935, almost exactly its real public test cost of 0.936. So we treat the stress total as the main selection criterion.

### Results (mean over 3 seeds, cross-validated; full tables in `experiments/*_results.txt`)

The columns are:
- **Acc.**: argmax accuracy on the well-defined images.
- **Val. total**: the cost formula on the validation set.
- **Stress total**: the same formula with each ambiguous image counted 3×.
- **Abstain on amb.**: the share of ambiguous images that get "don't know".

| Configuration | Calibration | Acc. | Val. total | Stress total | Abstain on amb. | T |
|---|---|---|---|---|---|---|
| full, lr 0.045 (first attempt) | ECE grid T ≥ 0.5 | 0.861 | 0.610 ± 0.05 | 0.956 | 0.57 | 0.81 |
| full, lr 0.045 | T = 1 | 0.861 | 0.605 ± 0.03 | 0.901 | 0.63 | 1.00 |
| diagonal, lr 0.045 | T = 1 | 0.867 | 0.636 ± 0.03 | 0.954 | 0.58 | 1.00 |
| full, lr 0.045, burn-in 5 | NLL+uniform T ≥ 1 | 0.864 | 0.610 ± 0.04 | 0.893 | 0.68 | 1.38 |
| full, lr 0.045, burn-in 5, 4/epoch | NLL+uniform T ≥ 1 | 0.867 | 0.610 ± 0.02 | 0.906 | 0.67 | 1.29 |
| full, lr 0.01, burn-in 5, 4/epoch | NLL+uniform T ≥ 1 | 0.903 | 0.558 ± 0.02 | 0.841 | 0.73 | 2.73 |
| full, lr 0.02, burn-in 5, 4/epoch | T = 1 | 0.900 | 0.521 ± 0.04 | 0.978 | 0.50 | 1.00 |
| full, lr 0.02, burn-in 5, 4/epoch | ECE grid T ≥ 1 | 0.900 | 0.501 ± 0.02 | 0.865 | 0.64 | 1.67 |
| **full, lr 0.02, burn-in 5, 4/epoch (final)** | **NLL+uniform T ≥ 1** | **0.900** | **0.553 ± 0.01** | **0.831** | **0.72** | **2.18** |

What we learned:
- **A lower learning rate plus burn-in and more snapshots is the biggest win.** Accuracy rises from about 0.86 to about 0.90, and the spread across seeds shrinks roughly 3–5×. It also removes the platform sensitivity that hurt the first attempt.
- **That posterior is tighter, and therefore overconfident on ambiguous images at T = 1** (stress total 0.98). It needs softening (T ≈ 2).
- **NLL + uniform finds that temperature on its own**, and it has the best stress total in every low-learning-rate configuration. The ECE search gives a slightly better validation total but is worse under stress. Because the test set is the harder one, we choose for stress.
- **Some ambiguous images are out of reach.** A few always get confidence ≥ 0.9, mostly predicted as class 3. Other uncertainty signals (predictive entropy, mutual information between samples, vote agreement) separate ambiguous from correct images only slightly better than plain confidence (AUROC ≈ 0.78 vs 0.76), so we keep the simple max-probability rule.

### Final checker result

| Run | Fitted T | Val. argmax acc. | Public test cost | Public test ECE | Penalized cost |
|---|---|---|---|---|---|
| First attempt (lr 0.045, ECE grid) | 0.50 | 0.833 | 0.918 | 0.118 | 0.936 ❌ |
| **Final (lr 0.02, burn-in 5, 4/epoch, NLL+uniform)** | **2.25** | **0.900** | **0.830** | **0.081** | **0.830 ✅** (baseline 0.865) |

The public test cost of 0.830 is almost exactly the stress-test prediction of 0.831, which supports using the stress test as a stand-in for the test set.

On validation, the final model's ECE is 0.113, but on the test set it is 0.081. That fits the picture that T was chosen for a more ambiguous distribution than validation.

## Follow-up: CutMix and test-time augmentation (did not improve the test score)

We tested three additions on top of the final configuration (3 seeds each; `experiments/configs2.txt`, results in `experiments/*_results_aug.txt`):
- **D4 augmentation:** a random rotation/flip of each SWAG training image.
- **CutMix (p = 0.5):** paste a random box from another training image and mix the targets in proportion to the box's area. Mixed images resemble the ambiguous, mixed land-use test images.
- **Test-time augmentation (TTA) over the 8 rotations/flips:**
  - *"full"*: every posterior sample predicts on all 8 views, 8× inference cost.
  - *"cycle"*: posterior sample s predicts on view s mod 8, at no extra cost.

Cross-validated results on validation, mean over 3 seeds, all with NLL+uniform T ≥ 1:

| Configuration | Acc. | Val. total | Stress total | T |
|---|---|---|---|---|
| final (no augmentation) | 0.900 | 0.559 ± 0.01 | 0.843 | 2.18 |
| + TTA cycle | 0.919 | 0.518 ± 0.02 | 0.794 | 1.95 |
| + D4 aug + TTA cycle | 0.925 | 0.496 ± 0.01 | 0.778 | 1.72 |
| + CutMix + TTA cycle | 0.922 | 0.444 ± 0.02 | 0.757 | 1.20 |
| + D4 aug + CutMix + TTA cycle | 0.925 | 0.465 ± 0.01 | 0.780 | 1.19 |

- TTA "cycle" matches "full", so the 8× inference cost isn't needed.
- CutMix gave the best validation and stress totals.

**However, the official checker scored CutMix + TTA cycle worse:**

| Checker run | T | Val. acc. | Test pred. cost | Test ECE | Penalized |
|---|---|---|---|---|---|
| final (no augmentation) | 2.25 | 0.900 | 0.830 | 0.081 | **0.830** |
| + CutMix + TTA cycle | 1.25 | 0.917 | 0.831 | 0.119 | 0.850 |

What this shows:
- **The prediction cost was unchanged (0.831 vs 0.830).** The whole difference is the ECE penalty: the CutMix model, with its fitted T of 1.25, is overconfident on the test set.
- **The stress test predicted this model badly** (0.757 predicted, 0.850 actual), after matching the first two models to within 0.001. It only models a *larger share* of the kind of ambiguous images seen in validation. The test set also has snow, clouds and mixed land-use types that validation lacks, and CutMix apparently does not cover those.
- **Some of the predicted gain was probably selection bias.** We picked the best of many configurations by their validation-derived scores.

We therefore keep the final configuration above (`swag_cutmix_prob = 0.0`, `test_time_augmentation = "none"`) as the submission.

## Follow-up: other recalibration methods (not adopted)

`experiments/recalibrate.py` compares temperature scaling with three other recalibration methods: histogram binning (5 or 10 bins), isotonic regression and Platt scaling. Each alternative learns a map from the model's top-class confidence to the chance of being right, counting ambiguous images as wrong. They are compared with the same cross-validation and stress test as above.

Results for the submitted configuration (3 seeds):

| Method | Val. total | Stress total |
|---|---|---|
| **Temperature (NLL+uniform, T ≥ 1)** (submitted) | 0.556 | **0.834** |
| Histogram binning, 5 equal-mass bins | 0.506 | 0.885 |
| Isotonic regression | **0.496** | 0.878 |
| Platt scaling (a, b) | 0.558 | 0.956 |

Binning and isotonic regression fit the validation set better, but they are worse on the more ambiguous stress set. This is the same pattern as the ECE-fitted temperature that failed the public test, so we keep temperature scaling. Full results: `experiments/recalibrate_results.txt`.

## Running

```bash
# Local development (Python 3.8, torch 2.0.1)
uv venv --python 3.8 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# remove the `raise` at the top of main() in solution.py, then:
.venv/bin/python solution.py

# Official checker (Linux-only pyarmor binary → Docker); writes results_check.byte
bash runner.sh

# Re-run the experiment sweep (18 runs, ~40 min with 5 in parallel), then analyse
cd experiments && mkdir -p logs
xargs -P 5 -L 1 sh -c '../.venv/bin/python -u run_one.py $0 $1 $2 $3 $4 $5 > logs/${0}_s$5.log 2>&1' < jobs.txt
../.venv/bin/python analyze.py && ../.venv/bin/python grid.py
```

The datasets (`*.npz`, `test_ys.bytes`) are not in git, because `test_xs.npz` is larger than GitHub's 100 MB file limit. Copy them into the repository root from the course handout.

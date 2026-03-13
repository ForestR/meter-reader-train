# Memo: Digit Natural Distribution and Sample Balancing

**Subject:** Whether to perform sample balancing for Stage 3 (digit classification) given natural digit frequency in meter readings.

**Date:** 2025-03-13

---

## 1. Actual Distribution in Our Dataset

Measured from `data/digit_crops/` (train split):

| Digit | Train | Val | % of Train |
|-------|-------|-----|------------|
| 0     | 2,894 | 723 | **31.96%** |
| 1     | 979   | 244 | 10.81%     |
| 2     | 803   | 200 | 8.87%      |
| 3     | 755   | 188 | 8.34%      |
| 4     | 676   | 168 | 7.47%      |
| 5     | 638   | 159 | 7.05%      |
| 6     | 626   | 156 | 6.91%      |
| 7     | 564   | 140 | 6.23%      |
| 8     | 555   | 138 | 6.13%      |
| 9     | 565   | 141 | 6.24%      |
| **Total** | **9,055** | **2,257** | 100% |

**Imbalance ratio:** Digit 0 has ~5.2× more samples than digit 8 (2,894 vs 555).

---

## 2. Explanation: Benford's Law and Meter Digits

### Benford's Law (Leading Digits)

Benford's Law states that in many naturally occurring datasets, the leading digit *d* (1–9) follows a logarithmic distribution:

$$P(d) = \log_{10}\left(1 + \frac{1}{d}\right)$$

So digit 1 appears ~30.1%, 2 ~17.6%, 3 ~12.5%, …, 9 ~4.6%.

### Why Digit 0 Replaces 1 in Meter Readings

Mechanical meter wheels (odometer-style) have a **fixed number of digits**. Readings are zero-padded on the left:

- `00123` — leading zeros
- `00047` — leading zeros
- `12345` — no leading zeros

Digit **0** therefore plays the role of the "leading digit" in many positions. It becomes the most frequent digit, analogous to how 1 is most frequent in Benford's Law for unbounded numbers.

Our data reflects this: digit 0 is ~32%, roughly matching the Benford expectation for the dominant digit, while 1–9 follow a decreasing trend.

### Summary

The observed distribution is **not noise** — it encodes real-world structure. At inference time, digit crops will have the same distribution. Training on it keeps the model calibrated to deployment conditions.

---

## 3. Potential Backfire of Sample Balancing

Artificially balancing classes (oversampling minority digits, undersampling digit 0, or forcing equal class weights in the data) can backfire in several ways:

### 3.1 Miscalibrated Posterior Probabilities

- Cross-entropy implicitly learns class priors from the training distribution.
- If we balance to 10% per class but in reality 0 appears ~32% of the time, the model will be **under-confident** on 0 and **over-confident** on rare digits.
- Downstream logic (confidence thresholds, multi-digit voting) will degrade.

### 3.2 Overfitting on Minority Classes

- Oversampling means repeating the same ~550 images for digits 7–9 many times.
- The model memorizes specific crops instead of learning generalizable features.
- Validation may look good; generalization suffers.

### 3.3 Discarding Informative Signal

- The imbalance is **informative**, not a bug. Meter readings genuinely have more zeros.
- Erasing this signal harms calibration and real-world accuracy.

### 3.4 When Balancing Might Be Considered

Only if **per-class accuracy** is poor for minority digits (e.g., 7, 8, 9) in the confusion matrix:

- Prefer **collecting more unique data** for underperforming classes.
- Or **augmenting** minority classes more (with care: no horizontal flip to avoid 6↔9 confusion).
- Or **mild class-weighted loss** (adjusts gradient, not data distribution).

---

## 4. Recommendation

**Do not perform sample balancing.** The natural distribution is the correct prior for this task. Train on the data as-is and monitor per-class accuracy. If minority digits underperform, address via data collection or augmentation, not by artificially altering class frequencies.

---

## 5. References

- Benford's Law: [Wikipedia](https://en.wikipedia.org/wiki/Benford%27s_law)
- Stage 3 training: `src/train_pipeline_stage3.py`, `configs/pipeline/stage3_cls/train.yaml`
- Data preparation: `scripts/prepare_stage3_data.py`

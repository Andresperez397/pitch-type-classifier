# Analysis Plan (frozen before modeling)

**Project:** Pitch types are relative. Classifying MLB pitches for pitchers the model has never seen.
**Data:** public Statcast pitch-level data, 2025 MLB regular season (Baseball Savant).
**Author:** Andres Perez
**Plan written:** 2026-10-02, committed before the season download finished and before any model was fit.
**Background:** this rebuilds a 2021 class project (NCAA pitch-type classification) with better validation and a clearer question.

## 1. Questions

- **Q1. Accuracy on unseen pitchers.** How accurately can MLB's pitch-type label be recovered from ball-flight and release data, for pitchers who are entirely absent from training?
- **Q2. Absolute vs relative features.** Does describing each pitch relative to the pitcher's own hardest pitches beat absolute values? Relative here means velocity gap, movement gap, spin gap and spin-axis gap. The hypothesis is yes: an 88 mph pitch is a fastball for one pitcher and a cutter for another.
- **Q3. Model error or label ambiguity?** Where the model disagrees with MLB's label, is it a model failure, or are the labels inconsistent across pitchers? Physically similar pitches carry different names (slider/sweeper/slurve/curveball; cutter/slider; changeup/splitter).

## 2. Data handling

**Labels kept:**
- FF, SI, FC, SL, ST, SV, CU, KC, CH, FS.
- CS (slow curve) is merged into CU.

**Rows dropped and counted:**
- labels FA, PO, EP, KN, SC, FO and unknown
- any pitch missing a model feature

**Handedness:** left-handers are mirrored so every pitcher is described as a right-hander. That means negating `pfx_x` and `release_pos_x`, and using 360° minus `spin_axis`.

**Units:** movement converted from feet to inches.

**Per-pitcher reference (label-free):** the mean of each pitcher's pitches at or above his own 90th-percentile velocity. It never uses the pitch-type label, because a model for a new pitcher can't know his labels in advance.
- Pitchers with fewer than 100 pitches are excluded, so the reference is stable.
- The reference is computed within the data split the pitcher belongs to.

## 3. Feature sets

| Set | Features |
|---|---|
| **A: absolute** | velocity, spin rate, spin axis (sin and cos), horizontal break, induced vertical break, release side, release height, extension, arm angle |
| **B: relative** | A, plus each pitch's gap from the pitcher's reference in velocity, horizontal break, vertical break, spin rate and spin axis (circular difference) |

## 4. Models and validation

**Models** (both fit to sets A and B):
- **Baseline:** multinomial logistic regression with standardized features.
- **Main:** histogram gradient-boosted trees, with fixed settings `max_depth=8`, `learning_rate=0.1`, `max_iter=300`, `class_weight=None`.

**Validation:**
- Split pitchers 80/20 into train and test (GroupShuffleSplit by pitcher), 5 repeats with seeds 0 to 4. No pitcher appears in both sets.
- For speed, the training set is subsampled to 250,000 pitches per repeat. The test set is kept whole.

**Metrics** (mean ± SD over repeats):
- accuracy
- macro-F1
- multiclass log loss
- per-class F1
- confusion matrix, from seed 0

**Pitcher-level metric:** the share of test pitchers whose whole arsenal is classified with at least 95% accuracy.

**Pre-declared comparison (Q2):** relative beats absolute only if it wins macro-F1 in all 5 repeats.

## 5. Label-ambiguity check (Q3)

1. For each test pitch, find its 50 nearest neighbours among other pitchers' training pitches, using standardized set-B features.
2. Record the share of those neighbours that carry the same MLB label.
3. Group misclassified pitches into **model errors**, where most neighbours agree with the MLB label, and **ambiguous**, where most neighbours carry the label the model predicted.
4. Report the split for each confusion pair.

## 6. Deliverables

- A report with figures.
- An interactive app: pick a pitcher and see his arsenal in movement space, MLB labels against model labels, and the disagreements.

## 7. Known limitations

- The MLB labels themselves come from a classification algorithm plus human review. They are the target here, not ground truth.
- One season of data.
- Pitchers with fewer than 100 pitches are excluded.
- The 90th-percentile velocity reference assumes the hardest pitches are fastballs, which can fail for knuckleballers or pitchers who rarely throw a fastball.

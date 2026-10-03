# Pitch types are relative

**What it is:** a classifier that recovers MLB's pitch-type labels from ball flight, for pitchers the model has never seen. When it disagrees with MLB, it checks whether the model is wrong or the label is.

**Data:** the full 2025 MLB regular season from public Statcast. That's 712,528 pitches; 699,305 pitches from 723 pitchers remain after cleaning.

**Author:** Andres Perez. This rebuilds a 2021 class project (NCAA pitch-type classification) with stricter validation and a sharper question.

![Paired comparison](reports/figures/fig1_feature_sets.png)

## Key findings

**1. About 85% of pitches are recovered for brand-new pitchers, but the average hides a lot.**
- Validation holds out 20% of pitchers entirely, 5 repeats.
- The best model is a plain multinomial logistic regression on pitcher-relative features: accuracy 0.849 (SD 0.009) and macro-F1 0.66.
- Four-seamers, sinkers and changeups are recovered well (F1 0.86 to 0.96).
- Knuckle curves (F1 0.22) and slurves (F1 ≈ 0) mostly are not.

**2. Describing a pitch relative to the pitcher's own fastball helps, in every repeat.**
- The relative features are gaps from the pitcher's hardest pitches: velocity, movement, spin and spin axis.
- They raised macro-F1 in 5 of 5 repeats for both models (+0.023 logistic, +0.026 trees). That meets the pre-declared rule.
- The share of pitchers whose whole arsenal is classified at least 95% correctly rose from 21% to 30%.
- The reference is label-free (each pitcher's top 10% of velocities), so it works for a pitcher the model has never seen.

**3. Gradient-boosted trees did not beat logistic regression.**
- On unseen pitchers the trees were slightly *worse* (macro-F1 0.65 vs 0.66, accuracy 0.84 vs 0.85).
- With 700k pitches, extra model flexibility mostly learns the training pitchers' personal naming habits, which don't transfer.

**4. Most disagreements look like inconsistent labels, not model mistakes.**
- For each misclassified pitch, I looked at its 50 nearest neighbors among *other* pitchers' training pitches in the same feature space.
- In 66% of errors, most neighbors carry the label the model chose. The pitch physically resembles what other pitchers call that type.
- In 21%, most neighbors agree with MLB's label, which is a genuine model error. The remaining 13% are mixed.
- Examples:
  - 96% of "cutters predicted as four-seamers" sit among other pitchers' four-seamers.
  - Knuckle curve vs curveball is a grip difference that ball flight can't see.
  - Splitter vs changeup is a similar naming boundary.

![Confusion matrix](reports/figures/fig2_confusion.png)
![Error kinds](reports/figures/fig4_error_kinds.png)

> **Reading the ambiguity check carefully:** a classifier tends to follow its local majority, so "neighbors agree with the model" is partly built in. The useful information is the *other* direction. Only about one disagreement in five sits in a region where other pitchers would agree with MLB. The rest are pitches whose name depends on who throws them, not on how they move.

## Why this matters

A pitch label is a pitcher's own naming decision, not a physical category. That's fine for describing an arsenal. It's a problem when labels are used as model inputs (stuff models, pitch-type splits) across pitchers. A slider for one pitcher can be another's cutter. Pitcher-relative physical descriptions, or clustering within each pitcher, are safer.

## Interactive app

`app.py` (Streamlit) lets you pick any held-out pitcher and compare MLB's labels with the model's in movement space, side by side. A disagreement table shows where they differ. You can sort pitchers by "most disagreement" to find the interesting arsenals.

```bash
streamlit run app.py
```

## How it was done

- **The plan came first.** [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) fixed the label set, features, models, validation and decision rules. It was committed before the season download finished and before any model was fit.
- **Clean data.** Savant caps each export at 25,000 rows, so the fetcher pulls one day per request and fails loudly if a day ever hits the cap. Multi-day pulls silently truncate.
- **Comparable pitchers.** Left-handers are mirrored into a right-handed frame of reference.
- **No label leakage.** The pitcher reference uses no labels and is computed separately within the training and test splits.
- **Honest validation.** GroupShuffleSplit by pitcher, 5 repeats, with the training set subsampled to 250k pitches. Metrics are accuracy, macro-F1, log loss, per-class F1 and a pitcher-level "whole arsenal" rate.
- **Tests.** `tests/` checks that mirroring makes a left-hander identical to a right-hander, that cleaning merges, drops and logs correctly, and that relative features ignore labels and handle the 0°/360° spin-axis wrap.

## Limitations

- **The labels are the target, not ground truth.** MLB's pitch types come from an algorithm plus review, and pitchers name their own pitches.
- **Scope:** one season, and pitchers with fewer than 100 pitches are excluded.
- **Fragile reference:** the hardest-pitch reference assumes the fastest pitches are fastballs, which can fail for pitchers who rarely throw one.
- **Class imbalance:** slurves (0.5% of pitches) and knuckle curves (1.8%) are rare. Their low F1 partly reflects that, though the confusion matrix shows the naming overlap is the bigger factor.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_statcast.py --season 2025     # about 30 min, polite one-day requests
python scripts/run_analysis.py                     # about 4 min
python scripts/make_figures.py
pytest -q
streamlit run app.py
```

Data: Baseball Savant / Statcast (MLB Advanced Media), public pitch-level data, not redistributed here. Code: MIT.

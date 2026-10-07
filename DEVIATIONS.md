# Deviations from ANALYSIS_PLAN.md

Every change made after the plan was frozen is logged here, dated, with its reason. Results from the first run are kept in git history (commit `27eeda4`).

## D1 (2026-10-02): gradient-boosted trees now run with the plan's fixed settings

- **Found in self-review:** scikit-learn's `HistGradientBoostingClassifier` turns on early stopping by default when n > 10,000. It holds out a random 10% of *pitches* to decide when to stop. The plan fixed `max_iter=300`, so the first run did not follow it. Worse, the stopping point was tuned on pitches from pitchers already in the training set: a small, pitcher-level leak into model selection.
- **Fix:** `early_stopping=False`, which is the plan's configuration. A test now asserts it.
- **Effect:** the GBM rows in every table were recomputed. The first run's GBM results are in git history for comparison. The logistic-regression results didn't change.

## D2 (2026-10-02): horizontal movement and release side are now "arm side positive"

- **Why:** the first version mirrored left-handers into a right-hander frame but kept Statcast's catcher-view sign, so arm-side movement was negative. The app's axis label ("arm side +") was therefore wrong.
- **Fix:** both are now multiplied by −1 for right-handers and +1 for left-handers, giving the conventional pitcher's-view frame. This is a sign flip of two features, so it cannot change what a standardized logistic regression or a tree model can learn. A test checks the convention.

## D3 (2026-10-02): added a secondary label-ambiguity check on logistic regression

- **What:** the plan runs the ambiguity check on the main model (GBM). Because logistic regression turned out to be the better model, the same check is also run on its errors and reported alongside.
- **Status:** secondary, labelled as such. The GBM figure remains the primary, pre-specified result.

## D4 (2026-10-02): fetcher hardened (data unchanged)

- **Fix:** an HTML response from Savant is now an error (retried, then raised). It used to be read as "no games that day", which could silently drop a day.
- **Check:** the download is now verified against MLB's official schedule. Every final regular-season game must be present, and no others. The existing 2025 file passes: 2,430 of 2,430 games, none missing or extra.

## D5 (2026-10-02): added an exploratory check with trees tuned on held-out pitchers

- **Why:** with the plan's fixed settings (300 rounds, depth 8), the boosted trees overfit badly: log loss about 2.9, against 0.43 for logistic regression. A fair question is whether properly tuned trees would catch up.
- **What:** `scripts/exploratory_tuned_gbm.py` uses the same 5 pitcher splits and training subsamples. It chooses the number of rounds by early stopping on 15% of the *training pitchers*, held out as a group.
- **Result:** the trees stop at 53 ± 5 rounds and reach accuracy 0.840 ± 0.013 and macro-F1 0.649 ± 0.018. Logistic regression is still better in all 5 repeats on both metrics.
- **Status:** exploratory and labelled as such. It does not replace the pre-specified GBM results.

## Added after publication

- **2026 replication (`scripts/external_season.py`).** Not in the analysis plan. The 2025 logistic regression was scored on the complete 2026 regular season to check that the result is not specific to one season. Pitchers who did not appear in 2025 are the headline group; returning pitchers are shown for contrast. New-pitcher accuracy (0.888) is higher than the 2025 held-out figure (0.849), while macro-F1 is about the same (0.676 against 0.660), so the gain in accuracy is read as a change in who the new pitchers are, not a better model.

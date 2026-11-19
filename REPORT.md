# Internet Usage Patterns Among Students — Analysis Report

## 1. Objective
To understand how students use the internet, how much of that use supports studying, and whether a simple machine learning model can predict if internet usage distracts students from their studies.

## 2. Data
- Source: Google Form survey (`Untitled form.csv`), collected 16 Aug 2026.
- Size: 107 rows x 16 columns; 1 completely blank response removed, leaving **106 responses**.
- Quality: no duplicate rows; 29 missing values in total (1–4 per column, mostly skipped questions). Missing inputs were imputed with the most frequent value for modelling.
- Columns cover device, age band, daily hours, time of day, academic use, purpose, platform, main activity, skill-learning, perceived academic benefit, distraction, aimless checking, self-rating, suggestions, and share of use related to studies. All answers are categorical.

## 3. Key findings (exploratory analysis)

| Topic | Result |
|---|---|
| Device | Smartphone dominates (92 of 105 respondents) |
| Age | Most common band is 19–20 (52 of 105) |
| Daily use | **53.3%** spend more than 3 hours online; 24.8% spend 2–3 h; 21.0% spend 1–2 h; 1 student spends under 1 h |
| Main purpose | Social media 41.5%, academic/studies 20.8%, entertainment 18.9%, other 7.5%, gaming 5.7% |
| Most-used platform | Instagram 61.0%, YouTube 15.2%, other 14.3%, Google 6.7%, WhatsApp 2.9% |
| Academic use | Sometimes 39.6%, often 31.1%, occasionally 21.7%, rarely 4.7%, never 2.8% |
| Distraction | "Yes, sometimes" 71 (67.6%), "Yes, frequently" 22 (21.0%), rarely 10, never 2 |
| Internet helps performance | Neutral 36, strongly agree 35, agree 27, disagree 4, strongly disagree 2 |

**Heavy use goes with frequent distraction.** Among students online more than 3 hours a day, 19 of 55 (about 35%) say they are *frequently* distracted. Among those online 1–3 hours, only 3 of 48 (about 6%) say so. Only 4 of the 55 heavy users answered "rarely" and none answered "never".

**Overall:** about 89% of respondents report being distracted at least sometimes, social media (especially Instagram) is the main use, and only about one in five students primarily uses the internet for studying.

![Daily usage](../figures/01_daily_internet_usage.png)
![Purpose](../figures/02_main_purpose.png)
![Platforms](../figures/03_platforms.png)
![Academic use](../figures/04_academic_usage.png)
![Distraction](../figures/05_distraction.png)
![Perceived performance](../figures/06_perceived_performance.png)

## 4. Machine learning model
- **Model:** Random Forest classifier (50 trees, `random_state=42`), categorical inputs one-hot encoded via `pd.get_dummies`.
- **Target:** "Does internet usage sometimes distract you from your studies?" (4 classes).
- **Features:** hours per day, main purpose, most-used platform, academic-use frequency.
- **Split:** 80/20 stratified — 84 training, 21 test samples.

**Results (test set)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Never | 0.00 | 0.00 | 0.00 | 1 |
| Rarely | 0.00 | 0.00 | 0.00 | 2 |
| Yes, frequently | 0.00 | 0.00 | 0.00 | 4 |
| Yes, sometimes | 0.63 | 0.86 | 0.73 | 14 |

Accuracy **57.14%**; macro-F1 0.18; weighted-F1 0.48.

![Confusion matrix](../figures/07_confusion_matrix.png)

**Interpretation.** The model only learned to predict the majority class ("Yes, sometimes"); it never correctly identified any of the other three classes. Its accuracy is actually *below* the naive baseline of always answering "Yes, sometimes" (14/21 = 66.7% on this test set, 67.6% overall). So the current model is not useful for prediction, and the 57% figure should not be presented as a good result.

**Example prediction:** the interactive cell was tried with free-text input (5 hours, "entertainment,academics", "netflix", "often"). None of those strings match the survey's categories, so after encoding the input was almost all zeros and the model returned the majority class. Inputs must use the exact survey options.

## 5. Limitations
- Small sample (106) from one population; likely a convenience sample, so results may not generalise.
- Self-reported answers, which can be biased.
- Severe class imbalance (Never = 2, Rarely = 10 overall) and 21 test rows, so metrics are very noisy.
- Only four features were used; hours are coarse bands.
- The improved pipeline version (class-weighted, in the project notebook) was never executed, so no results exist for it.

## 6. Recommendations
1. Re-run the pipeline version with `class_weight="balanced"`, and compare against the majority-class baseline.
2. Merge classes (e.g. "Frequently / Sometimes" vs "Rarely / Never", or distracted vs not) to reduce imbalance.
3. Use cross-validation (e.g. stratified 5-fold) instead of a single 21-row test set, and report per-class metrics.
4. Add features already in the survey (time of day, aimless checking, main activity, share of use for study).
5. Collect more responses, particularly from lightly-distracted students.
6. Use dropdown-constrained inputs in the prediction cell.

## 7. Conclusion
Students in this survey are heavy, smartphone-based, social-media-oriented internet users, and most report being at least sometimes distracted — with distraction rising sharply for those online more than 3 hours a day. The descriptive findings are clear; the predictive model, however, does not yet beat a trivial baseline and needs the improvements above.

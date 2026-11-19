# Internet Usage Patterns Among Students

Survey analysis and a Random Forest model predicting whether internet usage distracts students from studying.

## Contents
- `notebooks/internet_usage_analysis.ipynb` — combined notebook (load, clean, EDA, model, prediction) with saved outputs
- `notebooks/original_project.ipynb`, `notebooks/original_ml_model.ipynb` — the two original notebooks
- `figures/` — charts exported from the notebooks
- `reports/REPORT.md`, `reports/REPORT.pdf` — full report
- `data/` — place the survey CSV here (not included; contains survey responses)

## Run
Open the combined notebook in Google Colab, upload the survey CSV (`Untitled form.csv`) when prompted, and run the cells in order.

Requirements: `pip install -r requirements.txt`

## Headline results
- 53% of students use the internet more than 3 hours a day; Instagram and social media dominate.
- ~89% say internet use distracts them at least sometimes.
- Baseline Random Forest accuracy is 57.1%, below a majority-class baseline (~67%), so the model needs improvement (see report).

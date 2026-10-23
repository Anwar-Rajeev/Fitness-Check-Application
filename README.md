# FitScore Studio

An educational Streamlit regression app for the supplied fitness_score_dataset.csv.

## Run locally (Python 3.11 or 3.12)

Keep app.py, requirements.txt and fitness_score_dataset.csv in the same folder.

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first launch trains and compares models; later interactions reuse the cached result. No API keys, saved model file or external database are required.

## Publish on Streamlit Community Cloud

1. Extract this ZIP.
2. Create a GitHub repository and upload app.py, requirements.txt and fitness_score_dataset.csv together at the repository root. The CSV is needed at runtime.
3. Sign in at https://share.streamlit.io/ and choose Create app.
4. Select that repository and branch, and set the main file path to app.py.
5. Select Python 3.12 in advanced settings and deploy.
6. Share the resulting app URL or turn that URL into a QR code for your class.

Publishing the repository or app can expose its bundled dataset. Use only a dataset you are permitted to distribute.

## Dataset understanding

1,000 rows; five numeric predictors: height in cm, weight in kg, daily steps, water intake in litres, sleep in hours. Fitness_Score is a continuous target observed between 1 and 10. The supplied file has no missing values and no duplicate rows. No IDs, dates, age, sex, medical history or categorical columns are provided. Score derivation and provenance are unknown. No BMI feature is added: this app uses exactly the five supplied predictors.

## Missing values and modelling

Numeric coercion, infinity checks and basic impossible-value checks precede splitting. Exact duplicate rows and rows with missing or invalid targets are removed. Targets are never imputed. Predictors use SimpleImputer(strategy='median') inside the pipeline. Linear regression additionally uses StandardScaler; the two tree methods do not need scaling. Imputation statistics are learned separately within each CV fold. If a whole predictor has no observed values, training stops with an explanation.

The input form can mark individual fields unknown; these go through the same fitted imputer. This is a fallback, not proof of accuracy with missing inputs: the original data contains none, so missing-input accuracy has not been independently validated. All-unknown predictions are blocked.

Reproducible 80/20 random split (seed 42), five-fold shuffled CV on the 800 training rows, modest parameter grids, lowest mean CV MAE chooses the algorithm. Only the selected algorithm is evaluated on the 200 holdout rows, against a mean-only baseline. The selected recipe is then refitted on all 1,000 rows for interactive predictions. Raw model outputs are reported without clipping.

## Observed results (scikit-learn 1.8.0)

| Algorithm | CV MAE | Fold SD |
|---|---:|---:|
| Random Forest | 0.6927 | 0.0336 |
| Decision Tree | 0.9008 | 0.0336 |
| Linear Regression | 0.9771 | 0.0386 |

Selected Random Forest: 150 trees, unrestricted depth, minimum leaf size 1.
Held-out MAE 0.7054; RMSE 0.8802; R² 0.7469. Mean-only baseline MAE 1.4550.
R² is not percentage accuracy. Test MAE is an average error, not a personal uncertainty interval. Results may vary slightly across supported dependency versions.

To repeat evaluation without launching Streamlit:

```bash
python app.py --evaluate
```

## Interface

Responsive single-profile form, explicit unknown-value controls, score gauge, download of the current result, observed-range warnings, CV comparison, holdout scatter chart, missing-data audit, and preprocessing explanation. Personal form values stay in the current Streamlit session and are not written to files or a database by this code. Hosting providers have their own policies. Closing/restarting the session can clear the prediction. CSV download is initiated by the user.

## Scope

This predicts the dataset's fitness score, not overall clinical health. There is no evidence supplied for medical validity or recommended score thresholds. It cannot diagnose conditions, recommend treatment, or determine causal effects of lifestyle changes. Additional representative data, score documentation, and independent validation are needed before real health use.

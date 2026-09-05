# Equipment Failure Prediction — Logistic Regression

Predicting whether an industrial machine will fail within its next operating window, using sensor and maintenance data.

## Problem Statement

Industrial equipment failure is costly and often preventable if warning signs in sensor data are caught early. This project builds a binary classification model to predict equipment failure (`target`: 0 = no failure, 1 = failure) based on operational sensor readings.

## Dataset

`dataset_06_equipment_failure_warning.csv`

| Feature | Description |
|---|---|
| `temperature` | Operating temperature reading |
| `vibration` | Vibration sensor reading |
| `pressure` | Pressure sensor reading |
| `runtime_hours` | Continuous runtime since last cycle start |
| `maintenance_gap_days` | Days since last maintenance |
| `power_draw` | Power consumption reading |
| `target` | 0 = No failure, 1 = Failure (label) |

## Approach

1. **EDA** — inspected shape, dtypes, missing values, duplicates, and class balance
2. **Preprocessing** — train/test split performed *before* scaling to avoid data leakage; features standardized with `StandardScaler` (fit on train only)
3. **Model** — `LogisticRegression` (scikit-learn), `max_iter=1000`, `random_state=42`
4. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve
5. **Interpretation** — model coefficients ranked by magnitude to identify which sensor readings most influence failure risk

## Results

<!-- Fill in with your actual Cell 3 / Cell 6 / Cell 8 output before submitting -->

- **Class balance:** _e.g. XX% no-failure / XX% failure_
- **Accuracy:** _TBD_
- **Precision:** _TBD_
- **Recall:** _TBD_
- **F1 Score:** _TBD_
- **ROC-AUC:** _TBD_
- **Top predictive features (by |coefficient|):** _TBD_

## Key Insight

In predictive maintenance, a false negative (missed failure) is typically more costly than a false alarm — so recall is weighted as heavily as accuracy when judging this model's real-world usefulness.

## Limitations & Future Work

- Trained on a single dataset snapshot; real deployment would need retraining as sensor drift occurs over time
- Logistic Regression assumes a linear decision boundary — tree-based models (Random Forest, XGBoost) could be compared for potential accuracy gains
- No hyperparameter tuning (e.g. regularization strength `C`) was performed in this baseline

## How to Run

```bash
pip install pandas scikit-learn matplotlib
```

Open `Equipment_Failure_Warning_LogisticRegression.ipynb` in Jupyter or Google Colab and run all cells in order.

## Tech Stack

Python · pandas · scikit-learn · matplotlib

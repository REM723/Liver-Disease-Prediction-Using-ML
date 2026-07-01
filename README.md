# Liver Disease Prediction

A RandomForest model estimates the likelihood of liver disease from a standard
blood panel, served through a clean single-page Flask app.

Trained on the Indian Liver Patient Dataset (`liver.csv`). Educational only —
not a diagnosis.

## Run

```bash
pip install flask scikit-learn
python app.py
```

Open http://localhost:5000. The model trains from `liver.csv` on startup (~0.5s),
so there's no separate training step.

## Inputs

Age, sex, and nine liver-panel values: total/direct bilirubin, alkaline
phosphatase, ALT, AST, total proteins, albumin, and albumin/globulin ratio.
Returns an estimated probability of liver disease.

## Files

| File | What |
|------|------|
| `app.py` | Flask app — trains the model and serves the form |
| `liver.csv` | Indian Liver Patient Dataset |
| `Advance Project Liver Disease Prediction Using ML.ipynb` | EDA + modelling notebook |
| `roc_liver.jpeg`, `PE_liver.jpeg` | ROC curve and performance eval plots |

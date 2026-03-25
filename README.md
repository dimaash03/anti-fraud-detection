# Anti‑Fraud Detection Pipeline (BigQuery + LightGBM)

End‑to‑end fraud detection project built for a real‑world competition setup. The solution ingests user and transaction data from BigQuery, engineers behavioral features, trains a LightGBM classifier, and optimizes the decision threshold for F1.

## Highlights
- Feature engineering from transaction history (volume, diversity, velocity, error patterns, geo mismatch)
- Target encoding with out‑of‑fold safety to avoid leakage
- Graph‑style features (card sharing, fraud exposure, card failure rates)
- LightGBM model with cross‑validation and threshold tuning
- Submission generation and upload to GCS

## Results (OOF)
- Best F1 (OOF): **0.6527**
- Optimal threshold: **0.49**
- Fraud rate in training: **3.83%**

## Data
Raw data is not included in this repository due to size and privacy.  
Expected tables in BigQuery:

- `fraud_detection.train_users`
- `fraud_detection.train_transactions`
- `fraud_detection.test_users`
- `fraud_detection.test_transactions`

## Tech Stack
Python, pandas, scikit‑learn, LightGBM, BigQuery (GCP), GCS

## How to Run

1. Install requirements
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

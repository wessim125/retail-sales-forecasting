# Retail Sales Forecasting

A sales forecasting project for Rossmann stores, focused on predicting daily store sales using time-aware features, promotion signals, and historical demand patterns.

## Business objective

Rossmann needs an accurate sales forecast to improve:
- inventory planning
- staffing decisions
- promotions and campaign planning
- operational efficiency across stores

This project builds a predictive pipeline that benchmarks a simple temporal baseline against a machine learning model and evaluates the business impact of the improvement.

## Project structure

- `data/raw/` — original competition datasets
- `data/processed/` — cleaned and feature-engineered datasets
- `notebooks/` — exploratory and validation notebooks
- `src/` — preprocessing, feature engineering, and training logic
- `reports/` — quality summaries and comparison plots
- `app.py` — lightweight Streamlit dashboard for business exploration
- `requirements.txt` — project dependencies

## Data

The project uses Rossmann store sales data, including:
- historical daily sales
- promotional periods
- store metadata
- seasonal patterns
- competition indicators
- holiday context

## Pipeline

1. Data inspection and quality summary
2. Cleaning of invalid and inconsistent records
3. Feature engineering for time-series patterns
4. Temporal validation against a baseline
5. Model training with LightGBM
6. Performance comparison and business interpretation

## Key results

The final LightGBM model significantly outperforms a temporal baseline:

- Baseline MAE: 1240.35
- Baseline RMSE: 1649.65
- Baseline RMSPE: 0.2332
- LightGBM MAE: 372.15
- LightGBM RMSE: 549.73
- LightGBM RMSPE: 0.0742

This represents a strong improvement in forecast quality for daily retail sales.

## Model and technology stack

- Python
- pandas
- NumPy
- scikit-learn
- LightGBM
- matplotlib
- seaborn
- pyarrow
- Streamlit

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Run the dashboard:

```bash
streamlit run app.py
```

## Portfolio value

This project is designed to show end-to-end data science capability:
- real-world data cleaning
- feature engineering for forecasting
- chronological validation
- model benchmarking
- metric-based evaluation
- business-facing communication of results

## Notes

Large model artifacts are intentionally not committed to GitHub to keep the repository lightweight and portable. The code and notebooks are ready for reproduction in a local environment.

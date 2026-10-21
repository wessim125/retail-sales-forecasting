from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def rmspe(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true > 0
    diff = (y_pred[mask] - y_true[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval']
    numeric_features = [
        c for c in X.columns if c not in categorical_features + ['Store', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday']
    ]
    # keep the important raw numeric features explicitly if they are present
    numeric_features = list(dict.fromkeys([
        'Customers', 'Open', 'Promo', 'SchoolHoliday', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'year', 'month', 'day',
        'day_of_week', 'week_of_year', 'is_month_start', 'is_month_end', 'is_weekend',
        'competition_open', 'competition_months_since_open', 'promo2_active_month', 'promo2_since_months',
        'sales_lag_7', 'sales_lag_14', 'sales_lag_28', 'sales_rolling_mean_7', 'sales_rolling_mean_28'
    ] + numeric_features))
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )


def train_random_forest(project_root: Path) -> tuple[Pipeline, dict[str, float]]:
    df = pd.read_parquet(project_root / 'data' / 'processed' / 'train_features.parquet').sort_values('Date').copy()
    split_date = df['Date'].max() - pd.Timedelta(weeks=6)
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()

    X_train = train_df.drop(columns=['Sales'])
    y_train = train_df['Sales']
    X_test = test_df.drop(columns=['Sales'])
    y_test = test_df['Sales']

    preprocessor = build_preprocessor(X_train)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, preds),
        'RMSE': mean_squared_error(y_test, preds, squared=False),
        'RMSPE': rmspe(y_test, pd.Series(preds, index=y_test.index)),
    }

    model_path = project_root / 'models' / 'rf_model.joblib'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return pipeline, metrics


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    _, metrics = train_random_forest(project_root)
    print(metrics)

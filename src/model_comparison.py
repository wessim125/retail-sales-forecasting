from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def rmspe(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true > 0
    if not mask.any():
        return float('nan')
    diff = (y_pred[mask] - y_true[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def build_model_results(project_root: Path) -> pd.DataFrame:
    df = pd.read_parquet(project_root / 'data' / 'processed' / 'train_features.parquet').sort_values('Date').copy()
    split_date = df['Date'].max() - pd.Timedelta(weeks=6)
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()

    baseline = train_df.groupby(['Store', 'day_of_week'])['Sales'].mean().rename('baseline_sales').reset_index()
    test_with_baseline = test_df.merge(baseline, on=['Store', 'day_of_week'], how='left')
    base_pred = test_with_baseline['baseline_sales'].fillna(train_df['Sales'].mean())
    baseline_metrics = {
        'Model': 'Baseline',
        'MAE': mean_absolute_error(test_df['Sales'], base_pred),
        'RMSE': np.sqrt(mean_squared_error(test_df['Sales'], base_pred)),
        'RMSPE': rmspe(test_df['Sales'], pd.Series(base_pred, index=test_df.index)),
    }

    cat_cols = ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval']
    num_cols = [
        'Customers', 'Open', 'Promo', 'SchoolHoliday', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'year', 'month', 'day',
        'day_of_week', 'week_of_year', 'is_month_start', 'is_month_end', 'is_weekend', 'competition_open',
        'competition_months_since_open', 'promo2_active_month', 'promo2_since_months', 'sales_lag_7',
        'sales_lag_14', 'sales_lag_28', 'sales_rolling_mean_7', 'sales_rolling_mean_28'
    ]
    feature_cols = [col for col in train_df.columns if col not in {'Sales', 'Date', 'Store'}]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df['Sales']
    y_test = test_df['Sales']

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), [c for c in num_cols if c in X_train.columns]),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]), [c for c in cat_cols if c in X_train.columns]),
    ])

    linear_model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    linear_metrics = {
        'Model': 'Linear Regression',
        'MAE': mean_absolute_error(y_test, linear_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, linear_pred)),
        'RMSPE': rmspe(y_test, pd.Series(linear_pred, index=y_test.index)),
    }

    rf_model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1))
    ])
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = {
        'Model': 'Random Forest',
        'MAE': mean_absolute_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'RMSPE': rmspe(y_test, pd.Series(rf_pred, index=y_test.index)),
    }

    lgb_train = X_train.copy()
    lgb_test = X_test.copy()
    for col in cat_cols:
        if col in lgb_train.columns:
            lgb_train[col] = lgb_train[col].astype('category')
            lgb_test[col] = lgb_test[col].astype('category')

    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=255,
        subsample=0.85,
        colsample_bytree=0.9,
        min_child_samples=8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    lgb_model.fit(lgb_train, y_train, categorical_feature=[c for c in cat_cols if c in lgb_train.columns])
    lgb_pred = lgb_model.predict(lgb_test)
    lgb_metrics = {
        'Model': 'LightGBM',
        'MAE': mean_absolute_error(y_test, lgb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lgb_pred)),
        'RMSPE': rmspe(y_test, pd.Series(lgb_pred, index=y_test.index)),
    }

    results = pd.DataFrame([baseline_metrics, linear_metrics, rf_metrics, lgb_metrics]).sort_values('RMSE').reset_index(drop=True)
    return results


def save_model_comparison(project_root: Path) -> pd.DataFrame:
    results = build_model_results(project_root)
    out_dir = project_root / 'reports'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'model_comparison.csv'
    results.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(results['Model'], results['RMSE'], color=['#7f8c8d', '#3498db', '#f39c12', '#2ecc71'])
    ax.set_title('Model Comparison by RMSE')
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Model')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / 'model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return results


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    results = save_model_comparison(project_root)
    print(results.to_string(index=False))

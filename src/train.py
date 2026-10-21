from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd


def build_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
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


def train_model(project_root: Path) -> tuple[lgb.LGBMRegressor, pd.DataFrame]:
    df = pd.read_parquet(project_root / 'data' / 'processed' / 'train_features.parquet')
    df = df.sort_values('Date').copy()

    target = df['Sales']
    exclude_cols = {'Sales', 'Date', 'Store'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    features = df[feature_cols].copy()

    categorical_cols = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']
    for col in categorical_cols:
        if col in features.columns:
            features[col] = features[col].astype('category')

    model = build_model()
    model.fit(
        features,
        target,
        categorical_feature=[col for col in categorical_cols if col in features.columns],
    )

    model_path = project_root / 'models' / 'lightgbm_model.joblib'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model, features


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    model, _ = train_model(project_root)
    print('Model trained and saved to:', project_root / 'models' / 'lightgbm_model.joblib')
    print('Model type:', type(model).__name__)

from pathlib import Path

import pandas as pd


def normalize_state_holiday(series: pd.Series) -> pd.Series:
    """Convert StateHoliday values to a consistent categorical representation."""
    cleaned = series.astype(str).str.strip().str.lower()
    cleaned = cleaned.replace({'0': '0', 'nan': '0', 'none': '0', 'na': '0'})
    return cleaned


def clean_store_data(store: pd.DataFrame) -> pd.DataFrame:
    """Clean Store metadata and encode missing categories according to business rules."""
    store_clean = store.copy()

    # Competition distance: median imputation + missing indicator
    store_clean['CompetitionDistance_missing'] = store_clean['CompetitionDistance'].isna()
    comp_distance_median = store_clean['CompetitionDistance'].median()
    store_clean['CompetitionDistance'] = store_clean['CompetitionDistance'].fillna(comp_distance_median)

    # Competition dates: encode as not applicable rather than blind imputation
    for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']:
        store_clean[f'{col}_missing'] = store_clean[col].isna()
        store_clean[col] = store_clean[col].fillna(-1)

    # Promo2 dates: encode as not applicable rather than blind imputation
    for col in ['Promo2SinceWeek', 'Promo2SinceYear']:
        store_clean[f'{col}_missing'] = store_clean[col].isna()
        store_clean[col] = store_clean[col].fillna(-1)

    # PromoInterval: use explicit 'none' for non-applicable stores
    store_clean['PromoInterval'] = store_clean['PromoInterval'].fillna('none')
    store_clean['PromoInterval'] = store_clean['PromoInterval'].astype(str)

    # Standardize string codes
    for col in ['StoreType', 'Assortment', 'PromoInterval']:
        store_clean[col] = store_clean[col].astype(str).str.strip().str.lower()

    return store_clean


def clean_train_data(train: pd.DataFrame) -> pd.DataFrame:
    """Clean the sales table and keep only usable rows for model training."""
    train_clean = train.copy()
    train_clean['Date'] = pd.to_datetime(train_clean['Date'])
    train_clean['StateHoliday'] = normalize_state_holiday(train_clean['StateHoliday'])

    # Business rule: closed days are excluded from training and should be predicted as 0
    train_clean['is_closed'] = train_clean['Open'] == 0
    train_clean = train_clean[train_clean['Open'] == 1].copy()

    # Remove rare anomalies where the store is open but sales are zero
    train_clean = train_clean[train_clean['Sales'] > 0].copy()

    return train_clean


def build_processed_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(raw_dir / 'train.csv', low_memory=False)
    store_raw = pd.read_csv(raw_dir / 'store.csv', low_memory=False)

    train_clean = clean_train_data(train_raw)
    store_clean = clean_store_data(store_raw)
    merged = train_clean.merge(store_clean, on='Store', how='left')

    train_clean.to_parquet(processed_dir / 'train_clean.parquet', index=False)
    store_clean.to_parquet(processed_dir / 'store_clean.parquet', index=False)
    merged.to_parquet(processed_dir / 'train_store_clean.parquet', index=False)

    return train_clean, store_clean, merged


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    train_clean, store_clean, merged = build_processed_data(project_root)
    print('train_clean shape:', train_clean.shape)
    print('store_clean shape:', store_clean.shape)
    print('merged shape:', merged.shape)
    print('missing values after cleaning:\n', merged.isna().sum().to_string())

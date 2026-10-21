from pathlib import Path

import pandas as pd


MONTH_MAP = {
    1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
    7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
}


def load_feature_input(project_root: Path) -> pd.DataFrame:
    train = pd.read_parquet(project_root / 'data' / 'processed' / 'train_clean.parquet')
    store = pd.read_parquet(project_root / 'data' / 'processed' / 'store_clean.parquet')
    return train.merge(store, on='Store', how='left')


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df


def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['competition_open'] = df['CompetitionOpenSinceYear'].notna() & (df['CompetitionOpenSinceYear'] != -1)
    df['competition_months_since_open'] = (
        (df['year'] - df['CompetitionOpenSinceYear']) * 12 +
        (df['month'] - df['CompetitionOpenSinceMonth'])
    )
    df.loc[~df['competition_open'], 'competition_months_since_open'] = pd.NA
    df['competition_months_since_open'] = df['competition_months_since_open'].astype('Float64')
    return df


def add_promo2_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['promo2_active_month'] = 0
    df['promo2_since_months'] = 0

    def active_month(row):
        if row['Promo2'] != 1:
            return 0
        interval = str(row['PromoInterval']).strip().lower()
        if interval in ['none', 'nan', 'nat']:
            return 0
        current_month = MONTH_MAP.get(int(row['month']), 'jan')
        months = [m.strip().lower() for m in interval.split(',')]
        return int(current_month in months)

    df['promo2_active_month'] = df.apply(active_month, axis=1)

    df['promo2_since_months'] = (
        (df['year'] - df['Promo2SinceYear']) * 12 + (df['month'] - 1)
    ).where(df['Promo2'] == 1, 0)
    df.loc[df['Promo2SinceYear'].isna() | (df['Promo2SinceYear'] == -1), 'promo2_since_months'] = 0
    return df


def add_lag_features(df: pd.DataFrame, lags=(7, 14, 28), windows=(7, 28)) -> pd.DataFrame:
    df = df.sort_values(['Store', 'Date']).copy()
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby('Store')['Sales'].transform(lambda s: s.shift(lag))
    for window in windows:
        df[f'sales_rolling_mean_{window}'] = (
            df.groupby('Store')['Sales']
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
    return df


def build_feature_dataset(project_root: Path) -> pd.DataFrame:
    df = load_feature_input(project_root)
    df = add_calendar_features(df)
    df = add_competition_features(df)
    df = add_promo2_features(df)
    df = add_lag_features(df)

    # Columns used for model training; we keep target and remove unused identifiers
    feature_cols = [
        'Store', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
        'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionDistance_missing',
        'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth_missing',
        'CompetitionOpenSinceYear_missing', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear',
        'Promo2SinceWeek_missing', 'Promo2SinceYear_missing', 'PromoInterval',
        'year', 'month', 'day', 'day_of_week', 'week_of_year', 'is_month_start', 'is_month_end',
        'is_weekend', 'competition_open', 'competition_months_since_open',
        'promo2_active_month', 'promo2_since_months', 'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
        'sales_rolling_mean_7', 'sales_rolling_mean_28'
    ]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = pd.NA

    output_path = project_root / 'data' / 'processed' / 'train_features.parquet'
    df.to_parquet(output_path, index=False)
    return df


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    df = build_feature_dataset(project_root)
    print('Feature dataset shape:', df.shape)
    print('Columns:', list(df.columns))
    print('Missing values:\n', df.isna().sum().sort_values(ascending=False).head(20).to_string())

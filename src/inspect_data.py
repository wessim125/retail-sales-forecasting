from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parents[1]
data_dir = base / 'data' / 'raw'

train = pd.read_csv(data_dir / 'train.csv')
store = pd.read_csv(data_dir / 'store.csv')

print('train shape:', train.shape)
print('store shape:', store.shape)
print('\ntrain dtypes:\n', train.dtypes)
print('\nstore dtypes:\n', store.dtypes)
print('\ntrain head:\n', train.head().to_string(index=False))
print('\nstore head:\n', store.head().to_string(index=False))
print('\nmissing train:\n', train.isna().sum().to_string())
print('\nmissing store:\n', store.isna().sum().to_string())
print('\nunique StateHoliday train:', sorted(train['StateHoliday'].unique().tolist()))
print('\nunique Open train:', sorted(train['Open'].unique().tolist()))
print('\nrows Open=0:', int((train['Open'] == 0).sum()))
print('\nrows Open=1 & Sales=0:', int(((train['Open'] == 1) & (train['Sales'] == 0)).sum()))
print('\ntrain date min/max:', train['Date'].min(), train['Date'].max())

# Merge to inspect missing rows in store
merged = train.merge(store, on='Store', how='left')
print('\nmerged rows with missing store info:', int(merged['StoreType'].isna().sum()))
print('\nstore fields missing in merged:\n', merged.isna().sum().to_string())

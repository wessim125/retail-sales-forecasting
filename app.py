from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


@st.cache_data
def load_data() -> pd.DataFrame:
    path = Path(__file__).resolve().parent / 'data' / 'processed' / 'train_features.parquet'
    df = pd.read_parquet(path)
    df = df.sort_values('Date').copy()
    df['Date'] = pd.to_datetime(df['Date'])
    return df


@st.cache_resource
def load_model() -> object:
    model_path = Path(__file__).resolve().parent / 'models' / 'lightgbm_model.joblib'
    return joblib.load(model_path)


st.set_page_config(page_title='Rossmann Sales Forecasting', layout='wide')

st.title('Rossmann Sales Forecasting Dashboard')
st.caption('Business dashboard for sales prediction and model performance review')

feature_df = load_data()
model = load_model()

with st.sidebar:
    st.header('Filters')
    selected_store = st.selectbox('Store', sorted(feature_df['Store'].unique().tolist()))
    selected_year = st.selectbox('Year', sorted(feature_df['Date'].dt.year.unique().tolist()))

filtered = feature_df[(feature_df['Store'] == selected_store) & (feature_df['Date'].dt.year == selected_year)].copy()
filtered = filtered[filtered['Open'] == 1].copy()

if filtered.empty:
    st.warning('No open-store data available for this selection.')
    st.stop()

summary = filtered.groupby(filtered['Date'].dt.to_period('M'))['Sales'].agg(['sum', 'mean'])
summary.index = summary.index.astype(str)

st.subheader('Sales overview')
col1, col2, col3 = st.columns(3)
col1.metric('Average daily sales', f"€{filtered['Sales'].mean():,.0f}")
col2.metric('Total sales', f"€{filtered['Sales'].sum():,.0f}")
col3.metric('Promo days', int(filtered['Promo'].sum()))

st.subheader('Monthly sales summary')
st.bar_chart(summary['sum'])

st.subheader('Sales distribution by day of week')
day_sales = filtered.groupby('DayOfWeek')['Sales'].mean().sort_index()
st.line_chart(day_sales)

st.subheader('Recent sales trend')
trend = filtered[['Date', 'Sales']].sort_values('Date')
st.line_chart(trend.set_index('Date'))

st.subheader('Model-ready feature preview')
preview_cols = ['Date', 'Store', 'Sales', 'Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek', 'sales_lag_7', 'sales_rolling_mean_7']
st.dataframe(filtered[preview_cols].head(10), use_container_width=True)

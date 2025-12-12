import pandas as pd
import numpy as np

# заполнение пропусков медианой для логистической регрессии
def fill_median(df):
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=np.number).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    return df_clean

# заполнение пропусков специальным значением (-1) для градиентного бустинга
def fill_special(df):
    df_special = df.copy()
    df_special = df_special.fillna(-1)
    return df_special

# обработка выбросов (capping) - установка предела для выбросов
def limit_outliers(df, columns, quantile=0.99):
    df_new = df.copy()
    for col in columns:
        limit = df_new[col].quantile(quantile)
        df_new.loc[df_new[col] > limit, col] = limit
    return df_new

# обработка верхнего предела возраста
def clean_age(df, UPPER_AGE_LIMIT = 90):
    df_clean = df.copy()
    df_clean.loc[df_clean['age'] > UPPER_AGE_LIMIT, 'age'] = UPPER_AGE_LIMIT
    return df_clean

# обработка выбросов в DebtRatio
def process_debt_ratio(df, CRITICAL_VALUE=50):
    df_proc = df.copy()
    df_proc['DebtRatio_Flag'] = (df_proc['DebtRatio'] > CRITICAL_VALUE).astype(int)
    df_proc.loc[df_proc['DebtRatio'] > CRITICAL_VALUE, 'DebtRatio'] = CRITICAL_VALUE
    return df_proc

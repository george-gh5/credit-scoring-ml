import pandas as pd
import numpy as np


def limit_outliers(df, columns, quantile=0.99):
    """Обработка выбросов через capping"""
    df_clean = df.copy()
    for col in columns:
        limit = df_clean[col].quantile(quantile)
        df_clean.loc[df_clean[col] > limit, col] = limit
    return df_clean


def clean_age(df, upper_age_limit=90):
    """Обработка верхнего предела возраста"""
    df_clean = df.copy()
    df_clean.loc[df_clean['age'] > upper_age_limit, 'age'] = upper_age_limit
    return df_clean


def process_debt_ratio(df, critical_value=10):
    """Создание флага для выбросов и ограничение DebtRatio"""
    df_proc = df.copy()
    df_proc['DebtRatio_Flag'] = (df_proc['DebtRatio'] > critical_value).astype(int)
    df_proc.loc[df_proc['DebtRatio'] > critical_value, 'DebtRatio'] = critical_value
    return df_proc

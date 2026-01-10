import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def clean_age(df, upper_age_limit=90):
    """Обработка верхнего предела возраста"""
    df_clean = df.copy()
    df_clean.loc[df_clean['age'] > upper_age_limit, 'age'] = upper_age_limit
    return df_clean


def clean_utilization(df, cap_value=2):
    """Hard-ограничение для RevolvingUtilizationOfUnsecuredLines"""
    df_clean = df.copy()
    df_clean.loc[df_clean['RevolvingUtilizationOfUnsecuredLines'] > cap_value, 'RevolvingUtilizationOfUnsecuredLines'] = cap_value
    return df_clean


def process_debt_ratio(df, critical_value=10):
    """Создание флага для выбросов и ограничение DebtRatio"""
    df_proc = df.copy()
    df_proc['DebtRatio_Flag'] = (df_proc['DebtRatio'] > critical_value).astype(int)
    df_proc.loc[df_proc['DebtRatio'] > critical_value, 'DebtRatio'] = critical_value
    return df_proc


def plot_with_outliers(df, col):
    """Создание гистограммы и box-plot для распределений с выбросами"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    sns.histplot(df[col], bins=50, kde=True, ax=axes[0])
    sns.boxplot(x=df[col], ax=axes[1])
    plt.show()

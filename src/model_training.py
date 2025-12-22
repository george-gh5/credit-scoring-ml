import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier
import lightgbm as lgb

from xgboost import XGBClassifier
import xgboost as xgb


def logreg_pipeline():
    """Пайплайн для логистической регрессии"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            random_state=42,
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000
        ))
    ])

def rf_pipeline():
    """Пайплайн для рандомного леса"""
    return Pipeline([
        # заполнение пропусков уникальным значением для отдельной ветки
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('model', RandomForestClassifier(
            random_state=42,
            class_weight='balanced'
        ))
    ])


def lgb_model():
    """Модель LightGBM"""
    return lgb.LGBMClassifier(
        class_weight='balanced',
        random_state=42,
        importance_type='gain',  # важность признаков
        verbosity=-1
    )


def xgb_model():
    """Модель XGBoost"""
    return xgb.XGBClassifier(
        # вес (отношение негативных к позитивным)
        scale_pos_weight=14,
        random_state=42,
        eval_metric='auc'
    )


def get_param_grids():
    """Параметры для GridSearchCV"""
    return {
        'logreg': {
            'imputer__strategy': ['median', 'mean'],
            'model__C': [0.01, 0.1, 1, 10],
        },

        'rf': {
            'model__n_estimators': [100, 300],
            'model__max_depth': [3, 5],
            'model__min_samples_leaf': [50]
        },

        'lgb': {
            'max_depth': [3, 7],
            'num_leaves': [15, 31],
            'learning_rate': [0.03, 0.1],
            'n_estimators': [500]
        },

        'xgb': {
            'max_depth': [2, 4, 6],
            'learning_rate': [0.03, 0.1],
            'n_estimators': [500]
        }
    }

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import json
import joblib
import pathlib

from src.config import MODELS_PATH


def evaluate_metrics(y_true, y_pred_proba):
    """Расчёт метрик, возвращает словарь из ROC-AUC и Gini"""
    metrics = {}

    roc_auc = roc_auc_score(y_true, y_pred_proba)
    gini = roc_auc * 2 - 1
    metrics['roc_auc'] = round(roc_auc, 4)
    metrics['gini'] = round(gini, 4)

    return metrics


def save_model_results(name, model, params, metrics):
    """Сохранение результатов модели"""
    joblib.dump(model, MODELS_PATH / f'{name}.pkl')
    with open(MODELS_PATH / f'{name}_params.json', 'w') as f:
        json.dump(params, f, indent=4)
    with open(MODELS_PATH / f'{name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)


def display_results(models_path=MODELS_PATH):
    """Таблица результатов"""
    all_results = []
    for file in models_path.glob("*_metrics.json"):
        with open(file, 'r') as f:
            res = json.load(f)
            res['model_name'] = file.stem.replace('_metrics', '')
            all_results.append(res)

    df_results = pd.DataFrame(all_results).set_index('model_name')
    df_results = df_results.sort_values(by='roc_auc', ascending=False)

    return df_results

import numpy as np
import pandas as pd

# Regression Metrics
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr

# Classification Metrics
from sklearn.metrics import (
    f1_score, accuracy_score, 
    precision_score, recall_score, 
    roc_auc_score, matthews_corrcoef
)

def get_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Args:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values

    Returns:
        dict: Dictionary containing RMSE, R2, and PCC
    """
    metrics = {
        'mae':mean_absolute_error(y_true, y_pred),
        'mse':mean_squared_error(y_true, y_pred),
        'rmse':np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2':r2_score(y_true, y_pred),
        'pcc':pearsonr(y_true, y_pred)[0],
        'srcc':spearmanr(y_true, y_pred)[0]
    }
    
    return metrics

def get_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Args:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values

    Returns:
        dict: Dictionary containing F1-score, Accuracy, Precision, Recall, AUROC
    """
    metrics = {
        'f1':f1_score(y_true, y_pred),
        'acc':accuracy_score(y_true, y_pred),
        'precision':precision_score(y_true, y_pred),
        'recall':recall_score(y_true, y_pred),
        'aucroc':roc_auc_score(y_true, y_pred),
        'mcc':matthews_corrcoef(y_true, y_pred)
    }
    
    return metrics

def _aggregate_regression_metrics(df_ys: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df_ys (pd.DataFrame): concatenated predictions across folds

    Returns:
        pd.DataFrame: aggregated metrics
    """
    results = []
    model_names = [c for c in df_ys.columns if c not in ["GroundTruth", "Fold"]]

    for model_name in model_names:
        fold_metrics = []
        for fold in sorted(df_ys["Fold"].unique()):
            fd = df_ys[df_ys["Fold"] == fold]
            met = get_regression_metrics(fd["GroundTruth"], fd[model_name])
            fold_metrics.append(met)

        results.append({
            "model_name": model_name,
            "mae": float(np.mean([m["mae"] for m in fold_metrics])),
            "mae_std": float(np.std([m["mae"] for m in fold_metrics])),
            "mse": float(np.mean([m["mse"] for m in fold_metrics])),
            "mse_std": float(np.std([m["mse"] for m in fold_metrics])),
            "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
            "rmse_std": float(np.std([m["rmse"] for m in fold_metrics])),
            "r2": float(np.mean([m["r2"] for m in fold_metrics])),
            "r2_std": float(np.std([m["r2"] for m in fold_metrics])),
            "pcc": float(np.mean([m["pcc"] for m in fold_metrics])),
            "pcc_std": float(np.std([m["pcc"] for m in fold_metrics])),
            "srcc": float(np.mean([m["srcc"] for m in fold_metrics])),
            "srcc_std": float(np.std([m["srcc"] for m in fold_metrics])),
        })

    return pd.DataFrame(results)

def get_metric(y_true, y_pred, task_type, y_prob=None):
    if task_type == 'classification':
        if y_prob is not None:
             # If y_prob is provided, we might want to use it for AUC.
             # But the current get_classification_metrics only takes y_pred.
             # We will pass y_pred as is, assuming it contains class labels.
             # Note: roc_auc_score in get_classification_metrics will essentially calculate AUC on hard labels
             # unless we modify it. For now, following existing structure.
             pass
        return get_classification_metrics(y_true, y_pred)
    else:
        return get_regression_metrics(y_true, y_pred)

def _aggregate_classification_metrics(df_ys: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df_ys (pd.DataFrame): concatenated predictions across folds

    Returns:
        pd.DataFrame: aggregated metrics
    """
    results = []
    model_names = [c for c in df_ys.columns if c not in ["GroundTruth", "Fold"]]

    for model_name in model_names:
        fold_metrics = []
        for fold in sorted(df_ys["Fold"].unique()):
            fd = df_ys[df_ys["Fold"] == fold]
            met = get_classification_metrics(fd["GroundTruth"], fd[model_name])
            fold_metrics.append(met)

        results.append({
            "model_name": model_name,
            "f1": float(np.mean([m["f1"] for m in fold_metrics])),
            "f1_std": float(np.std([m["f1"] for m in fold_metrics])),
            "acc": float(np.mean([m["acc"] for m in fold_metrics])),
            "acc_std": float(np.std([m["acc"] for m in fold_metrics])),
            "precision": float(np.mean([m["precision"] for m in fold_metrics])),
            "precision_std": float(np.std([m["precision"] for m in fold_metrics])),
            "recall": float(np.mean([m["recall"] for m in fold_metrics])),
            "recall_std": float(np.std([m["recall"] for m in fold_metrics])),
            "aucroc": float(np.mean([m["aucroc"] for m in fold_metrics])),
            "aucroc_std": float(np.std([m["aucroc"] for m in fold_metrics])),
            "mcc": float(np.mean([m["mcc"] for m in fold_metrics])),
            "mcc_std": float(np.std([m["mcc"] for m in fold_metrics])),
        })

    return pd.DataFrame(results)
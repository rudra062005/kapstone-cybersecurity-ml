from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor, prepare_data

logger = logging.getLogger(__name__)

MODELS_DIR = Path('models')
REPORTS_DIR = Path('reports')
LABEL_COL = 'label'

def build_pipeline(df_columns, n_estimators=100, random_state=42, class_weight='balanced'):
    preprocessor = build_preprocessor(df_columns)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )
    return Pipeline([('preprocessor', preprocessor), ('classifier', clf)])

def train(train_df, test_df, n_estimators=100, random_state=42):
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    logger.info('Preparing training data ...')
    X_train, y_train = prepare_data(train_df, LABEL_COL)
    X_test,  y_test  = prepare_data(test_df,  LABEL_COL)
    logger.info('Train shape: %s  Test shape: %s', X_train.shape, X_test.shape)
    logger.info('Train class dist: %s', y_train.value_counts().to_dict())
    logger.info('Test  class dist: %s', y_test.value_counts().to_dict())

    # Baseline
    logger.info('Training dummy baseline ...')
    baseline = DummyClassifier(strategy='most_frequent', random_state=random_state)
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    base_acc = accuracy_score(y_test, base_pred)
    logger.info('Baseline accuracy: %.4f', base_acc)

    # Main model
    logger.info('Building Random Forest pipeline ...')
    pipeline = build_pipeline(list(X_train.columns), n_estimators=n_estimators, random_state=random_state)
    logger.info('Training Random Forest ...')
    pipeline.fit(X_train, y_train)

    # Evaluation
    logger.info('Evaluating model ...')
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:,1]

    metrics = {
        'baseline_accuracy': round(float(base_acc), 4),
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1_score':  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'n_train': int(len(X_train)),
        'n_test':  int(len(X_test)),
        'n_features': int(X_train.shape[1]),
        'dataset': 'UNSW-NB15',
        'target_col': LABEL_COL,
        'model': 'RandomForestClassifier',
        'evaluation_split': 'official_UNSW_NB15_train_test',
    }
    cm = metrics['confusion_matrix']
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    metrics['true_negatives']  = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives']  = tp
    metrics['fp_rate'] = round(fp/(fp+tn) if (fp+tn)>0 else 0, 4)
    metrics['fn_rate'] = round(fn/(fn+tp) if (fn+tp)>0 else 0, 4)

    report_str = classification_report(y_test, y_pred)
    logger.info('Classification report:\n%s', report_str)

    # Feature importance
    clf = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    try:
        feat_names = preprocessor.get_feature_names_out()
        importances = clf.feature_importances_
        fi = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
        feature_importance = [{'feature': f, 'importance': round(float(v),6)} for f,v in fi]
        metrics['top_10_features'] = feature_importance[:10]
    except Exception as e:
        logger.warning('Could not extract feature importances: %s', e)
        feature_importance = []

    # Save artifacts
    pipeline_path = MODELS_DIR / 'pipeline.joblib'
    joblib.dump(pipeline, pipeline_path)
    logger.info('Pipeline saved to %s', pipeline_path)

    metrics_path = REPORTS_DIR / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info('Metrics saved to %s', metrics_path)

    fi_path = REPORTS_DIR / 'feature_importance.json'
    with open(fi_path, 'w') as f:
        json.dump(feature_importance, f, indent=2)

    return pipeline, metrics, y_pred, y_prob

def load_pipeline(path=None):
    path = path or (MODELS_DIR / 'pipeline.joblib')
    return joblib.load(path)

def load_metrics(path=None):
    path = path or (REPORTS_DIR / 'metrics.json')
    with open(path) as f:
        return json.load(f)

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    from src.data_loader import load_dataset
    train_df, test_df = load_dataset()
    pipeline, metrics, y_pred, y_prob = train(train_df, test_df)
    print('Accuracy:', metrics['accuracy'])
    print('F1 Score:', metrics['f1_score'])
    print('Confusion Matrix:', metrics['confusion_matrix'])

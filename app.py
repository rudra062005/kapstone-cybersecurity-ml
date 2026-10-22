'''
app.py  -  Entry point: trains model and runs the complete pipeline.
Also serves as the Streamlit dashboard launcher.
'''
from __future__ import annotations
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def train_pipeline():
    from src.data_loader import load_dataset, profile_dataset, validate_schema
    from src.train_model import train
    import json
    from pathlib import Path

    logger.info('=== Phase 1: Loading dataset ===')
    train_df, test_df = load_dataset()

    logger.info('=== Phase 1a: Schema validation ===')
    warnings = validate_schema(train_df, 'train') + validate_schema(test_df, 'test')
    for w in warnings:
        logger.warning(w)

    logger.info('=== Phase 1b: Dataset profiling ===')
    profile = profile_dataset(train_df)
    logger.info('Train rows: %d | Attack rate: %.2f%%', profile['n_rows'], profile.get('attack_rate_pct',0))
    Path('reports').mkdir(exist_ok=True)
    with open('reports/dataset_profile.json','w') as f:
        json.dump({'train': profile, 'test': profile_dataset(test_df)}, f, indent=2)

    logger.info('=== Phase 2: Training model ===')
    pipeline, metrics, y_pred, y_prob = train(train_df, test_df)

    logger.info('=== Training Complete ===')
    logger.info('Accuracy:  %.4f', metrics['accuracy'])
    logger.info('Precision: %.4f', metrics['precision'])
    logger.info('Recall:    %.4f', metrics['recall'])
    logger.info('F1 Score:  %.4f', metrics['f1_score'])
    logger.info('Model saved to: models/pipeline.joblib')
    logger.info('Metrics saved to: reports/metrics.json')
    return pipeline, metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kapstone Security Detection System')
    parser.add_argument('--train', action='store_true', help='Train the ML model')
    parser.add_argument('--api',   action='store_true', help='Start the FastAPI server')
    parser.add_argument('--dashboard', action='store_true', help='Start the Streamlit dashboard')
    args = parser.parse_args()

    if args.train:
        train_pipeline()
    elif args.api:
        import uvicorn
        uvicorn.run('api:app', host='0.0.0.0', port=8000, reload=False)
    elif args.dashboard:
        import subprocess
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard.py'])
    else:
        train_pipeline()

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    LABEL_COL, ATTACK_CAT_COL, CATEGORICAL_COLS, NUMERIC_COLS,
    load_dataset, validate_schema, profile_dataset, row_to_event, dataframe_to_events,
)
from src.preprocessing import build_preprocessor, prepare_data
from src.detection import validate_event, assign_risk_level, get_recommendations
from src.correlation import correlate_events, _make_incident
from src.compliance import get_compliance_mapping, map_incident_compliance, COMPLIANCE_MAPPINGS
from src.reporting import generate_security_report

# ============================================================
# Fixtures
# ============================================================

TRAIN_PATH = Path('data/UNSW_NB15_training-set.csv')
TEST_PATH  = Path('data/UNSW_NB15_testing-set.csv')
SAMPLE_FRAC = 0.005   # tiny sample for fast testing

@pytest.fixture(scope='session')
def raw_dataframes():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        pytest.skip('Dataset files not found.')
    train, test = load_dataset(sample_frac=SAMPLE_FRAC)
    return train, test

@pytest.fixture(scope='session')
def sample_event():
    return {
        'event_id': 'test-001',
        'timestamp': '2023-01-01T00:00:00+00:00',
        'source_ip': '0.0.0.0',
        'destination_ip': '0.0.0.0',
        'asset_id': 'ASSET-1',
        'event_type': 'DoS',
        'severity': 'high',
        'model_prediction': 1,
        'anomaly_score': 0.92,
        'supporting_evidence': {'proto': 'tcp', 'sbytes': 1000, 'dbytes': 0},
    }

@pytest.fixture(scope='session')
def trained_pipeline(raw_dataframes):
    from src.train_model import train
    train_df, test_df = raw_dataframes
    pipeline, metrics, y_pred, y_prob = train(train_df, test_df)
    return pipeline, metrics

# ============================================================
# 1. Dataset loading and schema validation
# ============================================================

class TestDataLoading:
    def test_files_exist(self):
        assert TRAIN_PATH.exists(), 'Training CSV missing'
        assert TEST_PATH.exists(),  'Testing CSV missing'

    def test_load_returns_dataframes(self, raw_dataframes):
        train, test = raw_dataframes
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert len(train) > 0
        assert len(test)  > 0

    def test_label_column_present(self, raw_dataframes):
        train, test = raw_dataframes
        assert LABEL_COL in train.columns
        assert LABEL_COL in test.columns

    def test_label_is_binary(self, raw_dataframes):
        train, _ = raw_dataframes
        unique = set(train[LABEL_COL].dropna().unique())
        assert unique.issubset({0, 1, 0.0, 1.0}), f'Unexpected label values: {unique}'

    def test_attack_cat_present(self, raw_dataframes):
        train, _ = raw_dataframes
        assert ATTACK_CAT_COL in train.columns

    def test_validate_schema_no_critical_warnings(self, raw_dataframes):
        train, test = raw_dataframes
        w = validate_schema(train, 'train') + validate_schema(test, 'test')
        critical = [x for x in w if 'Missing label' in x]
        assert not critical, f'Critical schema warnings: {critical}'

    def test_profile_dataset(self, raw_dataframes):
        train, _ = raw_dataframes
        prof = profile_dataset(train)
        assert 'n_rows' in prof
        assert 'class_distribution' in prof
        assert prof['attack_rate_pct'] >= 0

# ============================================================
# 2. Preprocessing
# ============================================================

class TestPreprocessing:
    def test_prepare_data_drops_leakage_cols(self, raw_dataframes):
        train, _ = raw_dataframes
        X, y = prepare_data(train)
        assert LABEL_COL not in X.columns
        assert ATTACK_CAT_COL not in X.columns

    def test_X_shape(self, raw_dataframes):
        train, _ = raw_dataframes
        X, y = prepare_data(train)
        assert X.shape[0] == len(train)
        assert X.shape[1] > 0

    def test_preprocessor_transforms(self, raw_dataframes):
        train, _ = raw_dataframes
        X, y = prepare_data(train)
        prep = build_preprocessor(list(X.columns))
        X_t = prep.fit_transform(X)
        assert X_t.shape[0] == len(X)

# ============================================================
# 3. Model training and inference
# ============================================================

class TestModel:
    def test_train_returns_pipeline(self, trained_pipeline):
        pipeline, metrics = trained_pipeline
        assert pipeline is not None

    def test_metrics_keys(self, trained_pipeline):
        _, metrics = trained_pipeline
        for k in ['accuracy','precision','recall','f1_score','confusion_matrix']:
            assert k in metrics, f'Missing metric: {k}'

    def test_metrics_values_in_range(self, trained_pipeline):
        _, metrics = trained_pipeline
        for k in ['accuracy','precision','recall','f1_score']:
            v = metrics[k]
            assert 0.0 <= v <= 1.0, f'{k}={v} out of range'

    def test_pipeline_predict(self, raw_dataframes, trained_pipeline):
        _, test_df = raw_dataframes
        pipeline, _ = trained_pipeline
        from src.preprocessing import prepare_data
        X_test, _ = prepare_data(test_df)
        preds = pipeline.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, raw_dataframes, trained_pipeline):
        _, test_df = raw_dataframes
        pipeline, _ = trained_pipeline
        from src.preprocessing import prepare_data
        X_test, _ = prepare_data(test_df)
        probs = pipeline.predict_proba(X_test)[:, 1]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

# ============================================================
# 4. Detection logic
# ============================================================

class TestDetection:
    def test_validate_event_valid(self, sample_event):
        errs = validate_event(sample_event)
        assert errs == []

    def test_validate_event_missing_fields(self):
        bad = {'event_id': 'x'}
        errs = validate_event(bad)
        assert len(errs) > 0

    def test_assign_risk_level(self):
        assert assign_risk_level({'anomaly_score': 0.85}) == 'critical'
        assert assign_risk_level({'anomaly_score': 0.65}) == 'high'
        assert assign_risk_level({'anomaly_score': 0.45}) == 'medium'
        assert assign_risk_level({'anomaly_score': 0.25}) == 'low'
        assert assign_risk_level({'anomaly_score': 0.05}) == 'info'
        assert assign_risk_level({'anomaly_score': None}) == 'unknown'

    def test_get_recommendations_dos(self):
        recs = get_recommendations({'event_type': 'DoS', 'anomaly_score': 0.9})
        assert any('rate limit' in r.lower() or 'DDoS' in r for r in recs)

    def test_get_recommendations_default(self):
        recs = get_recommendations({'event_type': 'Normal', 'anomaly_score': 0.1})
        assert len(recs) > 0

# ============================================================
# 5. Alert correlation
# ============================================================

class TestCorrelation:
    def test_no_flagged_events_returns_empty(self):
        events = [{'flagged': False, 'event_id': '1', 'source_ip': '1.2.3.4', 'asset_id': 'ASSET-1',
                   'event_type': 'Normal', 'anomaly_score': 0.1, 'supporting_evidence': {}}]
        incs = correlate_events(events)
        assert incs == []

    def test_flagged_events_create_incidents(self):
        events = [
            {'flagged': True, 'event_id': str(i), 'source_ip': '1.2.3.4', 'asset_id': 'ASSET-1',
             'event_type': 'DoS', 'anomaly_score': 0.9, 'supporting_evidence': {}}
            for i in range(3)
        ]
        incs = correlate_events(events, min_alerts_per_incident=2)
        assert len(incs) >= 1
        total_alerts = sum(inc['alert_count'] for inc in incs)
        assert total_alerts == 3

    def test_incident_risk_levels(self):
        events = [
            {'flagged': True, 'event_id': '1', 'source_ip': '1.2.3.4', 'asset_id': 'ASSET-1',
             'event_type': 'Exploitation', 'anomaly_score': 0.95, 'supporting_evidence': {}}
        ]
        incs = correlate_events(events, min_alerts_per_incident=1)
        assert incs[0]['risk_level'] == 'critical'

# ============================================================
# 6. Compliance mapping
# ============================================================

class TestCompliance:
    def test_dos_mapping_exists(self):
        m = get_compliance_mapping('DoS')
        assert m['matched_type'] == 'DoS'
        assert len(m['frameworks']) > 0

    def test_all_known_types_have_mappings(self):
        for event_type in COMPLIANCE_MAPPINGS:
            m = get_compliance_mapping(event_type)
            assert m is not None

    def test_unknown_type_returns_default(self):
        m = get_compliance_mapping('RandomUnknownType12345')
        assert m is not None
        assert 'disclaimer' in m

    def test_map_incident_compliance(self):
        incident = {
            'incident_id': 'INC-test',
            'risk_level': 'high',
            'event_types': ['DoS'],
            'affected_assets': ['ASSET-1'],
            'supporting_evidence': {},
        }
        result = map_incident_compliance(incident)
        assert 'compliance_frameworks' in result
        assert len(result['compliance_frameworks']) > 0

# ============================================================
# 7. Report generation
# ============================================================

class TestReporting:
    def test_generate_report_keys(self, sample_event):
        flagged_event = dict(sample_event)
        flagged_event['flagged'] = True
        flagged_event['flag_reason'] = 'Test flag'
        incident = {
            'incident_id': 'INC-test',
            'alert_count': 1,
            'event_types': ['DoS'],
            'affected_assets': ['ASSET-1'],
            'source_ips': ['0.0.0.0'],
            'risk_level': 'high',
            'max_anomaly_score': 0.92,
            'alert_ids': ['test-001'],
            'supporting_evidence': {},
            'recommendations': ['Test recommendation'],
            'timestamp': '2023-01-01T00:00:00+00:00',
        }
        metrics = {'accuracy': 0.95, 'f1_score': 0.94, 'precision': 0.93, 'recall': 0.95,
                   'baseline_accuracy': 0.56, 'n_train': 1000, 'n_test': 250,
                   'confusion_matrix': [[100, 5],[3, 142]], 'model': 'RandomForestClassifier',
                   'dataset': 'UNSW-NB15', 'target_col': 'label',
                   'false_positives': 5, 'false_negatives': 3, 'fp_rate': 0.05, 'fn_rate': 0.02}
        comp = [map_incident_compliance(incident)]
        report = generate_security_report([flagged_event], [incident], comp, metrics)
        assert 'executive_summary' in report
        assert 'model_evaluation' in report
        assert 'detected_threats' in report
        assert 'correlated_incidents' in report
        assert 'compliance_mappings' in report

    def test_report_executive_summary(self, sample_event):
        flagged = dict(sample_event, flagged=True, flag_reason='Test')
        report = generate_security_report([flagged], [], [], {})
        es = report['executive_summary']
        assert es['total_events_analysed'] == 1
        assert es['events_flagged_as_suspicious'] == 1

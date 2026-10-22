'''
api.py  -  FastAPI backend for the Kapstone Security Detection System.

Endpoints:
  GET  /health    - application and model status
  GET  /metrics   - actual saved model evaluation metrics
  POST /analyze   - validate events, run detection, correlate, return report
  GET  /incidents - retrieve stored incidents (in-memory for this prototype)
'''
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import (
    AnalyzeRequest, AnalyzeResponse, HealthResponse, MetricsResponse, SecurityEvent
)
from src.compliance import map_incident_compliance
from src.correlation import correlate_events, summarise_incidents
from src.detection import build_events_from_df, flag_suspicious, validate_event
from src.reporting import generate_security_report, save_report_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR  = Path('models')
REPORTS_DIR = Path('reports')
PIPELINE_PATH = MODELS_DIR / 'pipeline.joblib'
METRICS_PATH  = REPORTS_DIR / 'metrics.json'

# --------------------------------------------------------------------------
from contextlib import asynccontextmanager

_pipeline = None
_metrics: Dict[str, Any] = {}
_stored_incidents: List[Dict] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _metrics
    if PIPELINE_PATH.exists():
        try:
            import joblib
            _pipeline = joblib.load(PIPELINE_PATH)
            logger.info('Pipeline loaded from %s', PIPELINE_PATH)
        except Exception as e:
            logger.error('Failed to load pipeline: %s', e)
    else:
        logger.warning('Model pipeline not found at %s. Run train_model.py first.', PIPELINE_PATH)
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
        logger.info('Metrics loaded: accuracy=%.4f, f1=%.4f', _metrics.get('accuracy',0), _metrics.get('f1_score',0))
    else:
        logger.warning('Metrics file not found at %s.', METRICS_PATH)
    yield

# --------------------------------------------------------------------------
# Application setup
# --------------------------------------------------------------------------
app = FastAPI(
    title='Kapstone Security Detection API',
    description='AI/ML-powered cybersecurity threat detection using UNSW-NB15 dataset.',
    version='1.0.0',
    contact={'name': 'Kapstone AI/ML Developer Intern Assessment'},
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)

# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get('/health', response_model=HealthResponse, tags=['System'])
def health_check():
    return HealthResponse(
        status='healthy',
        model_loaded=_pipeline is not None,
        dataset='UNSW-NB15',
    )

@app.get('/metrics', response_model=MetricsResponse, tags=['Model'])
def get_metrics():
    if not _metrics:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Metrics not available. Train the model first: python src/train_model.py',
        )
    return MetricsResponse(
        accuracy=_metrics.get('accuracy'),
        precision=_metrics.get('precision'),
        recall=_metrics.get('recall'),
        f1_score=_metrics.get('f1_score'),
        baseline_accuracy=_metrics.get('baseline_accuracy'),
        n_train=_metrics.get('n_train'),
        n_test=_metrics.get('n_test'),
        confusion_matrix=_metrics.get('confusion_matrix'),
        model=_metrics.get('model'),
        dataset=_metrics.get('dataset'),
        target_col=_metrics.get('target_col'),
        top_10_features=_metrics.get('top_10_features'),
    )

@app.post('/analyze', tags=['Detection'])
def analyze_events(request: AnalyzeRequest):
    if not request.events:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail='No events provided.')
    raw_dicts = [ev.model_dump() for ev in request.events]
    validation_errors = []
    for i, ev in enumerate(raw_dicts):
        errs = validate_event(ev)
        if errs:
            validation_errors.append({'event_index': i, 'errors': errs})
    if validation_errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={'message': 'Event validation failed', 'errors': validation_errors},
        )
    df = pd.DataFrame(raw_dicts)
    events = build_events_from_df(df, pipeline=_pipeline, threshold=request.threshold)
    incidents = correlate_events(events)
    compliance_maps = [map_incident_compliance(inc) for inc in incidents]
    _stored_incidents.extend(incidents)
    report = generate_security_report(events, incidents, compliance_maps, _metrics)
    save_report_json(report)
    flagged = [e for e in events if e.get('flagged')]
    return {
        'total_events': len(events),
        'flagged_count': len(flagged),
        'incidents': incidents,
        'compliance_maps': compliance_maps,
        'summary': report.get('executive_summary', {}),
        'events': events[:50],
    }

@app.get('/incidents', tags=['Incidents'])
def get_incidents(limit: int = 50):
    if not _stored_incidents:
        return {'incidents': [], 'total': 0, 'message': 'No incidents stored yet. Call POST /analyze first.'}
    return {
        'incidents': _stored_incidents[-limit:],
        'total': len(_stored_incidents),
    }

@app.get('/dataset/profile', tags=['Dataset'])
def dataset_profile():
    from src.data_loader import load_dataset, profile_dataset
    try:
        train_df, test_df = load_dataset(sample_frac=0.01)
        return {
            'train_profile': profile_dataset(train_df),
            'test_profile':  profile_dataset(test_df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------------
# Task 4: Data Connectors Endpoints
# --------------------------------------------------------------------------
from src.connectors import CsvLogConnector, ThreatIntelConnector, AssessmentFindingConnector

_csv_connector = CsvLogConnector('data/UNSW_NB15_testing-set.csv')
_intel_connector = ThreatIntelConnector()
_vuln_connector = AssessmentFindingConnector()

@app.get('/connectors/sources', tags=['Connectors'])
def get_connectors():
    '''List all registered security data connectors and their connection status.'''
    return {
        'connectors': [
            _csv_connector.get_status(),
            _intel_connector.get_status(),
            _vuln_connector.get_status(),
        ]
    }

@app.get('/connectors/threat-intel/lookup/{ip}', tags=['Connectors'])
def lookup_threat_intel(ip: str):
    '''Query threat intelligence reputation and MITRE ATT&CK mapping for an IP address.'''
    return _intel_connector.lookup_ip(ip)

@app.post('/connectors/ingest', tags=['Connectors'])
def ingest_from_connector(
    source: str = 'csv',
    limit: int = 50,
    offset: int = 0,
    threshold: float = 0.5,
):
    '''
    Retrieve security logs or findings directly from a configured connector,
    run ML detection and alert correlation, map to compliance, and return results.
    '''
    if source == 'csv':
        try:
            records = _csv_connector.fetch_records(limit=limit, offset=offset)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'CSV connector error: {e}')
        df = pd.DataFrame(records)
        events = build_events_from_df(df, pipeline=_pipeline, threshold=threshold)
    elif source == 'threat_intel':
        records = _intel_connector.fetch_records(limit=limit)
        events = []
        for r in records:
            ev = {
                'event_id': f"IOC-{r['indicator']}",
                'timestamp': '2026-10-22T10:00:00Z',
                'source_ip': r['indicator'],
                'asset_id': f"ASSET-{r['indicator'].replace('.', '-')}",
                'event_type': r.get('threat_types', ['Network-Threat'])[0] if r.get('threat_types') else 'Threat-Intel-Hit',
                'severity': 'critical' if r.get('reputation') == 'malicious' else 'medium',
                'flagged': True,
                'anomaly_score': r.get('confidence', 0.9),
                'flag_reason': f"Threat Intel Hit: Actor={r.get('actor')} Risk={r.get('risk_score')}",
                'supporting_evidence': {
                    'actor': r.get('actor'),
                    'mitre_tactics': r.get('mitre_tactics'),
                    'risk_score': r.get('risk_score'),
                },
            }
            events.append(ev)
    elif source == 'vulnerability':
        findings = _vuln_connector.fetch_records(limit=limit)
        events = []
        for f in findings:
            ev = {
                'event_id': f.get('finding_id', 'VULN-001'),
                'timestamp': '2026-10-22T10:00:00Z',
                'source_ip': f.get('source_ip', '0.0.0.0'),
                'asset_id': f.get('asset_id', 'SERVER-01'),
                'event_type': f.get('event_type', 'Vulnerability'),
                'severity': f.get('severity', 'high'),
                'flagged': True,
                'anomaly_score': 0.85,
                'flag_reason': f"Assessment Finding: {f.get('vulnerability')} ({f.get('cve')})",
                'supporting_evidence': f,
            }
            events.append(ev)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source '{source}'. Supported: 'csv', 'threat_intel', 'vulnerability'",
        )

    incidents = correlate_events(events)
    compliance_maps = [map_incident_compliance(inc) for inc in incidents]
    _stored_incidents.extend(incidents)
    report = generate_security_report(events, incidents, compliance_maps, _metrics)
    save_report_json(report)

    flagged = [e for e in events if e.get('flagged')]
    return {
        'source': source,
        'total_ingested': len(events),
        'flagged_count': len(flagged),
        'incidents': incidents,
        'compliance_maps': compliance_maps,
        'summary': report.get('executive_summary', {}),
        'sample_events': events[:20],
    }


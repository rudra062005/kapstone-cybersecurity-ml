from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.data_loader import row_to_event, NUMERIC_COLS, CATEGORICAL_COLS

logger = logging.getLogger(__name__)
REQUIRED_EVENT_FIELDS = ['event_id','timestamp','source_ip','event_type','severity']

def validate_event(event):
    return [f'Missing field: {f}' for f in REQUIRED_EVENT_FIELDS if f not in event]

def flag_suspicious(events, pipeline, threshold=0.5):
    rows = [{**ev.get('supporting_evidence',{}), **ev} for ev in events]
    df = pd.DataFrame(rows)
    feat_num = [c for c in NUMERIC_COLS if c in df.columns]
    feat_cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    feat = feat_num + feat_cat
    if not feat:
        for ev in events:
            ev.update({'model_prediction':None,'anomaly_score':None,'flagged':False})
        return events
    try:
        probs = pipeline.predict_proba(df[feat])[:,1]
        preds = (probs >= threshold).astype(int)
    except Exception as e:
        logger.error('Model inference failed: %s', e)
        for ev in events:
            ev.update({'model_prediction':None,'anomaly_score':None,'flagged':False})
        return events
    for i, ev in enumerate(events):
        ev.update({'model_prediction':int(preds[i]),'anomaly_score':round(float(probs[i]),4),'flagged':bool(preds[i]==1)})
        if ev['flagged']:
            sc = ev['anomaly_score']
            ev['flag_reason'] = f'ML attack probability {sc:.2%} >= {threshold:.0%}. POTENTIAL THREAT requiring human investigation.'
    return events

def build_events_from_df(df, pipeline=None, threshold=0.5):
    events = []
    num = [c for c in NUMERIC_COLS if c in df.columns]
    cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    feat = num + cat
    if pipeline and feat:
        try:
            probs = pipeline.predict_proba(df[feat])[:,1]
            preds = (probs >= threshold).astype(int)
        except Exception as e:
            logger.error('Batch inference failed: %s', e)
            probs = np.full(len(df), np.nan)
            preds = np.zeros(len(df), dtype=int)
    else:
        probs = np.full(len(df), np.nan)
        preds = np.zeros(len(df), dtype=int)
    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        isnan = probs[i] != probs[i]
        pred = int(preds[i]) if not isnan else None
        prob = float(probs[i]) if not isnan else None
        ev = row_to_event(row, model_prediction=pred, anomaly_score=prob)
        ev['flagged'] = bool(pred==1) if pred is not None else False
        if ev.get('flagged'):
            ev['flag_reason'] = f'ML prob {prob:.2%}. Potential threat requiring investigation.'
        events.append(ev)
    return events

def assign_risk_level(event):
    s = event.get('anomaly_score')
    if s is None: return 'unknown'
    if s >= 0.80: return 'critical'
    if s >= 0.60: return 'high'
    if s >= 0.40: return 'medium'
    if s >= 0.20: return 'low'
    return 'info'

def get_recommendations(event):
    recs = []
    t = event.get('event_type','')
    s = event.get('anomaly_score',0) or 0
    if s >= 0.80:
        recs += ['Isolate affected asset immediately.','Notify SOC/IR team.']
    if 'Exploit' in t or 'Shellcode' in t:
        recs.append('Patch vulnerable services; review exploit signatures.')
    if 'DoS' in t:
        recs.append('Enable rate limiting and upstream DDoS protection.')
    if 'Backdoor' in t:
        recs.append('Check for persistence mechanisms; rotate credentials.')
    if 'Reconnaissance' in t:
        recs.append('Block scanning source IPs; tighten firewall rules.')
    if 'Fuzzers' in t:
        recs.append('Review input validation on exposed service endpoints.')
    if not recs:
        recs.append('Monitor asset; collect further evidence before escalating.')
    return recs

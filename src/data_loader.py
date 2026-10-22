
from __future__ import annotations
import logging, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path('data')
TRAIN_FILE = DATA_DIR / 'UNSW_NB15_training-set.csv'
TEST_FILE  = DATA_DIR / 'UNSW_NB15_testing-set.csv'
LABEL_COL      = 'label'
ATTACK_CAT_COL = 'attack_cat'
CATEGORICAL_COLS = ['proto', 'service', 'state']
NUMERIC_COLS = [
    'dur','spkts','dpkts','sbytes','dbytes','rate',
    'sttl','dttl','sload','dload','sloss','dloss',
    'sinpkt','dinpkt','sjit','djit',
    'swin','stcpb','dtcpb','dwin',
    'tcprtt','synack','ackdat','smean','dmean',
    'trans_depth','response_body_len',
    'ct_srv_src','ct_state_ttl','ct_dst_ltm',
    'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm',
    'is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd',
    'ct_src_ltm','ct_srv_dst','is_sm_ips_ports',
]

def load_dataset(train_path=TRAIN_FILE, test_path=TEST_FILE, sample_frac=None, random_state=42):
    train_path, test_path = Path(train_path), Path(test_path)
    if not train_path.exists(): raise FileNotFoundError(f'Training set not found: {train_path}')
    if not test_path.exists():  raise FileNotFoundError(f'Testing set not found: {test_path}')
    logger.info('Loading training set ...')
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df  = pd.read_csv(test_path,  low_memory=False)
    train_df.columns = [c.strip().lower() for c in train_df.columns]
    test_df.columns  = [c.strip().lower() for c in test_df.columns]
    logger.info('Raw shapes -- train: %s  test: %s', train_df.shape, test_df.shape)
    if sample_frac and 0 < sample_frac < 1:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)
    return train_df, test_df

def validate_schema(df, name='dataset'):
    w = []
    if LABEL_COL not in df.columns: w.append(f'[{name}] Missing label column')
    if ATTACK_CAT_COL not in df.columns: w.append(f'[{name}] Missing attack_cat column')
    return w

def profile_dataset(df):
    profile = {
        'n_rows': int(len(df)), 'n_cols': int(len(df.columns)),
        'columns': list(df.columns),
        'dtypes': {c: str(df[c].dtype) for c in df.columns},
        'missing_values': {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum()>0},
        'duplicate_rows': int(df.duplicated().sum()),
    }
    if LABEL_COL in df.columns:
        vc = df[LABEL_COL].value_counts().to_dict()
        profile['class_distribution'] = {str(k): int(v) for k,v in vc.items()}
        profile['attack_rate_pct'] = round(df[LABEL_COL].mean()*100,2)
    if ATTACK_CAT_COL in df.columns:
        vc2 = df[ATTACK_CAT_COL].value_counts().to_dict()
        profile['attack_category_distribution'] = {str(k): int(v) for k,v in vc2.items()}
    return profile

def row_to_event(row, model_prediction=None, anomaly_score=None, asset_prefix='ASSET'):
    def _g(col, default=None):
        val = row.get(col, default)
        if val is None or (isinstance(val, float) and np.isnan(val)): return default
        return val
    row_id = _g('id','unknown')
    event_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f'unswnb15-{row_id}-{anomaly_score}'))
    ts = datetime(2023,1,1,0,0,0,tzinfo=timezone.utc).isoformat()
    cat = _g('attack_cat','Normal')
    event_type = str(cat).strip() if cat not in (None,'','Normal','-') else ('Normal' if model_prediction==0 else 'Potential_Attack')
    if anomaly_score is not None:
        if   anomaly_score>=0.80: severity='critical'
        elif anomaly_score>=0.60: severity='high'
        elif anomaly_score>=0.40: severity='medium'
        elif anomaly_score>=0.20: severity='low'
        else:                      severity='info'
    else:
        severity = 'high' if model_prediction==1 else 'info'
    ev_fields=['proto','service','state','sbytes','dbytes','sttl','dttl','dur','rate','ct_state_ttl']
    evidence={f: _g(f) for f in ev_fields if _g(f) is not None}
    return {
        'event_id':event_id,'timestamp':ts,
        'source_ip':_g('srcip','0.0.0.0'),'destination_ip':_g('dstip','0.0.0.0'),
        'asset_id':f'{asset_prefix}-{row_id}','event_type':event_type,'severity':severity,
        'model_prediction':model_prediction,
        'anomaly_score':round(float(anomaly_score),4) if anomaly_score is not None else None,
        'supporting_evidence':evidence,'attack_cat':cat,
        'label_ground_truth':int(_g('label',-1)),'_data_source':'UNSW-NB15-CSV',
        '_note':'srcip/dstip/timestamps absent from Kaggle CSVs; 0.0.0.0 is a documented placeholder.',
    }

def dataframe_to_events(df, predictions=None, probabilities=None, max_events=5000):
    df = df.head(max_events).reset_index(drop=True)
    events=[]
    for i, row in df.iterrows():
        pred=int(predictions[i]) if predictions is not None else None
        prob=float(probabilities[i]) if probabilities is not None else None
        events.append(row_to_event(row,model_prediction=pred,anomaly_score=prob))
    return events

class CSVDataSource:
    def __init__(self,train_path=TRAIN_FILE,test_path=TEST_FILE,sample_frac=None):
        self.train_path=Path(train_path);self.test_path=Path(test_path);self.sample_frac=sample_frac
    def fetch(self,split='train',**kw):
        t,e=load_dataset(self.train_path,self.test_path,self.sample_frac)
        return e if split=='test' else t

class LiveEventStub:
    def fetch(self,**kw):
        raise NotImplementedError('LiveEventStub: connect to SIEM API, normalise with row_to_event(), return DataFrame.')

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

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
DROP_COLS = ['id','attack_cat','label']

def get_feature_cols(df_columns):
    num = [c for c in NUMERIC_COLS if c in df_columns]
    cat = [c for c in CATEGORICAL_COLS if c in df_columns]
    return num, cat

def build_preprocessor(df_columns):
    num_cols, cat_cols = get_feature_cols(df_columns)
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])
    transformers = []
    if num_cols: transformers.append(('num', numeric_pipeline, num_cols))
    if cat_cols: transformers.append(('cat', categorical_pipeline, cat_cols))
    return ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)

def prepare_data(df, label_col='label'):
    drop = [c for c in DROP_COLS if c in df.columns]
    y = df[label_col].astype(int)
    X = df.drop(columns=drop, errors='ignore')
    return X, y

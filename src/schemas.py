from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

class SecurityEvent(BaseModel):
    '''Normalised security event submitted to /analyze.'''
    event_id: Optional[str] = None
    timestamp: Optional[str] = None
    source_ip: Optional[str] = '0.0.0.0'
    destination_ip: Optional[str] = '0.0.0.0'
    asset_id: Optional[str] = None
    event_type: Optional[str] = None
    severity: Optional[str] = None
    # Raw UNSW-NB15 numeric features (subset used by model)
    dur:      Optional[float] = None
    proto:    Optional[str]   = None
    service:  Optional[str]   = None
    state:    Optional[str]   = None
    spkts:    Optional[float] = None
    dpkts:    Optional[float] = None
    sbytes:   Optional[float] = None
    dbytes:   Optional[float] = None
    rate:     Optional[float] = None
    sttl:     Optional[float] = None
    dttl:     Optional[float] = None
    sload:    Optional[float] = None
    dload:    Optional[float] = None
    sloss:    Optional[float] = None
    dloss:    Optional[float] = None
    sinpkt:   Optional[float] = None
    dinpkt:   Optional[float] = None
    sjit:     Optional[float] = None
    djit:     Optional[float] = None
    swin:     Optional[float] = None
    stcpb:    Optional[float] = None
    dtcpb:    Optional[float] = None
    dwin:     Optional[float] = None
    tcprtt:   Optional[float] = None
    synack:   Optional[float] = None
    ackdat:   Optional[float] = None
    smean:    Optional[float] = None
    dmean:    Optional[float] = None
    trans_depth:       Optional[float] = None
    response_body_len: Optional[float] = None
    ct_srv_src:        Optional[float] = None
    ct_state_ttl:      Optional[float] = None
    ct_dst_ltm:        Optional[float] = None
    ct_src_dport_ltm:  Optional[float] = None
    ct_dst_sport_ltm:  Optional[float] = None
    ct_dst_src_ltm:    Optional[float] = None
    is_ftp_login:      Optional[float] = None
    ct_ftp_cmd:        Optional[float] = None
    ct_flw_http_mthd:  Optional[float] = None
    ct_src_ltm:        Optional[float] = None
    ct_srv_dst:        Optional[float] = None
    is_sm_ips_ports:   Optional[float] = None
    label:      Optional[int]   = None
    attack_cat: Optional[str]   = None

class AnalyzeRequest(BaseModel):
    events: List[SecurityEvent] = Field(..., min_length=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class AnalyzeResponse(BaseModel):
    total_events:     int
    flagged_count:    int
    incidents:        List[Dict[str, Any]]
    compliance_maps:  List[Dict[str, Any]]
    events:           List[Dict[str, Any]]
    summary:          Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    dataset: str

class MetricsResponse(BaseModel):
    accuracy:          Optional[float]
    precision:         Optional[float]
    recall:            Optional[float]
    f1_score:          Optional[float]
    baseline_accuracy: Optional[float]
    n_train:           Optional[int]
    n_test:            Optional[int]
    confusion_matrix:  Optional[List]
    model:             Optional[str]
    dataset:           Optional[str]
    target_col:        Optional[str]
    top_10_features:   Optional[List]

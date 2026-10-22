import pytest
from fastapi.testclient import TestClient
from api import app

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c

def test_api_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'healthy'
    assert data['dataset'] == 'UNSW-NB15'

def test_api_metrics(client):
    r = client.get('/metrics')
    assert r.status_code == 200
    data = r.json()
    assert 'accuracy' in data
    assert 'f1_score' in data
    assert data['accuracy'] > 0.8

def test_api_incidents_empty_or_list(client):
    r = client.get('/incidents')
    assert r.status_code == 200
    data = r.json()
    assert 'incidents' in data

def test_api_analyze_valid(client):
    payload = {
        'events': [
            {
                'event_id': 'EVT-API-01',
                'timestamp': '2026-10-22T10:00:00Z',
                'source_ip': '192.168.1.10',
                'destination_ip': '10.0.0.5',
                'asset_id': 'SERVER-WEB-01',
                'event_type': 'DoS',
                'dur': 0.1,
                'proto': 'tcp',
                'service': 'http',
                'state': 'FIN',
                'spkts': 10,
                'dpkts': 8,
                'sbytes': 1500,
                'dbytes': 2000,
                'rate': 25.0,
                'sttl': 64,
                'dttl': 64,
                'sload': 1000.0,
                'dload': 2000.0,
                'sloss': 0,
                'dloss': 0,
                'sinpkt': 10.0,
                'dinpkt': 10.0,
                'sjit': 0.1,
                'djit': 0.1,
                'swin': 255,
                'stcpb': 100,
                'dtcpb': 100,
                'dwin': 255,
                'tcprtt': 0.05,
                'synack': 0.02,
                'ackdat': 0.03,
                'smean': 150,
                'dmean': 250,
                'trans_depth': 1,
                'response_body_len': 500,
                'ct_srv_src': 2,
                'ct_state_ttl': 1,
                'ct_dst_ltm': 1,
                'ct_src_dport_ltm': 1,
                'ct_dst_sport_ltm': 1,
                'ct_dst_src_ltm': 1,
                'is_ftp_login': 0,
                'ct_ftp_cmd': 0,
                'ct_flw_http_mthd': 1,
                'ct_src_ltm': 1,
                'ct_srv_dst': 2,
                'is_sm_ips_ports': 0,
            }
        ],
        'threshold': 0.5,
    }
    r = client.post('/analyze', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data['total_events'] == 1
    assert 'incidents' in data
    assert 'compliance_maps' in data

def test_api_analyze_empty_payload(client):
    r = client.post('/analyze', json={'events': []})
    assert r.status_code == 422

def test_api_connectors_sources(client):
    r = client.get('/connectors/sources')
    assert r.status_code == 200
    data = r.json()
    assert 'connectors' in data
    assert len(data['connectors']) >= 3

def test_api_threat_intel_lookup(client):
    r = client.get('/connectors/threat-intel/lookup/175.45.176.0')
    assert r.status_code == 200
    data = r.json()
    assert data['found'] is True
    assert data['reputation'] == 'malicious'

def test_api_connector_ingest_csv(client):
    r = client.post('/connectors/ingest?source=csv&limit=5')
    assert r.status_code == 200
    data = r.json()
    assert data['source'] == 'csv'
    assert data['total_ingested'] == 5

def test_api_connector_ingest_threat_intel(client):
    r = client.post('/connectors/ingest?source=threat_intel&limit=3')
    assert r.status_code == 200
    data = r.json()
    assert data['source'] == 'threat_intel'
    assert data['total_ingested'] == 3

def test_api_connector_ingest_invalid_source(client):
    r = client.post('/connectors/ingest?source=invalid_source_123')
    assert r.status_code == 400

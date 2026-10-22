import pytest
from pathlib import Path
from src.connectors import CsvLogConnector, ThreatIntelConnector, AssessmentFindingConnector

def test_csv_log_connector_connect():
    conn = CsvLogConnector('data/UNSW_NB15_testing-set.csv')
    assert conn.connect() is True
    status = conn.get_status()
    assert status['connected'] is True
    assert status['source_type'] == 'csv_log_file'

def test_csv_log_connector_fetch_records():
    conn = CsvLogConnector('data/UNSW_NB15_testing-set.csv')
    records = conn.fetch_records(limit=5)
    assert len(records) == 5
    assert 'label' in records[0] or 'id' in records[0]

def test_csv_log_connector_missing_file():
    conn = CsvLogConnector('data/non_existent_file.csv')
    assert conn.connect() is False
    with pytest.raises(FileNotFoundError):
        conn.fetch_records(limit=5)

def test_threat_intel_connector_lookup():
    conn = ThreatIntelConnector()
    assert conn.connect() is True
    
    # Known malicious IOC
    res_known = conn.lookup_ip('175.45.176.0')
    assert res_known['found'] is True
    assert res_known['reputation'] == 'malicious'
    assert 'TA0001 Initial Access' in res_known['mitre_tactics']
    
    # Subnet match
    res_subnet = conn.lookup_ip('175.45.176.99')
    assert res_subnet['found'] is True
    assert res_subnet['reputation'] == 'suspicious'
    
    # Unknown IP
    res_unknown = conn.lookup_ip('8.8.8.8')
    assert res_unknown['found'] is False
    assert res_unknown['reputation'] == 'neutral'

def test_threat_intel_connector_enrich_event():
    conn = ThreatIntelConnector()
    event = {
        'event_id': 'TEST-01',
        'source_ip': '175.45.176.0',
        'severity': 'low',
        'flagged': False,
    }
    enriched = conn.enrich_event(event)
    assert enriched['severity'] == 'critical'
    assert enriched['flagged'] is True
    assert 'Threat Intel match' in enriched['flag_reason']

def test_assessment_finding_connector():
    conn = AssessmentFindingConnector()
    assert conn.connect() is True
    findings = conn.fetch_records(limit=2)
    assert len(findings) == 2
    assert 'vulnerability' in findings[0]
    assert 'severity' in findings[0]

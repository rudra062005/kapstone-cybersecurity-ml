'''
demo.py  -  Live End-to-End Prototype Demonstration Script

Demonstrates the complete Kapstone cybersecurity workflow in a single command:
1. Ingests network flow logs via CsvLogConnector
2. Enriches events using ThreatIntelConnector (MITRE ATT&CK tactics)
3. Evaluates flows using the trained Random Forest pipeline
4. Correlates suspicious alerts into contextual incidents with explainable reasoning
5. Maps incidents to NIST CSF 2.0 and ISO/IEC 27001:2022 controls
6. Generates and exports executive security reports
'''
from __future__ import annotations

import json
import logging
from pathlib import Path
import pandas as pd
import joblib

from src.connectors import CsvLogConnector, ThreatIntelConnector, AssessmentFindingConnector
from src.detection import build_events_from_df
from src.correlation import correlate_events, summarise_incidents
from src.compliance import map_incident_compliance
from src.reporting import generate_security_report, save_report_json, save_incidents_csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('KapstoneDemo')

def run_demonstration(sample_limit: int = 25, threshold: float = 0.5):
    print('=' * 80)
    print('  KAPSTONE TECHNOLOGICAL SERVICES - AI/ML SECURITY PROTOTYPE DEMO')
    print('=' * 80)

    # 1. Load Model Pipeline & Metrics
    model_path = Path('models/pipeline.joblib')
    metrics_path = Path('reports/metrics.json')

    if not model_path.exists():
        logger.error('Model not found at %s. Please run: python app.py --train', model_path)
        return

    logger.info('Loading trained Random Forest pipeline...')
    pipeline = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else {}

    print(f"\n[1] LOADED MODEL EVALUATION METRICS (UNSW-NB15 Official Test Set):")
    print(f"    * Accuracy:  {metrics.get('accuracy', 0):.2%}")
    print(f"    * Recall:    {metrics.get('recall', 0):.2%} (High sensitivity minimizes missed attacks)")
    print(f"    * Precision: {metrics.get('precision', 0):.2%}")
    print(f"    * F1-Score:  {metrics.get('f1_score', 0):.2%}")
    print(f"    * FN Rate:   {metrics.get('fn_rate', 0):.2%} (Critical false negative rate)")

    # 2. Ingest Logs via CsvLogConnector
    print(f"\n[2] DATA CONNECTOR INGESTION (Task 4):")
    csv_connector = CsvLogConnector('data/UNSW_NB15_testing-set.csv')
    csv_connector.connect()
    raw_records = csv_connector.fetch_records(limit=sample_limit)
    print(f"    * Retrieved {len(raw_records)} network flow records via CsvLogConnector.")

    # 3. Model Inference & Detection
    print(f"\n[3] THREAT DETECTION & RISK SCORING (Task 1 & 2):")
    df_raw = pd.DataFrame(raw_records)
    events = build_events_from_df(df_raw, pipeline=pipeline, threshold=threshold)
    flagged = [e for e in events if e.get('flagged')]
    print(f"    * Processed {len(events)} events | Flagged as Suspicious: {len(flagged)} ({len(flagged)/len(events):.1%})")

    # 4. Threat Intelligence Enrichment
    ti_connector = ThreatIntelConnector()
    print(f"\n[4] THREAT INTELLIGENCE ENRICHMENT:")
    enriched_events = [ti_connector.enrich_event(ev) for ev in events]
    print(f"    * Enriched {len(enriched_events)} events with Threat Intel reputation & MITRE ATT&CK tactics.")

    # 5. Alert Correlation Engine
    print(f"\n[5] ALERT CORRELATION & INCIDENT SYNTHESIS (Task 2):")
    incidents = correlate_events(enriched_events)
    print(f"    * Correlated {len(flagged)} alerts into {len(incidents)} distinct security incidents.")
    
    for idx, inc in enumerate(incidents[:3], 1):
        print(f"\n    Incident #{idx}: {inc['incident_id']} [{inc['risk_level'].upper()}]")
        print(f"      - Alert Count:      {inc['alert_count']}")
        print(f"      - Attack Types:     {', '.join(inc['event_types'])}")
        print(f"      - Target Asset(s):  {', '.join(inc['affected_assets'][:3])}")
        print(f"      - Max Anomaly Prob: {inc['max_anomaly_score']:.2%}")
        print(f"      - Reasoning:        {inc.get('correlation_reason', 'N/A')}")
        print(f"      - Recommendations:  {'; '.join(inc['recommendations'][:2])}")

    # 6. Compliance Mapping
    print(f"\n[6] COMPLIANCE MAPPING (Task 3 - NIST CSF 2.0 & ISO/IEC 27001:2022):")
    compliance_maps = [map_incident_compliance(inc) for inc in incidents]
    for idx, cm in enumerate(compliance_maps[:2], 1):
        print(f"\n    Compliance Finding #{idx} for {cm['incident_id']}:")
        for fw in cm.get('compliance_frameworks', []):
            print(f"      * [{fw['framework']}] {fw['control_id']}: {fw['control_name']}")
            print(f"        Rationale: {fw['rationale']}")

    # 7. Reporting & Export
    print(f"\n[7] SECURITY REPORT GENERATION (Task 5):")
    report = generate_security_report(enriched_events, incidents, compliance_maps, metrics)
    json_path = save_report_json(report, filename='demo_security_report.json')
    csv_path = save_incidents_csv(incidents, filename='demo_incidents.csv')
    print(f"    * Complete JSON Security Report saved to: {json_path}")
    print(f"    * Correlated Incidents CSV exported to:   {csv_path}")

    print('\n' + '=' * 80)
    print('  DEMONSTRATION COMPLETED SUCCESSFULLY (PROTOTYPE FULLY OPERATIONAL)')
    print('=' * 80)

if __name__ == '__main__':
    run_demonstration()

import csv, json, logging, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
REPORTS_DIR = Path('reports')

def _ts():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def generate_security_report(events, incidents, compliance_maps, metrics, dataset_profile=None):
    now = datetime.now(timezone.utc).isoformat()
    flagged = [e for e in events if e.get('flagged')]
    severity_counts = {}
    for ev in flagged:
        sev = ev.get('severity','unknown')
        severity_counts[sev] = severity_counts.get(sev,0) + 1
    risk_summary = {}
    for inc in incidents:
        r = inc.get('risk_level','unknown')
        risk_summary[r] = risk_summary.get(r,0) + 1
    affected_assets = list({a for inc in incidents for a in inc.get('affected_assets',[])})
    all_recs = []
    for inc in incidents:
        all_recs.extend(inc.get('recommendations',[]))
    all_recs = list(dict.fromkeys(all_recs))
    report = {
        'report_metadata': {
            'generated_at': now,
            'report_type': 'Security Threat Detection Report',
            'dataset': 'UNSW-NB15',
            'assessment': 'Kapstone AI/ML Developer Intern Technical Assessment',
            'disclaimer': (
                'This report is generated automatically for assessment purposes. '
                'Detections are model-based probabilities, NOT confirmed incidents. '
                'All findings require human investigation before action is taken. '
                'Compliance mappings are indicative, not a formal audit.'
            ),
        },
        'executive_summary': {
            'total_events_analysed': len(events),
            'events_flagged_as_suspicious': len(flagged),
            'detection_rate_pct': round(len(flagged)/max(len(events),1)*100,2),
            'total_incidents': len(incidents),
            'severity_distribution': severity_counts,
            'incident_risk_distribution': risk_summary,
            'unique_affected_assets': len(affected_assets),
        },
        'dataset_and_methodology': {
            'dataset': 'UNSW-NB15 (Network Intrusion Detection)',
            'source': 'University of New South Wales Canberra Cyber Range Lab',
            'split': 'Official training/testing CSV files',
            'target_column': metrics.get('target_col','label'),
            'model': metrics.get('model','RandomForestClassifier'),
            'features': metrics.get('n_features'),
            'training_samples': metrics.get('n_train'),
            'testing_samples': metrics.get('n_test'),
            'class_imbalance_handling': 'class_weight=balanced in RandomForestClassifier',
            'dataset_profile': dataset_profile or {},
            'limitations': [
                'srcip, dstip, and timestamps absent from Kaggle CSVs; IP-based correlation uses placeholders.',
                'Time-window correlation requires real timestamps not present in this CSV split.',
                'Model trained on network flow data; real deployment needs retraining on live traffic.',
                'False positives and negatives are inherent; human validation required for all findings.',
            ],
        },
        'model_evaluation': {
            'accuracy':          metrics.get('accuracy'),
            'precision':         metrics.get('precision'),
            'recall':            metrics.get('recall'),
            'f1_score':          metrics.get('f1_score'),
            'baseline_accuracy': metrics.get('baseline_accuracy'),
            'confusion_matrix':  metrics.get('confusion_matrix'),
            'false_positives':   metrics.get('false_positives'),
            'false_negatives':   metrics.get('false_negatives'),
            'fp_rate':           metrics.get('fp_rate'),
            'fn_rate':           metrics.get('fn_rate'),
            'top_10_features':   metrics.get('top_10_features',[]),
            'fp_explanation': (
                'False positives (normal traffic classified as attack) increase alert fatigue. '
                'Mitigation: tune threshold, add whitelist rules, collect more normal traffic.'
            ),
            'fn_explanation': (
                'False negatives (attacks missed by model) are the more dangerous error type. '
                'Mitigation: diversify training data, use ensemble approaches, add rule-based detection.'
            ),
        },
        'detected_threats': [
            {
                'event_id': ev.get('event_id'),
                'event_type': ev.get('event_type'),
                'severity': ev.get('severity'),
                'anomaly_score': ev.get('anomaly_score'),
                'asset_id': ev.get('asset_id'),
                'flag_reason': ev.get('flag_reason'),
                'evidence': ev.get('supporting_evidence',{}),
            }
            for ev in flagged[:100]
        ],
        'correlated_incidents': incidents,
        'affected_assets': affected_assets,
        'compliance_mappings': compliance_maps[:20],
        'remediation_recommendations': all_recs,
        'limitations_and_next_steps': [
            'Retrain on labelled live traffic for production deployment.',
            'Integrate a real SIEM connector to replace the CSV data source.',
            'Implement timestamp-based correlation when real timestamps are available.',
            'Expand compliance mapping coverage to all NIST CSF 2.0 subcategories.',
            'Add continuous model retraining and drift detection.',
        ],
    }
    return report

def save_report_json(report, filename=None):
    REPORTS_DIR.mkdir(exist_ok=True)
    filename = filename or ('security_report_' + _ts() + '.json')
    path = REPORTS_DIR / filename
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info('Report saved to %s', path)
    return str(path)

def save_events_csv(events, filename=None):
    REPORTS_DIR.mkdir(exist_ok=True)
    filename = filename or ('events_' + _ts() + '.csv')
    path = REPORTS_DIR / filename
    if not events: return str(path)
    flat = []
    for ev in events:
        row = {k:v for k,v in ev.items() if k != 'supporting_evidence'}
        for k,v in ev.get('supporting_evidence',{}).items():
            row['ev_' + k] = v
        flat.append(row)
    keys = list(flat[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(flat)
    logger.info('Events CSV saved to %s', path)
    return str(path)

def save_incidents_csv(incidents, filename=None):
    REPORTS_DIR.mkdir(exist_ok=True)
    filename = filename or ('incidents_' + _ts() + '.csv')
    path = REPORTS_DIR / filename
    if not incidents: return str(path)
    flat = [{
        'incident_id': inc.get('incident_id'),
        'alert_count': inc.get('alert_count'),
        'risk_level': inc.get('risk_level'),
        'max_anomaly_score': inc.get('max_anomaly_score'),
        'event_types': ','.join(inc.get('event_types',[])),
        'affected_assets': ','.join(inc.get('affected_assets',[])[:10]),
        'source_ips': ','.join(inc.get('source_ips',[])[:10]),
        'recommendations': '; '.join(inc.get('recommendations',[])),
    } for inc in incidents]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
    logger.info('Incidents CSV saved to %s', path)
    return str(path)

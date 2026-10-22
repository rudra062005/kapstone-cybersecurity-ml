from __future__ import annotations
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SEVERITY_WEIGHTS = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'info': 0, 'unknown': 0}

def correlate_events(
    events: List[Dict],
    time_window_seconds: int = 300,
    min_alerts_per_incident: int = 2,
) -> List[Dict]:
    '''
    Group related alert events into potential incidents.

    Correlation rules (applied in order):
    1. Same source_ip within the time window
    2. Same asset_id within the time window
    3. Same event_type (attack category) within the time window

    NOTE: Timestamps in UNSW-NB15 training/testing CSVs are absent;
    all events carry the same placeholder timestamp. The time-window
    grouping therefore has no effect for this dataset; grouping is done
    purely by source_ip and asset_id.  This limitation is documented.
    '''
    suspicious = [ev for ev in events if ev.get('flagged')]
    if not suspicious:
        return []

    groups: Dict[str, List[Dict]] = defaultdict(list)
    for ev in suspicious:
        key = (ev.get('source_ip','0.0.0.0'), ev.get('asset_id','unknown'))
        groups[key].append(ev)

    incidents = []
    for key, group_events in groups.items():
        if len(group_events) < min_alerts_per_incident:
            for ev in group_events:
                inc = _make_incident([ev])
                incidents.append(inc)
            continue
        inc = _make_incident(group_events)
        incidents.append(inc)

    incidents.sort(key=lambda x: SEVERITY_WEIGHTS.get(x['risk_level'], 0), reverse=True)
    return incidents

def _make_incident(events: List[Dict]) -> Dict:
    incident_id = str(uuid.uuid4())[:8]
    event_types = list({ev.get('event_type','Unknown') for ev in events})
    asset_ids   = list({ev.get('asset_id','unknown') for ev in events})
    source_ips  = list({ev.get('source_ip','0.0.0.0') for ev in events})

    max_score = max((ev.get('anomaly_score') or 0 for ev in events), default=0)
    if   max_score >= 0.80: risk = 'critical'
    elif max_score >= 0.60: risk = 'high'
    elif max_score >= 0.40: risk = 'medium'
    elif max_score >= 0.20: risk = 'low'
    else:                    risk = 'info'

    all_evidence = {}
    for ev in events:
        all_evidence.update(ev.get('supporting_evidence', {}))

    from src.detection import get_recommendations
    recommendations = []
    for ev in events:
        recommendations.extend(get_recommendations(ev))
    recommendations = list(dict.fromkeys(recommendations))

    # Explicit correlation reasoning explaining why alerts were grouped
    if len(events) > 1:
        corr_reason = (
            f"Correlated {len(events)} suspicious alerts sharing source IP(s) {', '.join(source_ips[:3])} "
            f"and target asset(s) {', '.join(asset_ids[:3])} (attack types: {', '.join(event_types)})."
        )
    else:
        corr_reason = (
            f"Isolated security alert for asset {asset_ids[0] if asset_ids else 'unknown'} "
            f"classified as {event_types[0] if event_types else 'unknown'}."
        )

    return {
        'incident_id': f'INC-{incident_id}',
        'alert_count': len(events),
        'event_types': event_types,
        'affected_assets': asset_ids,
        'source_ips': source_ips,
        'risk_level': risk,
        'max_anomaly_score': round(float(max_score), 4),
        'alert_ids': [ev.get('event_id') for ev in events],
        'correlation_reason': corr_reason,
        'supporting_evidence': all_evidence,
        'recommendations': recommendations,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        '_note': (
            'UNSW-NB15 training/testing CSVs lack real timestamps and IP addresses. '
            'Incident grouping uses asset_id and source_ip placeholders. '
            'Time-window correlation requires real timestamps not present in this dataset split.'
        ),
    }

def summarise_incidents(incidents: List[Dict]) -> Dict:
    summary = {
        'total_incidents': len(incidents),
        'by_risk_level': defaultdict(int),
        'unique_assets': set(),
        'unique_source_ips': set(),
        'event_type_counts': defaultdict(int),
    }
    for inc in incidents:
        summary['by_risk_level'][inc['risk_level']] += 1
        summary['unique_assets'].update(inc.get('affected_assets', []))
        summary['unique_source_ips'].update(inc.get('source_ips', []))
        for et in inc.get('event_types', []):
            summary['event_type_counts'][et] += 1
    summary['by_risk_level'] = dict(summary['by_risk_level'])
    summary['unique_assets'] = list(summary['unique_assets'])
    summary['unique_source_ips'] = list(summary['unique_source_ips'])
    summary['event_type_counts'] = dict(summary['event_type_counts'])
    return summary

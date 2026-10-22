import json, logging
from pathlib import Path
import pandas as pd
import streamlit as st

REPORTS_DIR = Path('reports')
MODELS_DIR  = Path('models')

st.set_page_config(page_title='Kapstone Security Dashboard', page_icon='S', layout='wide')
st.title('Kapstone AI/ML Security Detection Dashboard')
st.caption('UNSW-NB15 Network Intrusion Detection | Assessment Prototype')

st.sidebar.header('Navigation')
page = st.sidebar.radio('Go to', [
    'Dataset Overview', 'Model Metrics', 'Threat Detection',
    'Incidents', 'Compliance Mapping', 'Data Connectors', 'Export Reports',
])

@st.cache_data
def load_metrics():
    p = REPORTS_DIR / 'metrics.json'
    if not p.exists(): return {}
    with open(p) as f: return json.load(f)

@st.cache_data
def load_dataset_profile():
    p = REPORTS_DIR / 'dataset_profile.json'
    if not p.exists(): return {}
    with open(p) as f: return json.load(f)

@st.cache_data
def load_feature_importance():
    p = REPORTS_DIR / 'feature_importance.json'
    if not p.exists(): return []
    with open(p) as f: return json.load(f)

@st.cache_data
def load_latest_report():
    jsons = sorted(REPORTS_DIR.glob('security_report_*.json'), reverse=True)
    if not jsons: return {}
    with open(jsons[0]) as f: return json.load(f)

def model_trained():
    return (MODELS_DIR / 'pipeline.joblib').exists()

# ===== Dataset Overview =====
if page == 'Dataset Overview':
    st.header('Dataset Overview')
    profile = load_dataset_profile()
    if not profile:
        st.warning('Dataset profile not found. Run: python app.py --train')
    else:
        col1, col2 = st.columns(2)
        for col, split, label in [(col1, 'train', 'Training Set'), (col2, 'test', 'Testing Set')]:
            with col:
                st.subheader(label)
                p = profile.get(split, {})
                st.metric('Rows', '{:,}'.format(p.get('n_rows',0)))
                st.metric('Columns', p.get('n_cols', 0))
                st.metric('Attack Rate', '{:.1f}%'.format(p.get('attack_rate_pct',0)))
                st.metric('Duplicate Rows', p.get('duplicate_rows', 0))
                cd = p.get('class_distribution', {})
                if cd:
                    st.write('Class Distribution:')
                    st.dataframe(pd.DataFrame({'Class': list(cd.keys()), 'Count': list(cd.values())}), use_container_width=True)
        tp = profile.get('train', {})
        acd = tp.get('attack_category_distribution', {})
        if acd:
            st.subheader('Attack Category Distribution')
            df_acd = pd.DataFrame({'Category': list(acd.keys()), 'Count': list(acd.values())}).sort_values('Count', ascending=False)
            st.bar_chart(df_acd.set_index('Category'))
        mv = tp.get('missing_values', {})
        st.subheader('Missing Values')
        if mv:
            st.dataframe(pd.DataFrame({'Column': list(mv.keys()), 'Missing': list(mv.values())}), use_container_width=True)
        else:
            st.success('No missing values found in training set.')
        st.info('NOTE: srcip, dstip, sport, dsport, and timestamps are absent from the Kaggle UNSW-NB15 training/testing CSVs.')

# ===== Model Metrics =====
elif page == 'Model Metrics':
    st.header('Model Evaluation Metrics')
    if not model_trained():
        st.error('Model not trained. Run: python app.py --train')
    else:
        m = load_metrics()
        if not m:
            st.warning('Metrics file not found.')
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Accuracy',  '{:.4f}'.format(m.get('accuracy',0)))
            c2.metric('Precision', '{:.4f}'.format(m.get('precision',0)))
            c3.metric('Recall',    '{:.4f}'.format(m.get('recall',0)))
            c4.metric('F1 Score',  '{:.4f}'.format(m.get('f1_score',0)))
            st.divider()
            c5, c6, c7 = st.columns(3)
            c5.metric('Baseline Accuracy', '{:.4f}'.format(m.get('baseline_accuracy',0)))
            c6.metric('False Positive Rate', '{:.4f}'.format(m.get('fp_rate',0)))
            c7.metric('False Negative Rate', '{:.4f}'.format(m.get('fn_rate',0)))
            st.subheader('Confusion Matrix')
            cm = m.get('confusion_matrix', [])
            if cm:
                df_cm = pd.DataFrame(cm, index=['Actual Normal','Actual Attack'], columns=['Pred Normal','Pred Attack'])
                st.dataframe(df_cm, use_container_width=True)
            st.subheader('Top 10 Feature Importances')
            fi = load_feature_importance()
            if fi:
                df_fi = pd.DataFrame(fi[:10])
                st.bar_chart(df_fi.set_index('feature')['importance'])
            st.subheader('Details')
            st.json({'dataset': m.get('dataset'), 'model': m.get('model'), 'target': m.get('target_col'), 'n_train': m.get('n_train'), 'n_test': m.get('n_test')})
            st.info('False Positives (FP): Normal traffic flagged as attack - causes alert fatigue.')
            st.warning('False Negatives (FN): Attacks missed by model - most dangerous error type.')

# ===== Threat Detection =====
elif page == 'Threat Detection':
    st.header('Threat Detection')
    report = load_latest_report()
    if not report:
        st.warning('No security report found. Run the pipeline end-to-end first.')
    else:
        threats = report.get('detected_threats', [])
        st.metric('Threats Detected', len(threats))
        if threats:
            df_t = pd.DataFrame(threats)
            if 'severity' in df_t.columns:
                sev_opts = df_t['severity'].unique().tolist()
                sev_filter = st.multiselect('Filter by severity', options=sev_opts, default=sev_opts)
                df_filtered = df_t[df_t['severity'].isin(sev_filter)]
            else:
                df_filtered = df_t
            show_cols = [c for c in ['event_id','event_type','severity','anomaly_score','asset_id','flag_reason'] if c in df_filtered.columns]
            st.dataframe(df_filtered[show_cols], use_container_width=True)

# ===== Incidents =====
elif page == 'Incidents':
    st.header('Correlated Incidents')
    report = load_latest_report()
    incidents = report.get('correlated_incidents', [])
    if not incidents:
        st.info('No incidents to display. Run pipeline end-to-end first.')
    else:
        st.metric('Total Incidents', len(incidents))
        for inc in incidents[:20]:
            risk = inc.get('risk_level','unknown')
            color_map = {'critical': 'RED', 'high': 'ORANGE', 'medium': 'YELLOW', 'low': 'GREEN'}
            clr = color_map.get(risk, 'GREY')
            title = '{} | Risk: {} | Alerts: {}'.format(inc.get('incident_id','N/A'), risk.upper(), inc.get('alert_count',0))
            with st.expander(title):
                c1, c2 = st.columns(2)
                c1.write('Event Types: ' + ', '.join(inc.get('event_types',[])))
                c1.write('Assets: ' + ', '.join(inc.get('affected_assets',[])[:5]))
                c2.write('Max Score: ' + str(inc.get('max_anomaly_score')))
                c2.write('Source IPs: ' + ', '.join(inc.get('source_ips',[])[:5]))
                recs = inc.get('recommendations', [])
                if recs:
                    st.write('Recommendations:')
                    for rec in recs:
                        st.write('  - ' + rec)
                note = inc.get('_note','')
                if note:
                    st.caption(note)

# ===== Compliance Mapping =====
elif page == 'Compliance Mapping':
    st.header('Compliance Mapping')
    st.info('Indicative mapping only - NIST CSF v2.0 and ISO/IEC 27001:2022. Not a formal audit.')
    report = load_latest_report()
    comp_maps = report.get('compliance_mappings', [])
    if not comp_maps:
        st.warning('No compliance mappings. Run detection first.')
    else:
        for cm in comp_maps[:10]:
            et_str = ' | '.join(cm.get('event_types', []))
            title = '{} | {} | Risk: {}'.format(cm.get('incident_id','N/A'), et_str, cm.get('risk_level','?'))
            with st.expander(title):
                for fw in cm.get('compliance_frameworks', []):
                    st.write('**{}** - {}: {}'.format(fw.get('framework',''), fw.get('control_id',''), fw.get('control_name','')))
                    st.caption('Rationale: ' + fw.get('rationale',''))
                rems = cm.get('remediation', [])
                if rems:
                    st.write('Remediation:')
                    for r in rems:
                        st.write('  - ' + r)

# ===== Data Connectors =====
elif page == 'Data Connectors':
    st.header('Security Data Connectors & Ingestion')
    st.markdown('Integrates modular data connectors to retrieve security logs, threat intelligence, and vulnerability assessment findings.')

    from src.connectors import CsvLogConnector, ThreatIntelConnector, AssessmentFindingConnector
    from src.detection import build_events_from_df
    from src.correlation import correlate_events
    from src.compliance import map_incident_compliance
    from src.reporting import generate_security_report, save_report_json

    csv_conn = CsvLogConnector('data/UNSW_NB15_testing-set.csv')
    ti_conn = ThreatIntelConnector()
    vuln_conn = AssessmentFindingConnector()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader('Log Connector')
        st.write('Source: `UNSW_NB15_testing-set.csv`')
        st.write('Status: Connected' if csv_conn.connect() else 'Status: Disconnected')
    with c2:
        st.subheader('Threat Intel Feed')
        st.write('Source: Simulated OTX / AbuseIPDB')
        st.write('Status: Connected' if ti_conn.connect() else 'Status: Disconnected')
    with c3:
        st.subheader('Vulnerability Connector')
        st.write('Source: Security Assessment Findings')
        st.write('Status: Connected' if vuln_conn.connect() else 'Status: Disconnected')

    st.divider()

    st.subheader('Live Ingestion & Pipeline Analysis')
    sel_source = st.selectbox('Select Data Source', ['CSV Log Feed (UNSW-NB15)', 'Threat Intelligence Feed', 'Vulnerability Assessment Findings'])
    limit = st.slider('Number of records to ingest', min_value=10, max_value=200, value=30, step=10)
    thresh = st.slider('Anomaly / Threat Threshold', min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    if st.button('Fetch Records & Run Analysis Pipeline', type='primary'):
        with st.spinner('Ingesting records and running detection...'):
            pipe = None
            if (MODELS_DIR / 'pipeline.joblib').exists():
                import joblib
                pipe = joblib.load(MODELS_DIR / 'pipeline.joblib')

            if 'CSV' in sel_source:
                records = csv_conn.fetch_records(limit=limit)
                df_ingest = pd.DataFrame(records)
                events = build_events_from_df(df_ingest, pipeline=pipe, threshold=thresh)
            elif 'Threat' in sel_source:
                recs = ti_conn.fetch_records(limit=limit)
                events = [
                    {
                        'event_id': f"IOC-{r['indicator']}",
                        'timestamp': '2026-10-22T10:00:00Z',
                        'source_ip': r['indicator'],
                        'asset_id': f"ASSET-{r['indicator'].replace('.', '-')}",
                        'event_type': r.get('threat_types', ['Network-Threat'])[0] if r.get('threat_types') else 'Threat-Intel',
                        'severity': 'critical' if r.get('reputation') == 'malicious' else 'medium',
                        'flagged': True,
                        'anomaly_score': r.get('confidence', 0.9),
                        'flag_reason': f"Threat Intel Hit: Actor={r.get('actor')}",
                        'supporting_evidence': r,
                    }
                    for r in recs
                ]
            else:
                findings = vuln_conn.fetch_records(limit=limit)
                events = [
                    {
                        'event_id': f.get('finding_id', 'VULN-001'),
                        'timestamp': '2026-10-22T10:00:00Z',
                        'source_ip': f.get('source_ip', '0.0.0.0'),
                        'asset_id': f.get('asset_id', 'SERVER-01'),
                        'event_type': f.get('event_type', 'Vulnerability'),
                        'severity': f.get('severity', 'high'),
                        'flagged': True,
                        'anomaly_score': 0.85,
                        'flag_reason': f"Finding: {f.get('vulnerability')}",
                        'supporting_evidence': f,
                    }
                    for f in findings
                ]

            incidents = correlate_events(events)
            comp_maps = [map_incident_compliance(inc) for inc in incidents]
            report = generate_security_report(events, incidents, comp_maps, load_metrics())
            saved_path = save_report_json(report)

            flagged = [e for e in events if e.get('flagged')]
            st.success(f'Successfully ingested {len(events)} events | Flagged: {len(flagged)} | Correlated Incidents: {len(incidents)}')
            st.info(f'Report automatically generated and saved to: `{saved_path}`')

            if incidents:
                st.write('Correlated Incidents from Ingested Batch:')
                for inc in incidents[:5]:
                    st.write(f"- **{inc['incident_id']}** ({inc['risk_level'].upper()}): {inc.get('correlation_reason', '')}")

    st.divider()
    st.subheader('Threat Intelligence Indicator Lookup')
    lookup_ip = st.text_input('Lookup IP Address', value='175.45.176.0')
    if lookup_ip:
        res = ti_conn.lookup_ip(lookup_ip)
        st.json(res)

# ===== Export =====
elif page == 'Export Reports':
    st.header('Export Reports')
    report = load_latest_report()
    if report:
        st.download_button(
            label='Download Security Report (JSON)',
            data=json.dumps(report, indent=2, default=str),
            file_name='kapstone_security_report.json',
            mime='application/json',
        )
    jsons = sorted(REPORTS_DIR.glob('security_report_*.json'), reverse=True)
    if jsons:
        st.subheader('Available Reports')
        for j in jsons[:10]:
            st.write(j.name + ' ({} KB)'.format(j.stat().st_size//1024))
    all_csvs = sorted(REPORTS_DIR.glob('*.csv'), reverse=True)
    if all_csvs:
        st.subheader('CSV Files')
        for c in all_csvs[:10]:
            try:
                df = pd.read_csv(c)
                st.download_button(
                    label='Download ' + c.name,
                    data=df.to_csv(index=False),
                    file_name=c.name,
                    mime='text/csv',
                )
            except Exception:
                st.write(c.name)
    if not jsons and not all_csvs:
        st.warning('No reports yet. Run: python app.py --train')

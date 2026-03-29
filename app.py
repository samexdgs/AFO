"""
app.py
AFO Clinical Management Platform — dual-dashboard Streamlit application.

Physiotherapist portal: patient registration, clinical assessment, ML recommendation,
cohort overview, prescription export, reassessment scheduling.

Patient portal: AFO information, daily wearing log, satisfaction tracking,
education, and progress visualisation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
import streamlit.components.v1 as components

import store
import clinical
from clinical import AFO_TYPES, LABEL_TO_KEY, FEATURES

st.set_page_config(
    page_title="AFO Clinical Platform",
    page_icon="🦿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Typography and component styles ──────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.page-title {
    font-size: 1.7rem; font-weight: 600; color: #0f172a;
    letter-spacing: -0.02em; margin-bottom: 0.15rem;
}
.page-sub { font-size: 0.88rem; color: #64748b; margin-bottom: 1.4rem; }

.role-badge {
    display: inline-block; font-size: 0.75rem; font-weight: 600;
    padding: 3px 10px; border-radius: 4px; margin-bottom: 6px;
    letter-spacing: 0.04em; text-transform: uppercase;
}
.physio-badge { background: #dbeafe; color: #1d4ed8; }
.patient-badge { background: #dcfce7; color: #15803d; }

.rec-primary {
    border: 2px solid #2563eb; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
    background: #eff6ff;
}
.rec-alt {
    border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    background: white;
}
.rec-name { font-size: 1.05rem; font-weight: 600; color: #0f172a; }
.rec-conf { font-size: 0.82rem; color: #64748b; margin-top: 2px; }

.indicated { font-size: 0.83rem; color: #166534; margin-bottom: 3px; }
.contra    { font-size: 0.83rem; color: #991b1b; margin-bottom: 3px; }

.risk-box {
    background: #fffbeb; border: 1px solid #fde68a; border-left: 4px solid #f59e0b;
    border-radius: 8px; padding: 0.9rem 1rem; margin-bottom: 0.6rem;
}
.risk-title  { font-weight: 600; color: #78350f; font-size: 0.88rem; }
.risk-detail { color: #92400e; font-size: 0.82rem; margin: 3px 0; }
.risk-action { color: #166534; font-size: 0.83rem; background: #f0fdf4;
               border-radius: 5px; padding: 4px 8px; margin-top: 5px; display: block; }

.stat-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 0.9rem 1rem; text-align: center;
}
.stat-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase;
              letter-spacing: 0.06em; margin-bottom: 4px; }
.stat-value { font-size: 1.55rem; font-weight: 600; color: #0f172a; }

.alert-info { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
              padding: 0.7rem 0.9rem; font-size: 0.85rem; color: #1e40af; margin-bottom: 0.5rem; }
.alert-warn { background: #fefce8; border: 1px solid #fef08a; border-radius: 8px;
              padding: 0.7rem 0.9rem; font-size: 0.85rem; color: #854d0e; margin-bottom: 0.5rem; }
.disclaimer { background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 8px;
              padding: 0.7rem 0.9rem; font-size: 0.78rem; color: #475569; margin-top: 1.2rem; }

.patient-row {
    display: flex; justify-content: space-between; align-items: center;
    background: white; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 0.75rem 1rem; margin-bottom: 0.4rem; font-size: 0.86rem;
}
.wearing-on  { color: #16a34a; font-weight: 600; }
.wearing-off { color: #dc2626; font-weight: 600; }

[data-testid="stSidebar"] { background: #f8fafc; }
.stButton > button {
    background: #2563eb; color: white; border: none;
    border-radius: 8px; font-weight: 500; width: 100%;
    font-family: 'IBM Plex Sans', sans-serif;
}
.stButton > button:hover { background: #1d4ed8; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

def _init():
    defaults = {
        "authenticated": False,
        "role": None,
        "user": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


def _logout():
    for k in ["authenticated", "role", "user"]:
        st.session_state[k] = None
    st.session_state["authenticated"] = False
    st.rerun()


# ── Shared charts ─────────────────────────────────────────────────────────────

def _gauge(score: float, title: str) -> go.Figure:
    color = "#16a34a" if score < 4 else ("#f59e0b" if score < 7 else "#dc2626")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 13}},
        number={"font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 10]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, 4],  "color": "#dcfce7"},
                {"range": [4, 7],  "color": "#fef9c3"},
                {"range": [7, 10], "color": "#fee2e2"},
            ],
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def _proba_chart(proba: list) -> go.Figure:
    labels = [AFO_TYPES[LABEL_TO_KEY[i]]["abbreviation"] for i in range(len(proba))]
    colors = ["#2563eb", "#0891b2", "#059669", "#7c3aed", "#dc2626", "#ea580c"]
    fig = go.Figure(go.Bar(
        x=[p * 100 for p in proba],
        y=labels,
        orientation="h",
        marker_color=colors[:len(proba)],
        text=[f"{p:.0%}" for p in proba],
        textposition="outside",
    ))
    fig.update_layout(
        height=240, margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(range=[0, 115], title="Confidence (%)"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def _wearing_chart(logs: list) -> go.Figure | None:
    if len(logs) < 2:
        return None
    df = pd.DataFrame(logs).sort_values("date").tail(30)
    df["worn_n"] = df["worn"].astype(int)
    fig = go.Figure(go.Bar(
        x=pd.to_datetime(df["date"]),
        y=df["worn_n"],
        marker_color=["#16a34a" if w else "#dc2626" for w in df["worn"]],
    ))
    fig.update_layout(
        height=180, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(tickvals=[0, 1], ticktext=["Not worn", "Worn"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def _model_perf_chart(models: dict) -> go.Figure:
    names = list(models.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy", x=names, y=[models[n]["accuracy"] for n in names],
        marker_color="#2563eb",
        text=[f"{models[n]['accuracy']:.1%}" for n in names], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="ROC-AUC", x=names, y=[models[n]["roc_auc"] for n in names],
        marker_color="#059669",
        text=[f"{models[n]['roc_auc']:.3f}" for n in names], textposition="outside",
    ))
    fig.update_layout(
        barmode="group", height=240, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0, 1.15]),
        legend=dict(orientation="h", y=-0.3),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# ── Recommendation renderer ───────────────────────────────────────────────────

def _render_recommendation(label: int, proba: list, assessment: dict, audience: str) -> None:
    top3 = sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)[:3]

    for rank, idx in enumerate(top3):
        key  = LABEL_TO_KEY[idx]
        afo  = AFO_TYPES[key]
        conf = proba[idx]

        card_class = "rec-primary" if rank == 0 else "rec-alt"
        rank_label = {0: "Primary recommendation", 1: "Alternative", 2: "Consider also"}[rank]

        contra_html = ""
        if key == "fes" and assessment.get("has_pacemaker"):
            contra_html = '<div class="contra">⛔ FES CONTRAINDICATED — pacemaker present</div>'
        elif key == "carbon" and assessment.get("mas_ankle", 0) >= 3:
            contra_html = '<div class="contra">⚠ Carbon fibre may be insufficient for spasticity MAS ≥ 3</div>'
        elif key == "grafo" and not assessment.get("crouch_gait"):
            contra_html = '<div class="contra">ℹ GRAFO is specifically for crouch gait — verify indication</div>'

        phases = " · ".join(afo["gait_phases"])

        if audience == "physio":
            indicated = "".join(f'<div class="indicated">✓ {i}</div>' for i in afo["indicated_when"][:3])
            contra_list = "".join(f'<div class="contra">✗ {c}</div>' for c in afo["contraindicated_when"][:2])
            detail = f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px;">
                <div><div style="font-size:0.75rem;font-weight:600;color:#166534;text-transform:uppercase;
                    letter-spacing:.05em;margin-bottom:4px;">Indicated when</div>{indicated}</div>
                <div><div style="font-size:0.75rem;font-weight:600;color:#991b1b;text-transform:uppercase;
                    letter-spacing:.05em;margin-bottom:4px;">Avoid when</div>{contra_list}</div>
            </div>
            <div style="margin-top:8px;font-size:0.81rem;color:#64748b;">
                Gait phases: {phases} · Evidence: {afo['evidence']} · Compliance risk: {afo['compliance_risk']}
            </div>
            <div style="margin-top:4px;font-size:0.81rem;color:#64748b;">
                Cost: {afo['cost_usd']} / {afo['cost_ngn']}
            </div>"""
        else:
            detail = f"""
            <div style="margin-top:10px;font-size:0.86rem;color:#334155;line-height:1.7;">
                {afo['patient_description']}
            </div>
            <div style="margin-top:8px;font-size:0.81rem;color:#64748b;">
                Weight: {afo['weight_class']} · Profile: {afo['profile']}
                · Cost: {afo['cost_usd']} / {afo['cost_ngn']}
            </div>"""

        st.markdown(f"""
        <div class="{card_class}">
            <div style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;
                letter-spacing:.06em;margin-bottom:4px;">{rank_label}</div>
            <div class="rec-name">{afo['name']} ({afo['abbreviation']})</div>
            <div class="rec-conf">Model confidence: {conf:.0%}</div>
            {contra_html}
            {detail}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**Confidence across all device types**")
    st.plotly_chart(_proba_chart(proba), use_container_width=True)


# ── Login screen ──────────────────────────────────────────────────────────────

def _page_login():
    col = st.columns([1, 1.6, 1])[1]
    with col:
        st.markdown('<div class="page-title">🦿 AFO Clinical Platform</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="page-sub">Evidence-based ankle-foot orthosis management '
            'for physiotherapists and patients</div>', unsafe_allow_html=True
        )
        st.markdown("---")

        role = st.radio("Sign in as:", ["Physiotherapist", "Patient"], horizontal=True)
        st.markdown("---")

        username = st.text_input(
            "Physiotherapist username" if role == "Physiotherapist" else "Patient ID"
        ).strip().lower()
        password = st.text_input("Password", type="password")

        if st.button("Sign In"):
            if not username or not password:
                st.error("Enter both fields.")
            elif role == "Physiotherapist":
                user = store.physio_authenticate(username, password)
                if user:
                    st.session_state.update({"authenticated": True, "role": "physio", "user": user})
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
            else:
                user = store.patient_authenticate(username, password)
                if user:
                    st.session_state.update({"authenticated": True, "role": "patient", "user": user})
                    st.rerun()
                else:
                    st.error("Incorrect patient ID or password.")

        st.markdown("---")
        with st.expander("Register a physiotherapist account"):
            _register_physio_form()

        st.markdown("""
        <div class="disclaimer">
        Demo platform. All data stored locally. Recommendations are ML-generated guidance only —
        final AFO selection requires a qualified orthotist or rehabilitation physician.
        Built by Samuel Oluwakoya.
        </div>""", unsafe_allow_html=True)


def _register_physio_form():
    fname  = st.text_input("Full name", key="r_fname")
    clinic = st.text_input("Clinic / hospital", key="r_clinic")
    uname  = st.text_input("Choose username", key="r_uname").strip().lower()
    pw     = st.text_input("Password", type="password", key="r_pw")
    pw2    = st.text_input("Confirm password", type="password", key="r_pw2")
    if st.button("Create account", key="reg_physio"):
        if not all([fname, clinic, uname, pw]):
            st.error("All fields required.")
        elif pw != pw2:
            st.error("Passwords do not match.")
        elif store.physio_register(uname, fname, clinic, pw):
            st.success(f"Account created. Sign in as **{uname}**.")
        else:
            st.error(f"Username **{uname}** already taken.")


# ── Physiotherapist portal ────────────────────────────────────────────────────

def _physio_portal(models: dict, best: str):
    physio = st.session_state["user"]

    with st.sidebar:
        st.markdown(f'<div class="role-badge physio-badge">Physiotherapist</div>', unsafe_allow_html=True)
        st.markdown(f"**{physio['full_name']}**")
        st.markdown(f"*{physio['clinic']}*")
        st.markdown("---")
        nav = st.radio("", ["Patient Panel", "New Assessment", "Register Patient",
                             "Cohort Overview", "Model Performance"],
                       label_visibility="collapsed")
        st.markdown("---")
        if st.button("Sign out"):
            _logout()

    patients = store.patients_for_physio(physio["username"])

    if nav == "Patient Panel":
        _physio_patient_panel(patients, models, best)
    elif nav == "New Assessment":
        _physio_assessment_form(patients, models, best)
    elif nav == "Register Patient":
        _physio_register_patient(physio["username"])
    elif nav == "Cohort Overview":
        _physio_cohort(patients)
    elif nav == "Model Performance":
        _physio_model_perf(models, best)


def _physio_patient_panel(patients: list, models: dict, best: str):
    st.markdown('<div class="page-title">Patient Panel</div>', unsafe_allow_html=True)

    if not patients:
        st.info("No patients registered yet. Use **Register Patient** in the sidebar.")
        return

    options = {f"{p['full_name']} ({p['patient_id']})": p for p in patients}
    selected_label = st.selectbox("Select patient", list(options.keys()))
    patient = options[selected_label]

    st.markdown("---")

    assessment = store.assessment_latest(patient["patient_id"])
    wearing    = store.wearing_log_get(patient["patient_id"])
    worn_count = sum(1 for e in wearing if e.get("worn"))
    total_days = len(wearing)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val in [
        (c1, "Age", patient["age"]),
        (c2, "Diagnosis", patient["diagnosis"]),
        (c3, "Days post-onset", patient["days_post_onset"]),
        (c4, "Wearing compliance", f"{worn_count/total_days:.0%}" if total_days else "—"),
        (c5, "Assessments", len(store.assessments_get(patient["patient_id"]))),
    ]:
        with col:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value" style="font-size:1.1rem;">{val}</div>
            </div>""", unsafe_allow_html=True)

    if assessment:
        st.markdown("---")
        tab_rec, tab_wear, tab_hist = st.tabs(["Latest Recommendation", "Wearing Log", "Assessment History"])

        with tab_rec:
            label, proba = clinical.predict(models, best, {k: assessment.get(k, 0) for k in FEATURES})
            alerts = clinical.reassessment_alerts(
                assessment.get("days_post_onset", 0), assessment.get("mrc_dorsiflexors", 0))
            for a in alerts:
                st.markdown(f'<div class="alert-warn">⏰ {a}</div>', unsafe_allow_html=True)

            stiff = clinical.stiffness_score(
                assessment.get("weight_kg", 70), assessment.get("mas_ankle", 2),
                assessment.get("ankle_rom_df", 5), assessment.get("mrc_dorsiflexors", 2),
                assessment.get("activity_level", 1))

            col_rec, col_stiff = st.columns([1.6, 1])
            with col_rec:
                _render_recommendation(label, proba, assessment, "physio")
            with col_stiff:
                st.plotly_chart(_gauge(stiff, "Stiffness Requirement"), use_container_width=True)
                risks = clinical.compliance_risks(assessment)
                if risks:
                    st.markdown("**Compliance risk factors**")
                    for r in risks:
                        st.markdown(f"""<div class="risk-box">
                            <div class="risk-title">⚠ {r['factor']}</div>
                            <div class="risk-detail">{r['detail']}</div>
                            <span class="risk-action">→ {r['action']}</span>
                        </div>""", unsafe_allow_html=True)

            _prescription_export(patient, assessment, label, proba, stiff)

        with tab_wear:
            if wearing:
                fig = _wearing_chart(wearing)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(pd.DataFrame(wearing).sort_values("date", ascending=False),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No wearing log entries yet.")

        with tab_hist:
            history = store.assessments_get(patient["patient_id"])
            if history:
                st.dataframe(pd.DataFrame(history).sort_values("date", ascending=False),
                             use_container_width=True, hide_index=True)
    else:
        st.info("No assessment recorded for this patient yet. Use **New Assessment** in the sidebar.")


def _physio_assessment_form(patients: list, models: dict, best: str):
    st.markdown('<div class="page-title">New Clinical Assessment</div>', unsafe_allow_html=True)

    if not patients:
        st.warning("Register a patient first.")
        return

    options = {f"{p['full_name']} ({p['patient_id']})": p for p in patients}
    patient = options[st.selectbox("Patient", list(options.keys()))]

    with st.form("assessment"):
        st.markdown("**Neurological & motor function**")
        c1, c2, c3 = st.columns(3)
        with c1:
            mrc = st.slider("MRC dorsiflexors (0–5)", 0, 5, 2,
                            help="0=no movement · 3=against gravity · 5=normal")
            brunnstrom = st.slider("Brunnstrom stage (1–6)", 1, 6, 3,
                                   help="1=flaccidity · 6=near normal")
        with c2:
            fac = st.slider("FAC score (0–5)", 0, 5, 2,
                            help="Functional Ambulation Category · 0=non-ambulatory · 5=independent on stairs")
            mas = st.slider("MAS ankle (0–4)", 0, 4, 1,
                            help="Modified Ashworth Scale · 0=no tone · 4=rigid")
        with c3:
            rom = st.slider("Ankle dorsiflexion ROM (°)", -20, 20, 5,
                            help="Negative=contracture · 0=neutral · positive=can dorsiflex")
            strength_overall = st.slider("Overall lower limb strength (0–5 MRC)", 0, 5, 3)

        st.markdown("**Gait characteristics**")
        c4, c5 = st.columns(2)
        with c4:
            crouch = st.checkbox("Crouch gait present")
            knee_stable = st.checkbox("Knee stable during stance", value=True)
        with c5:
            vv = st.select_slider("Varus/valgus deformity",
                                  ["None", "Mild", "Moderate", "Severe"])
            vv_num = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}[vv]

        st.markdown("**Patient profile**")
        c6, c7, c8 = st.columns(3)
        with c6:
            weight = st.number_input("Weight (kg)", 30, 160, 70)
            activity_label = st.selectbox(
                "Activity level",
                ["Bed-bound / chair-bound", "Household ambulator",
                 "Community ambulator", "Active / returning to work"])
            activity_num = {"Bed-bound / chair-bound": 0, "Household ambulator": 1,
                            "Community ambulator": 2, "Active / returning to work": 3}[activity_label]
        with c7:
            sensation = st.checkbox("Intact skin sensation (affected limb)", value=True)
            pacemaker = st.checkbox("Implanted cardiac device (pacemaker/ICD)")
        with c8:
            days_post = st.number_input(
                "Days post-onset at today's assessment", 0, 2000,
                patient["days_post_onset"])

        notes = st.text_area("Clinical notes", height=70)
        submitted = st.form_submit_button("Save assessment and generate recommendation")

    if submitted:
        bucket = 0 if days_post < 30 else (1 if days_post < 90 else (2 if days_post < 180 else 3))
        row = {
            "age": patient["age"],
            "days_post_onset": days_post,
            "sex_num": 1 if patient["sex"] == "Male" else 0,
            "diagnosis_num": 0 if "ischemic" in patient["diagnosis"].lower()
                             else (1 if "hemorrhagic" in patient["diagnosis"].lower() else 2),
            "mrc_dorsiflexors": mrc,
            "brunnstrom_stage": brunnstrom,
            "fac_score": fac,
            "mas_ankle": mas,
            "ankle_rom_df": rom,
            "knee_stability": int(knee_stable),
            "skin_sensation": int(sensation),
            "crouch_gait": int(crouch),
            "varus_valgus": vv_num,
            "weight_kg": weight,
            "activity_level": activity_num,
            "has_pacemaker": int(pacemaker),
            "days_post_onset_bucket": bucket,
            "notes": notes,
        }
        store.assessment_save(patient["patient_id"], row)

        label, proba = clinical.predict(models, best, row)
        stiff  = clinical.stiffness_score(weight, mas, rom, mrc, activity_num)
        alerts = clinical.reassessment_alerts(days_post, mrc)

        st.success("Assessment saved.")

        for a in alerts:
            st.markdown(f'<div class="alert-warn">⏰ {a}</div>', unsafe_allow_html=True)

        col_rec, col_g = st.columns([1.5, 1])
        with col_rec:
            _render_recommendation(label, proba, row, "physio")
        with col_g:
            st.plotly_chart(_gauge(stiff, "Stiffness Requirement"), use_container_width=True)
            risks = clinical.compliance_risks(row)
            for r in risks:
                st.markdown(f"""<div class="risk-box">
                    <div class="risk-title">⚠ {r['factor']}</div>
                    <div class="risk-detail">{r['detail']}</div>
                    <span class="risk-action">→ {r['action']}</span>
                </div>""", unsafe_allow_html=True)

        _prescription_export(patient, row, label, proba, stiff)

        st.markdown("""<div class="disclaimer">
        Recommendation generated by ML model trained on synthetic data structured from
        Lee et al. (2021) PMC8055674. Not validated for clinical use without orthotist oversight.
        </div>""", unsafe_allow_html=True)


def _physio_register_patient(physio_username: str):
    st.markdown('<div class="page-title">Register New Patient</div>', unsafe_allow_html=True)

    with st.form("reg_patient"):
        c1, c2 = st.columns(2)
        with c1:
            fname   = st.text_input("Full name")
            pid     = st.text_input("Patient ID (they will use this to log in)",
                                     placeholder="e.g. OLU-001")
            age     = st.number_input("Age", 18, 100, 55)
            sex     = st.selectbox("Sex", ["Female", "Male", "Other"])
        with c2:
            diag    = st.selectbox("Primary diagnosis",
                                   ["Ischemic stroke", "Hemorrhagic stroke",
                                    "Traumatic brain injury", "Cerebral palsy",
                                    "Multiple sclerosis", "Spinal cord injury", "Other"])
            days    = st.number_input("Days since onset at registration", 0, 2000, 30)
            pw      = st.text_input("Temporary password (patient changes this on first login)",
                                     type="password")
            pw2     = st.text_input("Confirm password", type="password")

        submitted = st.form_submit_button("Register patient")

    if submitted:
        if not all([fname, pid, pw]):
            st.error("Name, patient ID, and password are required.")
        elif pw != pw2:
            st.error("Passwords do not match.")
        elif store.patient_register(physio_username, pid.strip(), fname, age, sex, diag, days, pw):
            st.success(f"Patient **{fname}** registered. Patient ID: `{pid.strip().lower()}`")
        else:
            st.error(f"Patient ID `{pid.strip().lower()}` is already in use.")


def _physio_cohort(patients: list):
    st.markdown('<div class="page-title">Cohort Overview</div>', unsafe_allow_html=True)

    if not patients:
        st.info("No patients yet.")
        return

    rows = []
    for p in patients:
        wearing = store.wearing_log_get(p["patient_id"])
        worn    = sum(1 for e in wearing if e.get("worn"))
        total   = len(wearing)
        last    = store.assessment_latest(p["patient_id"])
        rows.append({
            "Name":              p["full_name"],
            "ID":                p["patient_id"],
            "Age":               p["age"],
            "Diagnosis":         p["diagnosis"],
            "Days post-onset":   p["days_post_onset"],
            "Assessments":       len(store.assessments_get(p["patient_id"])),
            "Wearing compliance": f"{worn/total:.0%}" if total else "—",
            "Last assessment":   last["date"] if last else "None",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    total_p = len(patients)
    assessed = sum(1 for p in patients if store.assessment_latest(p["patient_id"]))
    logged   = sum(1 for p in patients if store.wearing_log_get(p["patient_id"]))

    c1, c2, c3 = st.columns(3)
    for col, label, val in [
        (c1, "Total patients", total_p),
        (c2, "Assessed", assessed),
        (c3, "Logging compliance", logged),
    ]:
        with col:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{val}</div>
            </div>""", unsafe_allow_html=True)


def _physio_model_perf(models: dict, best: str):
    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(f"Active model: **{best}** (highest ROC-AUC on held-out test set)")
    st.plotly_chart(_model_perf_chart(models), use_container_width=True)
    st.markdown("""
    **Training data:** 4,000 synthetic records. Feature schema matches the first published ML study
    predicting AFO need in stroke patients (Lee et al., 2021, PMC8055674), extended with stiffness
    features from the Frontiers Rehabilitation Sciences (2024) evidence-guided algorithm.

    **6 output classes:** PLS · Solid Ankle · Hinged · Carbon Fibre · GRAFO · FES

    **17 clinical features:** Age · Days post-onset · Sex · Diagnosis · MRC dorsiflexors ·
    Brunnstrom stage · FAC score · MAS ankle · Ankle ROM · Knee stability · Skin sensation ·
    Crouch gait · Varus/valgus · Weight · Activity level · Pacemaker · Days-post bucket
    """)


# ── Prescription export ───────────────────────────────────────────────────────

def _prescription_export(patient: dict, assessment: dict, label: int, proba: list, stiff: float):
    top_key = LABEL_TO_KEY[label]
    top_afo = AFO_TYPES[top_key]
    today   = date.today().strftime("%d %B %Y")

    questions = [
        f"Is a <strong>{top_afo['abbreviation']}</strong> appropriate for this patient's profile?",
        f"Stiffness score is <strong>{stiff}/10</strong> — what spring constant or material thickness do you recommend?",
        "Should this be custom-moulded or will an off-the-shelf fit suffice?",
        "What shoe modifications are needed?",
        "Is FES a viable alternative at this stage of recovery?",
        "When should the patient return for reassessment?",
    ]
    q_html = "".join(f"<li style='margin-bottom:5px;'>{q}</li>" for q in questions)

    summary_html = f"""
    <div style="font-family:'IBM Plex Sans',sans-serif;max-width:680px;border:2px solid #2563eb;
                border-radius:12px;overflow:hidden;">
        <div style="background:#2563eb;color:white;padding:18px 22px;">
            <div style="font-size:1.15rem;font-weight:600;">AFO Clinical Assessment — Prescription Summary</div>
            <div style="font-size:0.82rem;opacity:0.85;margin-top:3px;">
                Generated {today} · AFO Clinical Platform · Samuel Oluwakoya
            </div>
        </div>
        <div style="padding:18px 22px;background:white;">
            <div style="font-size:0.72rem;font-weight:600;color:#2563eb;text-transform:uppercase;
                        letter-spacing:.06em;margin-bottom:8px;">Patient</div>
            <table style="width:100%;font-size:0.84rem;border-collapse:collapse;margin-bottom:16px;">
                <tr>
                    <td style="color:#64748b;padding:3px 0;width:35%;">Name</td>
                    <td style="font-weight:500;">{patient['full_name']}</td>
                    <td style="color:#64748b;width:30%;">Age</td>
                    <td style="font-weight:500;">{patient['age']}</td>
                </tr>
                <tr>
                    <td style="color:#64748b;padding:3px 0;">Diagnosis</td>
                    <td style="font-weight:500;">{patient['diagnosis']}</td>
                    <td style="color:#64748b;">Days post-onset</td>
                    <td style="font-weight:500;">{assessment.get('days_post_onset','—')}</td>
                </tr>
                <tr>
                    <td style="color:#64748b;padding:3px 0;">MRC dorsiflexors</td>
                    <td style="font-weight:500;">{assessment.get('mrc_dorsiflexors','—')}/5</td>
                    <td style="color:#64748b;">MAS ankle</td>
                    <td style="font-weight:500;">{assessment.get('mas_ankle','—')}/4</td>
                </tr>
                <tr>
                    <td style="color:#64748b;padding:3px 0;">Ankle ROM</td>
                    <td style="font-weight:500;">{assessment.get('ankle_rom_df','—')}°</td>
                    <td style="color:#64748b;">Activity level</td>
                    <td style="font-weight:500;">{assessment.get('activity_level','—')}</td>
                </tr>
            </table>
            <div style="font-size:0.72rem;font-weight:600;color:#2563eb;text-transform:uppercase;
                        letter-spacing:.06em;margin-bottom:8px;">ML Recommendation</div>
            <div style="background:#eff6ff;border-radius:8px;padding:12px 14px;margin-bottom:14px;">
                <div style="font-weight:600;font-size:0.98rem;color:#1e40af;">
                    {top_afo['name']} ({top_afo['abbreviation']})
                </div>
                <div style="font-size:0.82rem;color:#3b82f6;margin-top:3px;">
                    Confidence: {proba[label]:.0%} · Stiffness score: {stiff}/10 ·
                    Compliance risk: {top_afo['compliance_risk']}
                </div>
            </div>
            <div style="font-size:0.72rem;font-weight:600;color:#2563eb;text-transform:uppercase;
                        letter-spacing:.06em;margin-bottom:8px;">Questions for the orthotist</div>
            <ol style="font-size:0.84rem;color:#334155;line-height:1.9;padding-left:18px;
                       margin-bottom:14px;">{q_html}</ol>
            <div style="font-size:0.72rem;color:#94a3b8;border-top:1px solid #f1f5f9;padding-top:8px;">
                Generated by AFO Clinical Platform (Samuel Oluwakoya). ML guidance only —
                final prescription requires a qualified orthotist or rehabilitation physician.
            </div>
        </div>
    </div>
    """

    with st.expander("🖨 View prescription summary for orthotist"):
        components.html(summary_html, height=620, scrolling=True)
        txt = (
            f"AFO PRESCRIPTION SUMMARY — {today}\n"
            f"Patient: {patient['full_name']} | Age: {patient['age']} | {patient['diagnosis']}\n"
            f"MRC: {assessment.get('mrc_dorsiflexors')} | MAS: {assessment.get('mas_ankle')} | ROM: {assessment.get('ankle_rom_df')}°\n"
            f"RECOMMENDATION: {top_afo['name']} | Confidence: {proba[label]:.0%} | Stiffness: {stiff}/10\n"
            f"Cost: {top_afo['cost_usd']} / {top_afo['cost_ngn']}\n"
            f"Platform: AFO Clinical Platform · Samuel Oluwakoya\n"
        )
        st.download_button("⬇ Download summary (.txt)", txt,
                           f"prescription_{patient['patient_id']}_{date.today()}.txt", "text/plain")


# ── Patient portal ────────────────────────────────────────────────────────────

EDUCATION = [
    ("Will my AFO cure foot drop?",
     "An AFO is a functional support tool, not a cure. It improves how you walk and reduces fall risk "
     "while your nervous system continues to recover. Many patients with early post-stroke foot drop "
     "see motor improvements over 6 months — your physiotherapist will monitor whether you still need "
     "the AFO at each review."),
    ("How long do I wear it each day?",
     "In early recovery, typically throughout all walking time. As strength returns, your physiotherapist "
     "may trial walking without it. Never stop using it without clinical guidance — "
     "falls are the main risk when the AFO is removed too early."),
    ("My AFO is causing pain — what do I do?",
     "Stop wearing it and check your skin immediately. Pain indicates pressure. Do not push through it. "
     "A single adjustment appointment with your orthotist solves most pain problems — "
     "common causes are liner wear, strap position, or footwear that is too tight."),
    ("What shoes can I wear?",
     "You typically need one full size up on the affected foot to accommodate the AFO. "
     "Wide toe-box lace-up shoes or trainers work best. "
     "Slip-ons and heels are generally incompatible with solid and hinged designs."),
    ("What is the difference between AFO and FES?",
     "An AFO supports your foot passively — it holds it in the right position. "
     "FES (electrical stimulation) actively fires your muscles — which may help re-train the nerve pathways "
     "over time. Evidence shows both improve gait speed and balance. "
     "Your physiotherapist will advise which is appropriate for your stage of recovery."),
    ("When will I be reassessed?",
     "The recommended schedule is every 4–6 weeks in the first 6 months, then every 3–6 months after that. "
     "Triggers for an earlier reassessment: AFO no longer fits, walking has changed significantly, "
     "skin problems, or you feel the device is no longer helping."),
]

SATISFACTION_ITEMS = [
    "Overall fit and comfort",
    "Ease of putting on and taking off",
    "How it feels during walking",
    "Safety — how secure it feels",
    "Its effect on your walking ability",
    "How it looks (appearance)",
    "Your confidence while wearing it",
]


def _patient_portal(models: dict, best: str):
    patient = st.session_state["user"]

    with st.sidebar:
        st.markdown('<div class="role-badge patient-badge">Patient</div>', unsafe_allow_html=True)
        st.markdown(f"**{patient['full_name']}**")
        st.markdown(f"*{patient['diagnosis']}*")
        st.markdown("---")
        nav = st.radio("", ["My AFO", "Daily Log", "Satisfaction", "Education"],
                       label_visibility="collapsed")
        st.markdown("---")
        if st.button("Sign out"):
            _logout()

    if nav == "My AFO":
        _patient_my_afo(patient, models, best)
    elif nav == "Daily Log":
        _patient_wearing_log(patient)
    elif nav == "Satisfaction":
        _patient_satisfaction(patient)
    elif nav == "Education":
        _patient_education()


def _patient_my_afo(patient: dict, models: dict, best: str):
    st.markdown('<div class="page-title">My AFO Recommendation</div>', unsafe_allow_html=True)

    assessment = store.assessment_latest(patient["patient_id"])

    if not assessment:
        st.info("Your physiotherapist has not completed a clinical assessment yet. "
                "Your AFO recommendation will appear here once they have.")
        return

    label, proba = clinical.predict(models, best, {k: assessment.get(k, 0) for k in FEATURES})
    top_key = LABEL_TO_KEY[label]
    top_afo = AFO_TYPES[top_key]

    alerts = clinical.reassessment_alerts(
        assessment.get("days_post_onset", 0), assessment.get("mrc_dorsiflexors", 0))
    for a in alerts:
        st.markdown(f'<div class="alert-info">ℹ {a}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="rec-primary">
        <div style="font-size:0.72rem;font-weight:600;color:#1d4ed8;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:4px;">Your recommended device</div>
        <div class="rec-name">{top_afo['name']}</div>
        <div style="margin-top:10px;font-size:0.88rem;color:#334155;line-height:1.75;">
            {top_afo['patient_description']}
        </div>
        <div style="margin-top:10px;font-size:0.82rem;color:#64748b;">
            Approximate cost: {top_afo['cost_usd']} / {top_afo['cost_ngn']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    wearing   = store.wearing_log_get(patient["patient_id"])
    worn_days = sum(1 for e in wearing if e.get("worn"))
    total_d   = len(wearing)
    streak    = 0
    for e in reversed(sorted(wearing, key=lambda x: x["date"])):
        if e.get("worn"):
            streak += 1
        else:
            break

    if total_d:
        c1, c2, c3 = st.columns(3)
        for col, label, val in [
            (c1, "Days worn", worn_days),
            (c2, "Current streak 🔥", streak),
            (c3, "Compliance", f"{worn_days/total_d:.0%}" if total_d else "—"),
        ]:
            with col:
                st.markdown(f"""<div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{val}</div>
                </div>""", unsafe_allow_html=True)

    _render_recommendation(label, proba, assessment, "patient")

    st.markdown("""<div class="disclaimer">
    This recommendation is generated by a machine learning model and reviewed by your physiotherapist.
    Your orthotist will make the final device selection. Always follow clinical guidance.
    </div>""", unsafe_allow_html=True)


def _patient_wearing_log(patient: dict):
    st.markdown('<div class="page-title">Daily Wearing Log</div>', unsafe_allow_html=True)
    st.markdown("Track whether you wore your AFO today. Consistent use is the single biggest factor in your outcome.")

    log = store.wearing_log_get(patient["patient_id"])
    today_entry = store.wearing_log_today(patient["patient_id"])

    col_form, col_chart = st.columns([1, 1.5])

    with col_form:
        if today_entry:
            status = "✅ Worn" if today_entry["worn"] else "❌ Not worn"
            st.success(f"Today logged: **{status}** · {today_entry.get('hours', 0)} hours")
        else:
            worn  = st.radio("Did you wear your AFO today?", ["Yes", "No"], horizontal=True)
            hours = st.slider("Hours worn", 0, 16, 6) if worn == "Yes" else 0
            pain  = st.slider("Comfort level today (1 = painful, 10 = comfortable)", 1, 10, 7)
            note  = st.text_input("Any notes?", placeholder="e.g. rubbing at heel")
            if st.button("Save today's entry"):
                store.wearing_log_save(patient["patient_id"], {
                    "worn": worn == "Yes",
                    "hours": hours,
                    "comfort": pain,
                    "note": note,
                })
                st.rerun()

    with col_chart:
        if len(log) >= 2:
            fig = _wearing_chart(log)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    if log:
        worn_n = sum(1 for e in log if e.get("worn"))
        total  = len(log)
        if worn_n / total < 0.7:
            st.markdown("""<div class="alert-warn">
            ⚠ Your wearing compliance is below 70%. Consistent AFO use is important for your recovery.
            If discomfort is the reason, tell your physiotherapist — most fitting issues are easily fixed.
            </div>""", unsafe_allow_html=True)

        with st.expander("View all entries"):
            df = pd.DataFrame(log).sort_values("date", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download log", df.to_csv(index=False),
                               f"wearing_log_{date.today()}.csv", "text/csv")


def _patient_satisfaction(patient: dict):
    st.markdown('<div class="page-title">Satisfaction Assessment</div>', unsafe_allow_html=True)
    st.markdown(
        "Rate your current AFO on each item. Your physiotherapist can see this — "
        "it helps them advocate for adjustments at your next appointment."
    )

    with st.form("satisfaction"):
        ratings = {}
        for i, item in enumerate(SATISFACTION_ITEMS):
            ratings[item] = st.slider(item, 1, 5, 3, key=f"sat_{i}",
                                       help="1 = not satisfied · 5 = very satisfied")
        submitted = st.form_submit_button("Save assessment")

    if submitted:
        store.satisfaction_save(patient["patient_id"], ratings)
        st.success("Satisfaction scores saved.")

    history = store.satisfaction_get(patient["patient_id"])
    if history:
        latest = history[-1]
        total  = round(sum(latest["scores"].values()) / len(latest["scores"]), 2)
        st.markdown(f"**Latest score: {total}/5** ({latest['date']})")

        items = list(latest["scores"].keys())
        vals  = list(latest["scores"].values())
        fig = go.Figure(go.Bar(
            x=vals, y=[i[:35] for i in items], orientation="h",
            marker_color=["#16a34a" if v >= 4 else "#f59e0b" if v >= 3 else "#dc2626" for v in vals],
            text=vals, textposition="outside",
        ))
        fig.update_layout(
            height=280, margin=dict(l=10, r=40, t=10, b=10),
            xaxis=dict(range=[0, 6.5]),
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

        if total < 3.0:
            st.markdown("""<div class="alert-warn">
            ⚠ Your satisfaction score is low. Please mention this at your next physiotherapy appointment.
            Many issues that cause dissatisfaction — fit, comfort, appearance — can be resolved
            through adjustment or a different device type.
            </div>""", unsafe_allow_html=True)


def _patient_education():
    st.markdown('<div class="page-title">Understanding Your AFO</div>', unsafe_allow_html=True)
    st.markdown("Common questions about ankle-foot orthoses — answered clearly.")

    for q, a in EDUCATION:
        with st.expander(q):
            st.markdown(f'<div style="font-size:0.88rem;color:#334155;line-height:1.75;">{a}</div>',
                        unsafe_allow_html=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    models, best = clinical.load_models()

    if not st.session_state["authenticated"]:
        _page_login()
    elif st.session_state["role"] == "physio":
        _physio_portal(models, best)
    elif st.session_state["role"] == "patient":
        _patient_portal(models, best)


if __name__ == "__main__":
    main()

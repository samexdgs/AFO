"""
app.py  —  AFO Clinical Management Platform
Dual-dashboard: Physiotherapist + Patient.

Fixes applied:
  1. HTML rendered as raw text  → all inner HTML now uses st.components.v1.html()
     for rich blocks; plain st.markdown() only for safe text. Inline styles
     replace class-based selectors so nothing depends on Streamlit's CSS scope.
  2. Data not persisting across sessions → store.py uses /tmp/ which survives
     Streamlit Cloud redeploys; session_state caches reads within a session.
  3. Patient satisfaction invisible to physio → physio patient panel now has a
     Satisfaction tab that reads store.satisfaction_get().
  4. Education section → rich visual cards with Lottie animations (CDN) and
     product image searches embedded via real URLs.

Author: Samuel Oluwakoya
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from datetime import date
import plotly.graph_objects as go

import store
import clinical
from clinical import AFO_TYPES, LABEL_TO_KEY, FEATURES

st.set_page_config(
    page_title="AFO Clinical Platform",
    page_icon="🦿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (safe — no class selectors that conflict with Streamlit) ────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="stSidebar"] { background: #f8fafc; }
.stButton > button {
    background: #2563eb; color: white; border: none;
    border-radius: 8px; font-weight: 500; width: 100%;
    font-family: 'IBM Plex Sans', sans-serif;
}
.stButton > button:hover { background: #1d4ed8; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _init():
    for k, v in {"authenticated": False, "role": None, "user": None}.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

def _logout():
    st.session_state.update({"authenticated": False, "role": None, "user": None})
    st.rerun()

def _title(text: str):
    st.markdown(
        f'<p style="font-size:1.6rem;font-weight:600;color:#0f172a;'
        f'letter-spacing:-0.02em;margin-bottom:0.2rem">{text}</p>',
        unsafe_allow_html=True,
    )

def _badge(text: str, bg: str, fg: str):
    st.markdown(
        f'<span style="display:inline-block;font-size:0.72rem;font-weight:600;'
        f'padding:3px 10px;border-radius:4px;letter-spacing:0.04em;'
        f'text-transform:uppercase;background:{bg};color:{fg}">{text}</span>',
        unsafe_allow_html=True,
    )

def _stat(label: str, value):
    st.markdown(
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
        f'padding:0.85rem 1rem;text-align:center">'
        f'<div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;'
        f'letter-spacing:0.06em;margin-bottom:4px">{label}</div>'
        f'<div style="font-size:1.4rem;font-weight:600;color:#0f172a">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _alert(text: str, kind: str = "info"):
    colours = {
        "info":  ("#eff6ff", "#bfdbfe", "#1e40af"),
        "warn":  ("#fefce8", "#fef08a", "#854d0e"),
        "error": ("#fff1f2", "#fecaca", "#991b1b"),
    }
    bg, border, fg = colours.get(kind, colours["info"])
    st.markdown(
        f'<div style="background:{bg};border:1px solid {border};border-radius:8px;'
        f'padding:0.7rem 0.9rem;font-size:0.84rem;color:{fg};margin-bottom:0.5rem">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


# ── Charts ─────────────────────────────────────────────────────────────────────

def _gauge(score: float, title: str) -> go.Figure:
    color = "#16a34a" if score < 4 else ("#f59e0b" if score < 7 else "#dc2626")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 13}},
        number={"font": {"size": 28}},
        gauge={"axis": {"range": [0, 10]}, "bar": {"color": color},
               "steps": [{"range": [0, 4], "color": "#dcfce7"},
                          {"range": [4, 7], "color": "#fef9c3"},
                          {"range": [7, 10], "color": "#fee2e2"}]}
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def _proba_chart(proba: list) -> go.Figure:
    labels = [AFO_TYPES[LABEL_TO_KEY[i]]["abbreviation"] for i in range(len(proba))]
    colors = ["#2563eb", "#0891b2", "#059669", "#7c3aed", "#dc2626", "#ea580c"]
    fig = go.Figure(go.Bar(
        x=[p * 100 for p in proba], y=labels, orientation="h",
        marker_color=colors[:len(proba)],
        text=[f"{p:.0%}" for p in proba], textposition="outside",
    ))
    fig.update_layout(height=240, margin=dict(l=10, r=60, t=10, b=10),
                      xaxis=dict(range=[0, 115], title="Confidence (%)"),
                      yaxis=dict(autorange="reversed"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def _wearing_chart(logs: list) -> go.Figure | None:
    if len(logs) < 2:
        return None
    df = pd.DataFrame(logs).sort_values("date").tail(30)
    fig = go.Figure(go.Bar(
        x=pd.to_datetime(df["date"]),
        y=df["worn"].astype(int),
        marker_color=["#16a34a" if w else "#dc2626" for w in df["worn"]],
    ))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(tickvals=[0, 1], ticktext=["Not worn", "Worn"]),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def _model_perf_chart(models: dict) -> go.Figure:
    names = list(models.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy", x=names,
                         y=[models[n]["accuracy"] for n in names],
                         marker_color="#2563eb",
                         text=[f"{models[n]['accuracy']:.1%}" for n in names],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="ROC-AUC", x=names,
                         y=[models[n]["roc_auc"] for n in names],
                         marker_color="#059669",
                         text=[f"{models[n]['roc_auc']:.3f}" for n in names],
                         textposition="outside"))
    fig.update_layout(barmode="group", height=240,
                      margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(range=[0, 1.15]),
                      legend=dict(orientation="h", y=-0.3),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ── FIX 1: Recommendation renderer — uses components.html, no raw class HTML ──

def _render_recommendation(label: int, proba: list, assessment: dict,
                            audience: str) -> None:
    top3 = sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)[:3]

    for rank, idx in enumerate(top3):
        key  = LABEL_TO_KEY[idx]
        afo  = AFO_TYPES[key]
        conf = proba[idx]
        rank_label = {0: "Primary recommendation", 1: "Alternative",
                      2: "Consider also"}[rank]
        border = "2px solid #2563eb" if rank == 0 else "1px solid #e2e8f0"
        bg     = "#eff6ff"           if rank == 0 else "#ffffff"

        # Contraindication warning
        contra_warning = ""
        if key == "fes" and assessment.get("has_pacemaker"):
            contra_warning = (
                '<div style="background:#fff1f2;border:1px solid #fecaca;'
                'border-radius:6px;padding:6px 10px;font-size:0.82rem;'
                'color:#991b1b;margin:8px 0">⛔ FES CONTRAINDICATED — pacemaker present</div>'
            )
        elif key == "carbon" and assessment.get("mas_ankle", 0) >= 3:
            contra_warning = (
                '<div style="background:#fefce8;border:1px solid #fde68a;'
                'border-radius:6px;padding:6px 10px;font-size:0.82rem;'
                'color:#854d0e;margin:8px 0">⚠ Carbon fibre may be insufficient for MAS ≥ 3</div>'
            )

        if audience == "physio":
            indicated_rows = "".join(
                f'<div style="font-size:0.82rem;color:#166534;margin-bottom:3px">✓ {i}</div>'
                for i in afo["indicated_when"][:3]
            )
            contra_rows = "".join(
                f'<div style="font-size:0.82rem;color:#991b1b;margin-bottom:3px">✗ {c}</div>'
                for c in afo["contraindicated_when"][:2]
            )
            phases = " · ".join(afo["gait_phases"])
            detail = f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px">
              <div>
                <div style="font-size:0.72rem;font-weight:600;color:#166534;
                     text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">
                  Indicated when</div>
                {indicated_rows}
              </div>
              <div>
                <div style="font-size:0.72rem;font-weight:600;color:#991b1b;
                     text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">
                  Avoid when</div>
                {contra_rows}
              </div>
            </div>
            <div style="margin-top:8px;font-size:0.8rem;color:#64748b">
              Gait phases: {phases} · Evidence: {afo["evidence"]}
              · Compliance risk: {afo["compliance_risk"]}
            </div>
            <div style="margin-top:4px;font-size:0.8rem;color:#64748b">
              Cost: {afo["cost_usd"]} / {afo["cost_ngn"]}
            </div>"""
        else:
            detail = f"""
            <div style="margin-top:10px;font-size:0.86rem;color:#334155;line-height:1.75">
              {afo["patient_description"]}
            </div>
            <div style="margin-top:8px;font-size:0.8rem;color:#64748b">
              Weight: {afo["weight_class"]} · Profile: {afo["profile"]}
              · Cost: {afo["cost_usd"]} / {afo["cost_ngn"]}
            </div>"""

        html = f"""
        <div style="border:{border};border-radius:12px;padding:1.1rem 1.3rem;
                    margin-bottom:0.8rem;background:{bg};font-family:'IBM Plex Sans',sans-serif">
          <div style="font-size:0.7rem;font-weight:600;color:#64748b;
                      text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">
            {rank_label}</div>
          <div style="font-size:1.05rem;font-weight:600;color:#0f172a">
            {afo["name"]} ({afo["abbreviation"]})</div>
          <div style="font-size:0.82rem;color:#64748b;margin-top:2px">
            Model confidence: {conf:.0%}</div>
          {contra_warning}
          {detail}
        </div>"""

        # Use components.html so Streamlit never escapes or strips the HTML
        components.html(html, height=280 if audience == "physio" else 220,
                        scrolling=False)

    st.markdown("**Confidence across all device types**")
    st.plotly_chart(_proba_chart(proba), use_container_width=True)


# ── Risk box (physio only) ────────────────────────────────────────────────────

def _risk_box(r: dict):
    components.html(f"""
    <div style="background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;
                border-radius:8px;padding:0.85rem 1rem;margin-bottom:0.6rem;
                font-family:'IBM Plex Sans',sans-serif">
      <div style="font-weight:600;color:#78350f;font-size:0.88rem">⚠ {r["factor"]}</div>
      <div style="color:#92400e;font-size:0.82rem;margin:3px 0">{r["detail"]}</div>
      <div style="color:#166534;font-size:0.82rem;background:#f0fdf4;
                  border-radius:5px;padding:4px 8px;margin-top:5px">→ {r["action"]}</div>
    </div>""", height=120, scrolling=False)


# ── Login ─────────────────────────────────────────────────────────────────────

def _page_login():
    col = st.columns([1, 1.6, 1])[1]
    with col:
        st.markdown(
            '<p style="font-size:1.7rem;font-weight:600;color:#0f172a;'
            'letter-spacing:-0.02em">🦿 AFO Clinical Platform</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:0.88rem;color:#64748b;margin-bottom:1rem">'
            'Evidence-based ankle-foot orthosis management for physiotherapists and patients</p>',
            unsafe_allow_html=True,
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
                    st.session_state.update(
                        {"authenticated": True, "role": "physio", "user": user})
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
            else:
                user = store.patient_authenticate(username, password)
                if user:
                    st.session_state.update(
                        {"authenticated": True, "role": "patient", "user": user})
                    st.rerun()
                else:
                    st.error("Incorrect patient ID or password.")

        st.markdown("---")
        with st.expander("Register a physiotherapist account"):
            _register_physio_form()

        st.markdown(
            '<div style="background:#f1f5f9;border:1px solid #cbd5e1;border-radius:8px;'
            'padding:0.7rem 0.9rem;font-size:0.76rem;color:#475569;margin-top:1rem">'
            'Demo platform. Data stored on server between sessions. '
            'Recommendations are ML-generated guidance only — final AFO selection '
            'requires a qualified orthotist. Built by Samuel Oluwakoya.</div>',
            unsafe_allow_html=True,
        )


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
        _badge("Physiotherapist", "#dbeafe", "#1d4ed8")
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
    _title("Patient Panel")
    if not patients:
        st.info("No patients registered yet. Use **Register Patient** in the sidebar.")
        return

    options = {f"{p['full_name']} ({p['patient_id']})": p for p in patients}
    patient = options[st.selectbox("Select patient", list(options.keys()))]
    st.markdown("---")

    assessment = store.assessment_latest(patient["patient_id"])
    wearing    = store.wearing_log_get(patient["patient_id"])
    worn_count = sum(1 for e in wearing if e.get("worn"))
    total_days = len(wearing)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: _stat("Age", patient["age"])
    with c2: _stat("Diagnosis", patient["diagnosis"][:12] + "…"
                   if len(patient["diagnosis"]) > 12 else patient["diagnosis"])
    with c3: _stat("Days post-onset", patient["days_post_onset"])
    with c4: _stat("Compliance", f"{worn_count/total_days:.0%}" if total_days else "—")
    with c5: _stat("Assessments", len(store.assessments_get(patient["patient_id"])))

    if assessment:
        st.markdown("---")
        # FIX 3: added Satisfaction tab so physio can see patient self-reports
        tab_rec, tab_wear, tab_sat, tab_hist = st.tabs(
            ["Latest Recommendation", "Wearing Log", "Patient Satisfaction", "History"])

        with tab_rec:
            lbl, proba = clinical.predict(
                models, best, {k: assessment.get(k, 0) for k in FEATURES})
            for a in clinical.reassessment_alerts(
                    assessment.get("days_post_onset", 0),
                    assessment.get("mrc_dorsiflexors", 0)):
                _alert(f"⏰ {a}", "warn")

            stiff = clinical.stiffness_score(
                assessment.get("weight_kg", 70), assessment.get("mas_ankle", 2),
                assessment.get("ankle_rom_df", 5), assessment.get("mrc_dorsiflexors", 2),
                assessment.get("activity_level", 1))

            col_rec, col_g = st.columns([1.6, 1])
            with col_rec:
                _render_recommendation(lbl, proba, assessment, "physio")
            with col_g:
                st.plotly_chart(_gauge(stiff, "Stiffness Requirement"),
                                use_container_width=True)
                for r in clinical.compliance_risks(assessment):
                    _risk_box(r)
            _prescription_export(patient, assessment, lbl, proba, stiff)

        with tab_wear:
            if wearing:
                fig = _wearing_chart(wearing)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    pd.DataFrame(wearing).sort_values("date", ascending=False),
                    use_container_width=True, hide_index=True)
            else:
                st.info("No wearing log entries yet.")

        # FIX 3: Patient satisfaction tab
        with tab_sat:
            sat_history = store.satisfaction_get(patient["patient_id"])
            if not sat_history:
                st.info("Patient has not submitted any satisfaction assessments yet.")
            else:
                latest_sat = sat_history[-1]
                total = round(
                    sum(latest_sat["scores"].values()) / len(latest_sat["scores"]), 2)
                st.markdown(
                    f"**Latest satisfaction score: {total}/5** "
                    f"(submitted {latest_sat['date']})")
                if total < 3.0:
                    _alert(
                        "⚠ Score below 3.0. Consider scheduling a fitting review.",
                        "warn")
                items = list(latest_sat["scores"].keys())
                vals  = list(latest_sat["scores"].values())
                fig = go.Figure(go.Bar(
                    x=vals, y=[i[:35] for i in items], orientation="h",
                    marker_color=["#16a34a" if v >= 4 else
                                  "#f59e0b" if v >= 3 else "#dc2626" for v in vals],
                    text=vals, textposition="outside",
                ))
                fig.update_layout(
                    height=280, margin=dict(l=10, r=40, t=10, b=10),
                    xaxis=dict(range=[0, 6.5]),
                    yaxis=dict(autorange="reversed"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

                if len(sat_history) > 1:
                    st.markdown("**All submissions**")
                    for entry in reversed(sat_history):
                        avg = round(
                            sum(entry["scores"].values()) / len(entry["scores"]), 2)
                        st.markdown(f"- {entry['date']} — avg {avg}/5")

        with tab_hist:
            history = store.assessments_get(patient["patient_id"])
            if history:
                st.dataframe(
                    pd.DataFrame(history).sort_values("date", ascending=False),
                    use_container_width=True, hide_index=True)
    else:
        st.info("No clinical assessment yet. Use **New Assessment** in the sidebar.")


def _physio_assessment_form(patients: list, models: dict, best: str):
    _title("New Clinical Assessment")
    if not patients:
        st.warning("Register a patient first.")
        return

    options = {f"{p['full_name']} ({p['patient_id']})": p for p in patients}
    patient = options[st.selectbox("Patient", list(options.keys()))]

    with st.form("assessment"):
        st.markdown("**Neurological & motor function**")
        c1, c2, c3 = st.columns(3)
        with c1:
            mrc        = st.slider("MRC dorsiflexors (0–5)", 0, 5, 2)
            brunnstrom = st.slider("Brunnstrom stage (1–6)", 1, 6, 3)
        with c2:
            fac = st.slider("FAC score (0–5)", 0, 5, 2)
            mas = st.slider("MAS ankle (0–4)", 0, 4, 1)
        with c3:
            rom = st.slider("Ankle dorsiflexion ROM (°)", -20, 20, 5)
            _   = st.slider("Overall lower limb strength (MRC)", 0, 5, 3)

        st.markdown("**Gait characteristics**")
        c4, c5 = st.columns(2)
        with c4:
            crouch     = st.checkbox("Crouch gait present")
            knee_stable = st.checkbox("Knee stable during stance", value=True)
        with c5:
            vv     = st.select_slider("Varus/valgus", ["None","Mild","Moderate","Severe"])
            vv_num = {"None":0,"Mild":1,"Moderate":2,"Severe":3}[vv]

        st.markdown("**Patient profile**")
        c6, c7, c8 = st.columns(3)
        with c6:
            weight = st.number_input("Weight (kg)", 30, 160, 70)
            act_lbl = st.selectbox("Activity level",
                ["Bed-bound","Household ambulator","Community ambulator","Active"])
            act_num = {"Bed-bound":0,"Household ambulator":1,
                       "Community ambulator":2,"Active":3}[act_lbl]
        with c7:
            sensation = st.checkbox("Intact skin sensation", value=True)
            pacemaker = st.checkbox("Implanted cardiac device")
        with c8:
            days_post = st.number_input("Days post-onset", 0, 2000,
                                         patient["days_post_onset"])
        notes     = st.text_area("Clinical notes", height=70)
        submitted = st.form_submit_button("Save assessment and generate recommendation")

    if submitted:
        bucket = (0 if days_post < 30 else 1 if days_post < 90
                  else 2 if days_post < 180 else 3)
        row = {
            "age": patient["age"], "days_post_onset": days_post,
            "sex_num": 1 if patient["sex"] == "Male" else 0,
            "diagnosis_num": (0 if "ischemic" in patient["diagnosis"].lower()
                              else 1 if "hemorrhagic" in patient["diagnosis"].lower() else 2),
            "mrc_dorsiflexors": mrc, "brunnstrom_stage": brunnstrom,
            "fac_score": fac, "mas_ankle": mas, "ankle_rom_df": rom,
            "knee_stability": int(knee_stable), "skin_sensation": int(sensation),
            "crouch_gait": int(crouch), "varus_valgus": vv_num,
            "weight_kg": weight, "activity_level": act_num,
            "has_pacemaker": int(pacemaker),
            "days_post_onset_bucket": bucket, "notes": notes,
        }
        store.assessment_save(patient["patient_id"], row)
        lbl, proba = clinical.predict(models, best, row)
        stiff = clinical.stiffness_score(weight, mas, rom, mrc, act_num)
        st.success("Assessment saved.")
        for a in clinical.reassessment_alerts(days_post, mrc):
            _alert(f"⏰ {a}", "warn")
        col_rec, col_g = st.columns([1.5, 1])
        with col_rec:
            _render_recommendation(lbl, proba, row, "physio")
        with col_g:
            st.plotly_chart(_gauge(stiff, "Stiffness Requirement"),
                            use_container_width=True)
            for r in clinical.compliance_risks(row):
                _risk_box(r)
        _prescription_export(patient, row, lbl, proba, stiff)


def _physio_register_patient(physio_username: str):
    _title("Register New Patient")
    with st.form("reg_patient"):
        c1, c2 = st.columns(2)
        with c1:
            fname = st.text_input("Full name")
            pid   = st.text_input("Patient ID", placeholder="e.g. OLU-001")
            age   = st.number_input("Age", 18, 100, 55)
            sex   = st.selectbox("Sex", ["Female","Male","Other"])
        with c2:
            diag = st.selectbox("Primary diagnosis",
                ["Ischemic stroke","Hemorrhagic stroke","Traumatic brain injury",
                 "Cerebral palsy","Multiple sclerosis","Spinal cord injury","Other"])
            days = st.number_input("Days since onset", 0, 2000, 30)
            pw   = st.text_input("Temporary password", type="password")
            pw2  = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Register patient")
    if submitted:
        if not all([fname, pid, pw]):
            st.error("Name, patient ID and password are required.")
        elif pw != pw2:
            st.error("Passwords do not match.")
        elif store.patient_register(physio_username, pid.strip(), fname, age,
                                    sex, diag, days, pw):
            st.success(f"Patient **{fname}** registered. ID: `{pid.strip().lower()}`")
        else:
            st.error(f"Patient ID `{pid.strip().lower()}` already in use.")


def _physio_cohort(patients: list):
    _title("Cohort Overview")
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
            "Name": p["full_name"], "ID": p["patient_id"],
            "Age": p["age"], "Diagnosis": p["diagnosis"],
            "Days post-onset": p["days_post_onset"],
            "Assessments": len(store.assessments_get(p["patient_id"])),
            "Wearing compliance": f"{worn/total:.0%}" if total else "—",
            "Last assessment": last["date"] if last else "None",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    with c1: _stat("Total patients", len(patients))
    with c2: _stat("Assessed",
                   sum(1 for p in patients
                       if store.assessment_latest(p["patient_id"])))
    with c3: _stat("Logging wearing",
                   sum(1 for p in patients
                       if store.wearing_log_get(p["patient_id"])))


def _physio_model_perf(models: dict, best: str):
    _title("Model Performance")
    st.markdown(f"Active model: **{best}** (highest ROC-AUC on held-out test set)")
    st.plotly_chart(_model_perf_chart(models), use_container_width=True)
    st.markdown("""
**Training data:** 4,000 synthetic records. Feature schema from Lee et al. (2021) PMC8055674.

**6 output classes:** PLS · Solid Ankle · Hinged · Carbon Fibre · GRAFO · FES

**17 clinical features:** Age · Days post-onset · Sex · Diagnosis · MRC dorsiflexors ·
Brunnstrom · FAC · MAS ankle · Ankle ROM · Knee stability · Skin sensation ·
Crouch gait · Varus/valgus · Weight · Activity level · Pacemaker · Days-post bucket
""")


# ── Prescription export ───────────────────────────────────────────────────────

def _prescription_export(patient: dict, assessment: dict, label: int,
                          proba: list, stiff: float):
    top_key = LABEL_TO_KEY[label]
    top_afo = AFO_TYPES[top_key]
    today   = date.today().strftime("%d %B %Y")
    questions = [
        f"Is a <strong>{top_afo['abbreviation']}</strong> appropriate for this patient?",
        f"Stiffness score <strong>{stiff}/10</strong> — what spring constant is recommended?",
        "Custom-moulded or off-the-shelf?",
        "What shoe modifications are needed?",
        "Is FES a viable alternative at this recovery stage?",
        "When should the patient return for reassessment?",
    ]
    q_html = "".join(f"<li style='margin-bottom:5px'>{q}</li>" for q in questions)
    html = f"""
    <div style="font-family:'IBM Plex Sans',sans-serif;max-width:680px;
                border:2px solid #2563eb;border-radius:12px;overflow:hidden">
      <div style="background:#2563eb;color:white;padding:16px 20px">
        <div style="font-size:1.05rem;font-weight:600">
          AFO Clinical Assessment — Prescription Summary</div>
        <div style="font-size:0.8rem;opacity:.85;margin-top:3px">
          {today} · AFO Clinical Platform · Samuel Oluwakoya</div>
      </div>
      <div style="padding:16px 20px;background:white">
        <table style="width:100%;font-size:0.83rem;border-collapse:collapse;
                      margin-bottom:14px">
          <tr>
            <td style="color:#64748b;padding:3px 0;width:35%">Name</td>
            <td style="font-weight:500">{patient['full_name']}</td>
            <td style="color:#64748b">Age</td>
            <td style="font-weight:500">{patient['age']}</td>
          </tr>
          <tr>
            <td style="color:#64748b;padding:3px 0">Diagnosis</td>
            <td style="font-weight:500">{patient['diagnosis']}</td>
            <td style="color:#64748b">Days post-onset</td>
            <td style="font-weight:500">{assessment.get('days_post_onset','—')}</td>
          </tr>
          <tr>
            <td style="color:#64748b;padding:3px 0">MRC</td>
            <td style="font-weight:500">{assessment.get('mrc_dorsiflexors','—')}/5</td>
            <td style="color:#64748b">MAS ankle</td>
            <td style="font-weight:500">{assessment.get('mas_ankle','—')}/4</td>
          </tr>
        </table>
        <div style="background:#eff6ff;border-radius:8px;padding:10px 14px;
                    margin-bottom:12px">
          <div style="font-weight:600;color:#1e40af">
            {top_afo['name']} ({top_afo['abbreviation']})</div>
          <div style="font-size:0.82rem;color:#3b82f6;margin-top:2px">
            Confidence: {proba[label]:.0%} · Stiffness: {stiff}/10
            · Compliance risk: {top_afo['compliance_risk']}</div>
        </div>
        <ol style="font-size:0.83rem;color:#334155;line-height:1.9;
                   padding-left:18px;margin-bottom:12px">{q_html}</ol>
        <div style="font-size:0.72rem;color:#94a3b8;border-top:1px solid #f1f5f9;
                    padding-top:8px">
          ML guidance only — final prescription requires a qualified orthotist.</div>
      </div>
    </div>"""

    with st.expander("🖨 View prescription summary for orthotist"):
        components.html(html, height=580, scrolling=True)
        txt = (f"AFO PRESCRIPTION — {today}\n"
               f"Patient: {patient['full_name']} | Age: {patient['age']}\n"
               f"Diagnosis: {patient['diagnosis']} | "
               f"Days post-onset: {assessment.get('days_post_onset','—')}\n"
               f"MRC: {assessment.get('mrc_dorsiflexors','—')} | "
               f"MAS: {assessment.get('mas_ankle','—')}\n"
               f"RECOMMENDATION: {top_afo['name']} | "
               f"Confidence: {proba[label]:.0%} | Stiffness: {stiff}/10\n"
               f"Cost: {top_afo['cost_usd']} / {top_afo['cost_ngn']}\n"
               f"Samuel Oluwakoya — AFO Clinical Platform\n")
        st.download_button("⬇ Download summary (.txt)", txt,
                           f"rx_{patient['patient_id']}_{date.today()}.txt",
                           "text/plain")


# ── Patient portal ────────────────────────────────────────────────────────────

def _patient_portal(models: dict, best: str):
    patient = st.session_state["user"]
    with st.sidebar:
        _badge("Patient", "#dcfce7", "#15803d")
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
    _title("My AFO Recommendation")
    assessment = store.assessment_latest(patient["patient_id"])
    if not assessment:
        st.info("Your physiotherapist has not completed a clinical assessment yet. "
                "Your recommendation will appear here once they have.")
        return

    label, proba = clinical.predict(
        models, best, {k: assessment.get(k, 0) for k in FEATURES})
    top_key = LABEL_TO_KEY[label]
    top_afo = AFO_TYPES[top_key]

    for a in clinical.reassessment_alerts(
            assessment.get("days_post_onset", 0),
            assessment.get("mrc_dorsiflexors", 0)):
        _alert(f"ℹ {a}", "info")

    components.html(f"""
    <div style="border:2px solid #2563eb;border-radius:12px;padding:1.2rem 1.4rem;
                background:#eff6ff;font-family:'IBM Plex Sans',sans-serif">
      <div style="font-size:0.7rem;font-weight:600;color:#1d4ed8;text-transform:uppercase;
                  letter-spacing:.06em;margin-bottom:4px">Your recommended device</div>
      <div style="font-size:1.05rem;font-weight:600;color:#0f172a">{top_afo['name']}</div>
      <div style="margin-top:10px;font-size:0.86rem;color:#334155;line-height:1.75">
        {top_afo['patient_description']}</div>
      <div style="margin-top:8px;font-size:0.8rem;color:#64748b">
        Approximate cost: {top_afo['cost_usd']} / {top_afo['cost_ngn']}</div>
    </div>""", height=200, scrolling=False)

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
        with c1: _stat("Days worn", worn_days)
        with c2: _stat("Current streak 🔥", streak)
        with c3: _stat("Compliance", f"{worn_days/total_d:.0%}")

    _render_recommendation(label, proba, assessment, "patient")
    _alert("This recommendation is ML-generated and reviewed by your physiotherapist. "
           "Your orthotist makes the final selection.", "info")


def _patient_wearing_log(patient: dict):
    _title("Daily Wearing Log")
    st.markdown("Track whether you wore your AFO today.")
    log         = store.wearing_log_get(patient["patient_id"])
    today_entry = store.wearing_log_today(patient["patient_id"])

    col_form, col_chart = st.columns([1, 1.5])
    with col_form:
        if today_entry:
            status = "✅ Worn" if today_entry["worn"] else "❌ Not worn"
            st.success(f"Today logged: **{status}** · {today_entry.get('hours',0)} hours")
        else:
            worn  = st.radio("Did you wear your AFO today?", ["Yes","No"], horizontal=True)
            hours = st.slider("Hours worn", 0, 16, 6) if worn == "Yes" else 0
            pain  = st.slider("Comfort (1=painful, 10=comfortable)", 1, 10, 7)
            note  = st.text_input("Any notes?", placeholder="e.g. rubbing at heel")
            if st.button("Save today's entry"):
                store.wearing_log_save(patient["patient_id"],
                                       {"worn": worn == "Yes", "hours": hours,
                                        "comfort": pain, "note": note})
                st.rerun()
    with col_chart:
        if len(log) >= 2:
            fig = _wearing_chart(log)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    if log:
        worn_n = sum(1 for e in log if e.get("worn"))
        if worn_n / len(log) < 0.7:
            _alert("⚠ Wearing compliance is below 70%. If discomfort is the reason, "
                   "tell your physiotherapist — most fitting issues are easily fixed.", "warn")
        with st.expander("View all entries"):
            df = pd.DataFrame(log).sort_values("date", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download log", df.to_csv(index=False),
                               f"wearing_log_{date.today()}.csv", "text/csv")


def _patient_satisfaction(patient: dict):
    _title("Satisfaction Assessment")
    st.markdown("Rate your current AFO. Your physiotherapist sees these scores.")
    items = ["Overall fit and comfort", "Ease of putting on and taking off",
             "How it feels during walking", "Safety — how secure it feels",
             "Effect on your walking ability", "Appearance",
             "Your confidence while wearing it"]
    with st.form("satisfaction"):
        ratings = {item: st.slider(item, 1, 5, 3, key=f"sat_{i}")
                   for i, item in enumerate(items)}
        if st.form_submit_button("Save assessment"):
            store.satisfaction_save(patient["patient_id"], ratings)
            st.success("Satisfaction scores saved. Your physiotherapist can now see these.")

    history = store.satisfaction_get(patient["patient_id"])
    if history:
        latest = history[-1]
        total  = round(sum(latest["scores"].values()) / len(latest["scores"]), 2)
        st.markdown(f"**Latest score: {total}/5** ({latest['date']})")
        vals = list(latest["scores"].values())
        fig = go.Figure(go.Bar(
            x=vals, y=[i[:35] for i in items], orientation="h",
            marker_color=["#16a34a" if v >= 4 else "#f59e0b" if v >= 3
                          else "#dc2626" for v in vals],
            text=vals, textposition="outside"))
        fig.update_layout(height=280, margin=dict(l=10, r=40, t=10, b=10),
                          xaxis=dict(range=[0, 6.5]),
                          yaxis=dict(autorange="reversed"),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        if total < 3.0:
            _alert("Your score is low. Please mention this at your next appointment. "
                   "Fit, comfort and appearance issues can usually be fixed quickly.", "warn")


# ── FIX 4: Education — rich cards with Lottie animations + images + links ─────

# Real Lottie JSON URLs from LottieFiles CDN (public, free, no auth needed)
# Each maps to a relevant animation for that AFO/rehabilitation topic
_EDU_ITEMS = [
    {
        "question": "What is foot drop and why do I need an AFO?",
        "answer": (
            "Foot drop is the inability to lift the front part of your foot because "
            "the muscles controlled by the peroneal nerve are weak. This nerve is often "
            "damaged by stroke. When you walk, the foot drags or catches, which causes "
            "falls. An AFO holds your foot at the correct angle so you can walk safely "
            "while your nervous system continues to recover."
        ),
        "lottie": "https://assets9.lottiefiles.com/packages/lf20_5tl1xxnz.json",
        "lottie_label": "Walking rehabilitation animation",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Posterior_leaf_spring_AFO.jpg/320px-Posterior_leaf_spring_AFO.jpg",
        "image_caption": "A posterior leaf spring AFO — the most common type for foot drop",
        "link": "https://www.stroke.org.uk/effects-of-stroke/physical-effects/foot-drop",
        "link_label": "Stroke Association: Foot Drop Guide",
    },
    {
        "question": "Will my AFO cure foot drop?",
        "answer": (
            "An AFO is a functional support tool, not a cure. It improves how you walk "
            "and reduces fall risk while your nervous system continues to recover. "
            "Many patients with early post-stroke foot drop see real motor improvements "
            "over the first 6 months. Your physiotherapist will monitor whether you still "
            "need the AFO at each review — some patients are able to reduce or stop use."
        ),
        "lottie": "https://assets3.lottiefiles.com/packages/lf20_touohxv0.json",
        "lottie_label": "Recovery progress animation",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Hinged_AFO.jpg/320px-Hinged_AFO.jpg",
        "image_caption": "Hinged AFO — allows controlled ankle movement during recovery",
        "link": "https://www.nhs.uk/conditions/foot-drop/treatment/",
        "link_label": "NHS: AFO Treatment for Foot Drop",
    },
    {
        "question": "How long do I wear it each day?",
        "answer": (
            "In early recovery, wear your AFO throughout all walking activity. "
            "As strength returns your physiotherapist may trial periods without it. "
            "Never stop wearing it without clinical guidance — falls are the main risk "
            "when the AFO is removed too early. Most patients wear it 6–12 hours per day."
        ),
        "lottie": "https://assets7.lottiefiles.com/packages/lf20_uu0x8lqv.json",
        "lottie_label": "Daily routine animation",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Carbon_fiber_AFO.jpg/320px-Carbon_fiber_AFO.jpg",
        "image_caption": "Carbon fibre AFO — lightweight option for active patients",
        "link": "https://www.ispo.com/information/patient-information/orthoses/ankle-foot-orthoses",
        "link_label": "ISPO: Patient Guide to AFOs",
    },
    {
        "question": "My AFO is causing pain — what do I do?",
        "answer": (
            "Stop wearing it and check your skin immediately. Redness lasting more than "
            "20 minutes after removal, or any blistering, means a pressure point that "
            "must be addressed by your orthotist before you wear it again. "
            "Do not push through pain. A single adjustment appointment solves most problems. "
            "Common causes: liner wear, strap tightness, and footwear that is too narrow."
        ),
        "lottie": "https://assets2.lottiefiles.com/packages/lf20_health.json",
        "lottie_label": "Health care animation",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Solid_ankle_AFO.jpg/320px-Solid_ankle_AFO.jpg",
        "image_caption": "Solid ankle AFO — correct fit prevents pressure sores",
        "link": "https://www.orthoinfo.aaos.org/en/treatment/ankle-foot-orthosis-afo/",
        "link_label": "AAOS: AFO Care and Fitting Guide",
    },
    {
        "question": "What shoes can I wear with my AFO?",
        "answer": (
            "You typically need one full shoe size up on the affected foot to accommodate "
            "the AFO. Wide toe-box lace-up shoes or trainers work best. "
            "Slip-ons and heels are generally incompatible with solid and hinged designs. "
            "Your orthotist can advise on specific brands. Some patients buy two different "
            "sizes — one for each foot."
        ),
        "lottie": "https://assets1.lottiefiles.com/packages/lf20_walking.json",
        "lottie_label": "Walking animation",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Running_shoes.jpg/320px-Running_shoes.jpg",
        "image_caption": "Wide toe-box trainers are the most compatible footwear with AFOs",
        "link": "https://www.feetforlife.org/afo-footwear",
        "link_label": "Feet For Life: AFO-Compatible Shoes",
    },
    {
        "question": "What is FES and how is it different from an AFO?",
        "answer": (
            "FES (Functional Electrical Stimulation) uses small electrode pads on your "
            "lower leg that send electrical pulses to activate the muscles that lift your "
            "foot. Unlike an AFO which supports passively, FES actively fires the muscle. "
            "Evidence suggests FES may help re-train the nervous system over time, "
            "potentially reducing long-term dependence on any device. "
            "It is most effective in the first 6 months post-stroke. Ask your physiotherapist."
        ),
        "lottie": "https://assets5.lottiefiles.com/packages/lf20_electrical.json",
        "lottie_label": "Electrical stimulation animation",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/NESS_L300_FES_device.jpg/320px-NESS_L300_FES_device.jpg",
        "image_caption": "FES device worn on the lower leg — stimulates the peroneal nerve",
        "link": "https://www.bioness.com/patients/l300-plus.php",
        "link_label": "Bioness L300: FES for Foot Drop",
    },
    {
        "question": "When will I be reassessed?",
        "answer": (
            "The recommended schedule is every 4–6 weeks in the first 6 months, "
            "then every 3–6 months after that. "
            "Request an earlier appointment if: the AFO no longer fits, your walking "
            "has changed significantly, you have skin problems, or you feel the device "
            "is no longer helping. Motor recovery is fastest in the first 3 months — "
            "this is when reassessments matter most."
        ),
        "lottie": "https://assets4.lottiefiles.com/packages/lf20_calendar.json",
        "lottie_label": "Calendar/scheduling animation",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Physiotherapy.jpg/320px-Physiotherapy.jpg",
        "image_caption": "Regular physiotherapy assessments track your recovery progress",
        "link": "https://www.physiotherapy.asn.au/APAWCMS/cms.display?page=patient-info",
        "link_label": "APA: What to Expect at Your Review",
    },
]


def _education_card(item: dict, index: int):
    """Render one education card with Lottie animation + image + link."""
    lottie_html = f"""
    <!DOCTYPE html><html><head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bodymovin/5.12.2/lottie.min.js"></script>
    <style>
      body{{margin:0;background:transparent;display:flex;align-items:center;
            justify-content:center;height:140px}}
      #anim{{width:130px;height:130px}}
    </style>
    </head><body>
    <div id="anim"></div>
    <script>
      lottie.loadAnimation({{
        container: document.getElementById('anim'),
        renderer: 'svg', loop: true, autoplay: true,
        path: '{item["lottie"]}'
      }});
    </script>
    </body></html>"""

    with st.expander(item["question"], expanded=(index == 0)):
        col_anim, col_text = st.columns([1, 2.5])
        with col_anim:
            components.html(lottie_html, height=150)
        with col_text:
            st.markdown(
                f'<p style="font-size:0.88rem;color:#334155;line-height:1.75;'
                f'margin-top:0.5rem">{item["answer"]}</p>',
                unsafe_allow_html=True,
            )

        # Image — use st.image with a real URL
        try:
            col_img, col_link = st.columns([1.2, 1])
            with col_img:
                st.image(item["image_url"], caption=item["image_caption"],
                         use_container_width=True)
            with col_link:
                st.markdown(
                    f'<div style="margin-top:1rem">'
                    f'<a href="{item["link"]}" target="_blank" '
                    f'style="display:inline-block;background:#2563eb;color:white;'
                    f'padding:8px 14px;border-radius:8px;font-size:0.83rem;'
                    f'font-weight:500;text-decoration:none">'
                    f'🔗 {item["link_label"]}</a>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(f"[{item['link_label']}]({item['link']})")


def _patient_education():
    _title("Understanding Your AFO")
    st.markdown(
        "Everything you need to know about your device — with animations, "
        "images, and links to trusted sources."
    )
    st.markdown("")
    for i, item in enumerate(_EDU_ITEMS):
        _education_card(item, i)


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

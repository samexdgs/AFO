# AFO Clinical Management Platform

**Dual-dashboard ankle-foot orthosis management for physiotherapists and patients**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-url.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Clinical Rationale

This platform directly addresses five documented gaps in the literature:

**Gap 1 — No objective prescription tool.**
Lee et al. (2021) conducted the first ML study predicting AFO need in 474 consecutive stroke patients. Their feature schema — MRC dorsiflexor score, Brunnstrom classification, Functional Ambulation Category — forms the foundation of this platform's clinical assessment form.

**Gap 2 — Premature prescription during rapid motor recovery.**
Motor function can improve from MRC 1 to MRC 4 within two weeks of stroke onset, rendering an AFO unnecessary. The platform flags reassessment triggers automatically when MRC ≥ 3 and days post-onset > 60.

**Gap 3 — Systematic abandonment.**
A 2025 cross-sectional study (PMC12317181) found that of 100 patients advised to use an AFO, only 59 were wearing one. Patients consulted 3–4 providers before prescription. This platform's prescription summary reduces that to a single focused orthotist appointment.

**Gap 4 — Expectation mismatch.**
59% of patients in that study disliked their AFO's appearance; 87% paid out of pocket. The patient portal explains each device's appearance and cost range before the patient receives it — addressing the leading driver of dissatisfaction.

**Gap 5 — No physiotherapist-patient data loop.**
The IoT/ML literature (Healthcare 2024, PMC11593354) shows that sending analysis results to the clinician's device for validation and then making them accessible to the patient significantly improves outcomes. This platform implements that feedback loop in software.

---

## Architecture

```
afo-platform/
├── app.py          Main Streamlit application — both dashboards
├── clinical.py     AFO knowledge base, ML model, clinical logic
├── store.py        Data persistence layer (JSON → replaceable with SQL)
├── requirements.txt
└── data/           Auto-created on first run
    ├── physios.json
    ├── patients.json
    ├── assessments.json
    ├── wearing_logs.json
    └── satisfaction.json
```

`store.py` is intentionally isolated — every function maps to a single database operation. Replacing JSON with PostgreSQL requires changes only in this file.

---

## ML Model

**Training data:** 4,000 synthetic records structured on the Lee et al. (2021) feature schema (PMC8055674), extended with stiffness scoring features from Frontiers Rehabilitation Sciences (2024).

**Features (17):** Age · Days post-onset · Sex · Diagnosis · MRC dorsiflexors · Brunnstrom stage · FAC score · MAS ankle · Ankle ROM · Knee stability · Skin sensation · Crouch gait · Varus/valgus · Weight · Activity level · Pacemaker · Days-post bucket

**Output classes (6):** PLS-AFO · Solid Ankle · Hinged · Carbon Fibre · GRAFO · FES

**Models:** Logistic Regression · Random Forest · LightGBM — best by ROC-AUC is selected automatically.

---

## Public Datasets for Future Clinical Validation

| Dataset | Source | Content |
|---|---|---|
| StrokeRehab | simtk.org/projects/primseq | 51 stroke patients, IMU + video |
| UI-PRMD | archive.ics.uci.edu | Physical rehab movements, skeleton data |
| KIMORE | University of Florence | Kinematic data + physiotherapist scores |
| Mendeley IUB Stroke | data.mendeley.com/datasets/ygpdzx52g2 | 3D depth + IMU, IRB approved |
| PhysioNet | physionet.org | Clinical time-series across conditions |

---

## Deployment

```bash
git clone https://github.com/samueloluwakoya/afo-platform.git
cd afo-platform
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Cloud: connect repo, set main file to `app.py`, deploy.

---

## Academic Citation

> **Samuel Oluwakoya** (2026). *AFO Clinical Management Platform: A Dual-Dashboard Machine Learning System for Physiotherapist-Led Ankle-Foot Orthosis Prescription and Patient Compliance Monitoring*. GitHub. https://github.com/samueloluwakoya/afo-platform

**References:**
- Lee SH et al. Machine learning analysis to predict the need for ankle foot orthosis in patients with stroke. *Scientific Reports* 2021. PMC8055674
- Shefa FR et al. Deep Learning and IoT-Based AFO for Enhanced Gait Optimization. *Healthcare* 2024. PMC11593354
- Assessing patient use and satisfaction with AFOs. *PMC* 2025. PMC12317181
- Evidence-guided AFO stiffness adjustment algorithm. *Frontiers Rehabilitation Sciences* 2024

---

## Medical Disclaimer

This platform provides ML-generated clinical guidance. Final AFO selection requires assessment by a qualified orthotist or rehabilitation physician. Not validated for standalone clinical use.

**Author: Samuel Oluwakoya** — Computer scientist, AI health researcher, foot drop patient.

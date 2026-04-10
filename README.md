https://afoplat.streamlit.app/
# AFO Clinical Management Platform

A dual-dashboard clinical tool for physiotherapists and patients built around machine learning-based ankle-foot orthosis prescription. One login for the clinician, a separate login for the patient. What happens on one side is visible on the other.

Built by Samuel Oluwakoya as part of an ongoing series of rehabilitation tools. Samuel has foot drop, which gives him both the clinical perspective of a patient and the technical perspective of someone who has read the research and built systems to address what the research identifies as unsolved.

---

## Background and motivation

A 2021 study published in Scientific Reports was the first to apply machine learning to the problem of predicting AFO need in stroke patients. The authors used 474 consecutive stroke patients and showed that early clinical variables like MRC dorsiflexor score, Brunnstrom classification, and Functional Ambulation Category could predict whether a patient would still need an AFO at six months post-stroke. Before that study, there was no predictive tool at all.

A 2025 cross-sectional study found that of 100 patients advised to use an AFO, only 59 were actually wearing one. Patients had to consult three to four healthcare providers before anyone prescribed the device. The mean wait from measurement to delivery was 19 days. 59% of users disliked the appearance of their AFO. 87% paid for it themselves.

These are not edge cases. They are the norm. This platform was built to address them directly.

The physiotherapist side provides ML-based prescription, compliance risk profiling, stiffness scoring, reassessment alerts, and a formatted clinical summary the patient takes to their orthotist. The patient side shows their recommendation in plain language, a daily wearing log with streak tracking, a QUEST-based satisfaction assessment that feeds back to the clinician, and an education module with Lottie animations and trusted external links.

---

## What the ML model does

Three classifiers are trained on 4,000 synthetic records structured around the Lee et al. (2021) feature schema, extended with stiffness scoring features from the Frontiers Rehabilitation Sciences 2024 evidence-guided algorithm. The best model by ROC-AUC is selected automatically.

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | varies by run | auto-evaluated |
| Random Forest | varies by run | auto-evaluated |
| LightGBM | varies by run | auto-evaluated |

17 clinical features are used: age, days post-onset, sex, diagnosis, MRC dorsiflexors, Brunnstrom stage, FAC score, MAS ankle, ankle ROM, knee stability, skin sensation, crouch gait, varus/valgus, weight, activity level, pacemaker presence, and days post-onset bucket.

6 output classes: PLS-AFO, Solid Ankle, Hinged, Carbon Fibre, GRAFO, and FES.

---

## Tech stack

- Python 3.10
- Streamlit (dual-dashboard web interface)
- scikit-learn (Logistic Regression, Random Forest)
- LightGBM
- Plotly (gauge charts, bar charts, wearing compliance charts)
- lottie-web via CDN (education module animations)
- pandas, numpy
- SHA-256 password hashing
- JSON file persistence via /tmp/ for Streamlit Cloud compatibility

---

## Project structure

```
afo-platform/
├── app.py          All UI logic, both dashboards
├── clinical.py     AFO type definitions, ML model, stiffness scoring, compliance risks
├── store.py        Data persistence, all read/write operations isolated here
├── requirements.txt
└── assets/         Lottie JSON animations and AFO diagram image
    ├── Ankle_Brace__1_.json
    ├── Ankle_Brace__2_.json
    ├── TENS_devices.json
    ├── Man_Walk.json
    ├── Man_Walking_On_Treadmill.json
    ├── Man_Doing_Calf_Raise_Exercise_for_Legs.json
    ├── Man_Rolling_Calf_Exercise_For_Leg_Relaxed.json
    ├── Side_Right_Leg_Raises_Exercise.json
    ├── Woman_Doing_Dumbbell_...json
    ├── Woman_Doing_Resistance_Band_...json
    ├── Leg_Pull_Low_Plank.json
    ├── Plank_Lunges_Exercise.json
    └── hpls_afo.png
```


## Public datasets for clinical validation

| Dataset | Source | Content |
|---|---|---|
| StrokeRehab | simtk.org/projects/primseq | 51 stroke patients, IMU and video |
| UI-PRMD | archive.ics.uci.edu | Physical rehabilitation movements, skeleton data |
| KIMORE | University of Florence | Kinematic data with physiotherapist scores |
| Mendeley IUB Stroke | data.mendeley.com/datasets/ygpdzx52g2 | 3D depth and IMU, IRB approved |
| PhysioNet | physionet.org | Clinical time-series across conditions |

---

## Research papers that informed this build

Lee SH, Lee SY, Kim DK. Machine learning analysis to predict the need for ankle foot orthosis in patients with stroke. Scientific Reports 2021. PMC8055674. This was the first ML study on AFO prediction in stroke and its feature schema directly structures the clinical assessment form in this platform.

Shefa FR et al. Deep Learning and IoT-Based Ankle-Foot Orthosis for Enhanced Gait Optimization. Healthcare 2024. PMC11593354. Used to inform the stiffness scoring algorithm and the IoT feedback loop architecture.

Assessing patient use and satisfaction with ankle foot orthoses and service. PMC 2025. PMC12317181. The source of the abandonment statistics (only 59 of 100 patients still wearing their AFO) that motivated the compliance risk and satisfaction modules.

Evidence-guided AFO stiffness adjustment algorithm. Frontiers Rehabilitation Sciences 2024. Used for the stiffness score calculation and the orthotist question set in the prescription export.

---

## Where this fits in the wider project

1. Foot Drop Management App — live at fdmapp.streamlit.app, the first published tool
2. Stroke Recovery Progress Tracker — daily ML-based monitoring
3. Stroke Recovery Monitor v2 — adds family dashboard and email alerts
4. AFO Clinical Management Platform (this tool) — dual-dashboard prescription system
5. NeuroKinetics — camera-based upper limb kinematic tracking, no specialist hardware

---

## Academic citation

Samuel Oluwakoya (2026). AFO Clinical Management Platform: A Dual-Dashboard Machine Learning System for Physiotherapist-Led Ankle-Foot Orthosis Prescription and Patient Compliance Monitoring. GitHub. (https://github.com/samexdgs/AFO)


## Disclaimer

This platform provides ML-generated clinical guidance. Final AFO selection requires assessment by a qualified orthotist or rehabilitation physician. This is a research tool and has not been validated as a medical device.

---
Samuel Oluwakoya — computer science graduate, foot drop patient, AI health researcher.

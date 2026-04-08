"""
clinical.py
AFO type definitions, clinical decision logic, and ML model.

Feature schema based on: Lee et al. (2021) PMC8055674 — the first published ML
study predicting AFO need in 474 consecutive stroke patients, using MRC score,
Brunnstrom classification, and functional ambulation category as primary predictors.

Extended with stiffness features from: Frontiers Rehabilitation Sciences (2024)
evidence-guided AFO adjustment algorithm.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import streamlit as st


# ── AFO type definitions ──────────────────────────────────────────────────────

AFO_TYPES = {
    "pls": {
        "name": "Posterior Leaf Spring",
        "abbreviation": "PLS-AFO",
        "rigidity": "Flexible",
        "profile": "Low",
        "weight_class": "Light",
        "appearance_score": 8,
        "compliance_risk": "Low",
        "cost_usd": "$30–$90",
        "cost_ngn": "₦18,000–₦55,000",
        "evidence": "Strong",
        "gait_phases": ["Swing phase", "Initial contact"],
        "indicated_when": [
            "Mild to moderate foot drop (MRC dorsiflexors 1–2)",
            "Ankle ROM at or above neutral (≥ 0°)",
            "Spasticity absent or minimal (MAS 0–1)",
            "Community or home ambulator",
        ],
        "contraindicated_when": [
            "Spasticity MAS ≥ 3",
            "Significant varus or valgus deformity",
            "Body weight above 100 kg",
        ],
        "patient_description": (
            "A thin curved plastic shell that fits along the back of your calf and "
            "into your shoe. Lightweight and discreet — most patients can wear it with "
            "standard footwear one size larger on the affected side."
        ),
    },
    "solid": {
        "name": "Solid Ankle AFO",
        "abbreviation": "SA-AFO",
        "rigidity": "Rigid",
        "profile": "Moderate",
        "weight_class": "Moderate",
        "appearance_score": 5,
        "compliance_risk": "Moderate",
        "cost_usd": "$40–$120",
        "cost_ngn": "₦24,000–₦72,000",
        "evidence": "Strong",
        "gait_phases": ["Swing phase", "Loading response", "Mid-stance", "Terminal stance"],
        "indicated_when": [
            "Moderate to severe foot drop",
            "Ankle instability with varus or valgus",
            "Spasticity MAS 2–3",
            "Requires full stance-phase support",
        ],
        "contraindicated_when": [
            "Fixed plantar flexion contracture",
            "Active dorsiflexion present (MRC ≥ 3)",
            "Primary stair-climbing activity requirement",
        ],
        "patient_description": (
            "A firm plastic brace encasing your foot and lower leg. "
            "Provides maximum stability for both the swing and stance phases of walking. "
            "Requires wider shoes — your physiotherapist will advise on footwear."
        ),
    },
    "hinged": {
        "name": "Hinged (Articulated) AFO",
        "abbreviation": "H-AFO",
        "rigidity": "Adjustable",
        "profile": "Moderate",
        "weight_class": "Moderate",
        "appearance_score": 6,
        "compliance_risk": "Moderate",
        "cost_usd": "$70–$200",
        "cost_ngn": "₦42,000–₦120,000",
        "evidence": "Strong",
        "gait_phases": ["All phases with appropriate stiffness setting"],
        "indicated_when": [
            "Moderate foot drop with preserved ankle ROM",
            "Spasticity MAS 2–3 requiring controlled dorsiflexion",
            "Active rehabilitation with expected motor recovery",
            "Patient requires stair climbing",
        ],
        "contraindicated_when": [
            "Fixed plantar flexion contracture",
            "Limited access to orthotist for adjustment",
        ],
        "patient_description": (
            "A polypropylene brace with a mechanical hinge at the ankle. "
            "Unlike a solid brace, the hinge allows some ankle movement — "
            "producing a more natural walking pattern. The resistance can be "
            "adjusted by your orthotist as your recovery progresses."
        ),
    },
    "carbon": {
        "name": "Carbon Fibre AFO",
        "abbreviation": "CF-AFO",
        "rigidity": "Dynamic",
        "profile": "Very low",
        "weight_class": "Very light",
        "appearance_score": 9,
        "compliance_risk": "Low",
        "cost_usd": "$160–$500",
        "cost_ngn": "₦96,000–₦300,000",
        "evidence": "Moderate",
        "gait_phases": ["Swing phase", "Push-off"],
        "indicated_when": [
            "Mild to moderate foot drop",
            "Active or community-ambulatory patient",
            "Strong preference for discretion and appearance",
            "Ankle ROM ≥ 5° dorsiflexion",
        ],
        "contraindicated_when": [
            "Spasticity MAS ≥ 3",
            "Significant varus or valgus",
            "Limited financial means without subsidy",
        ],
        "patient_description": (
            "A lightweight carbon fibre shell — the most discreet AFO available. "
            "Extremely thin and fits in most standard shoes with minimal modification. "
            "Also stores and releases energy during walking, which reduces fatigue "
            "compared to plastic designs."
        ),
    },
    "grafo": {
        "name": "Ground Reaction AFO",
        "abbreviation": "GRAFO",
        "rigidity": "Rigid — anterior shell",
        "profile": "High",
        "weight_class": "Heavy",
        "appearance_score": 3,
        "compliance_risk": "High",
        "cost_usd": "$100–$300",
        "cost_ngn": "₦60,000–₦180,000",
        "evidence": "Moderate",
        "gait_phases": ["Loading response", "Mid-stance", "Terminal stance"],
        "indicated_when": [
            "Crouch gait (excessive knee flexion during stance)",
            "Weak quadriceps combined with foot drop",
            "Both knee and ankle instability present",
        ],
        "contraindicated_when": [
            "Pure foot drop without knee involvement",
            "Plantar flexion contracture",
            "Primarily swing-phase deficit only",
        ],
        "patient_description": (
            "A larger brace that wraps around the front of your shin as well as "
            "supporting your foot and ankle. Specifically designed for patients who "
            "walk with a stooped or crouching pattern due to weak knee muscles."
        ),
    },
    "fes": {
        "name": "Functional Electrical Stimulation",
        "abbreviation": "FES",
        "rigidity": "None — wearable electronics",
        "profile": "Minimal",
        "weight_class": "Very light",
        "appearance_score": 10,
        "compliance_risk": "Low",
        "cost_usd": "$400–$1,600",
        "cost_ngn": "₦240,000–₦960,000",
        "evidence": "Strong",
        "gait_phases": ["Swing phase via timed peroneal nerve stimulation"],
        "indicated_when": [
            "Early post-stroke (onset < 6 months) — therapeutic neuroplasticity window",
            "Intact skin sensation at electrode site",
            "No implanted cardiac device",
            "Patient motivation for active rehabilitation",
        ],
        "contraindicated_when": [
            "Implanted cardiac pacemaker or defibrillator",
            "Active epilepsy",
            "Skin breakdown at electrode site",
            "Fixed plantar flexion contracture",
        ],
        "patient_description": (
            "Small electrode pads placed on your lower leg that deliver brief electrical "
            "pulses to activate the muscles that lift your foot. Unlike a brace, FES "
            "actively stimulates the muscle — which may help re-train your nervous system "
            "and reduce long-term dependence on an external device."
        ),
    },
}

LABEL_TO_KEY = {0: "pls", 1: "solid", 2: "hinged", 3: "carbon", 4: "grafo", 5: "fes"}

# ── Feature schema (matches PMC8055674 and Frontiers 2024) ────────────────────

FEATURES = [
    "age",
    "days_post_onset",
    "sex_num",            # 0=female, 1=male
    "diagnosis_num",      # 0=ischemic stroke, 1=hemorrhagic stroke, 2=other
    "mrc_dorsiflexors",   # 0–5 Medical Research Council scale
    "brunnstrom_stage",   # 1–6 Brunnstrom motor recovery
    "fac_score",          # 0–5 Functional Ambulation Category
    "mas_ankle",          # 0–4 Modified Ashworth Scale
    "ankle_rom_df",       # degrees, -20 to +20
    "knee_stability",     # 0=unstable, 1=stable
    "skin_sensation",     # 0=impaired, 1=intact
    "crouch_gait",        # 0=absent, 1=present
    "varus_valgus",       # 0=none, 1=mild, 2=moderate, 3=severe
    "weight_kg",
    "activity_level",     # 0=bed-bound, 1=household, 2=community, 3=active
    "has_pacemaker",      # 0=no, 1=yes
    "days_post_onset_bucket",  # 0=<1mo, 1=1-3mo, 2=3-6mo, 3=>6mo
]


# ── Synthetic training data ───────────────────────────────────────────────────

def _generate(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age              = rng.integers(18, 85, n)
    days             = rng.integers(1, 730, n)
    sex              = rng.integers(0, 2, n)
    diagnosis        = rng.choice([0, 1, 2], n, p=[0.75, 0.20, 0.05])
    mrc              = rng.integers(0, 6, n)
    brunnstrom       = rng.integers(1, 7, n)
    fac              = rng.integers(0, 6, n)
    mas              = rng.integers(0, 5, n)
    rom              = rng.integers(-15, 21, n)
    knee             = rng.integers(0, 2, n)
    sensation        = rng.choice([0, 1], n, p=[0.2, 0.8])
    crouch           = rng.choice([0, 1], n, p=[0.8, 0.2])
    vv               = rng.integers(0, 4, n)
    weight           = rng.integers(45, 130, n)
    activity         = rng.integers(0, 4, n)
    pacemaker        = rng.choice([0, 1], n, p=[0.96, 0.04])
    bucket           = np.where(days < 30, 0, np.where(days < 90, 1, np.where(days < 180, 2, 3)))

    labels = np.zeros(n, dtype=int)
    for i in range(n):
        if pacemaker[i]:
            labels[i] = 1  # solid — FES contraindicated
        elif crouch[i] and not knee[i]:
            labels[i] = 4  # grafo
        elif days[i] < 180 and sensation[i] and mas[i] <= 1 and not pacemaker[i]:
            labels[i] = 5  # fes — early window
        elif activity[i] >= 2 and mrc[i] >= 1 and mas[i] <= 1 and rom[i] >= 5:
            labels[i] = 3  # carbon
        elif mas[i] >= 3 or vv[i] >= 2:
            labels[i] = 1  # solid
        elif mas[i] >= 2 and mrc[i] <= 2 and rom[i] < 10:
            labels[i] = 2  # hinged
        else:
            labels[i] = 0  # pls

    return pd.DataFrame({f: v for f, v in zip(FEATURES, [
        age, days, sex, diagnosis, mrc, brunnstrom, fac, mas, rom,
        knee, sensation, crouch, vv, weight, activity, pacemaker, bucket,
    ])} | {"label": labels})


# ── Model training ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading clinical prediction models…")
def load_models():
    df = _generate(4000)
    X, y = df[FEATURES], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)

    lr = LogisticRegression(max_iter=2000, random_state=42).fit(Xtr_s, ytr)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    lgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, random_state=42, verbose=-1).fit(Xtr, ytr)

    results = {
        "Logistic Regression": {
            "model": lr, "scaler": scaler, "scaled": True,
            "accuracy": accuracy_score(yte, lr.predict(Xte_s)),
            "roc_auc": roc_auc_score(yte, lr.predict_proba(Xte_s), multi_class="ovr"),
        },
        "Random Forest": {
            "model": rf, "scaler": None, "scaled": False,
            "accuracy": accuracy_score(yte, rf.predict(Xte)),
            "roc_auc": roc_auc_score(yte, rf.predict_proba(Xte), multi_class="ovr"),
        },
        "LightGBM": {
            "model": lgbm, "scaler": None, "scaled": False,
            "accuracy": accuracy_score(yte, lgbm.predict(Xte)),
            "roc_auc": roc_auc_score(yte, lgbm.predict_proba(Xte), multi_class="ovr"),
        },
    }
    best = max(results, key=lambda k: results[k]["roc_auc"])
    return results, best


def predict(models: dict, best: str, row: dict) -> tuple[int, list]:
    m = models[best]
    X = pd.DataFrame([row])[FEATURES]
    if m["scaled"]:
        X = m["scaler"].transform(X)
    label = int(m["model"].predict(X)[0])
    proba = m["model"].predict_proba(X)[0].tolist()
    return label, proba


# ── Stiffness scoring ─────────────────────────────────────────────────────────

def stiffness_score(weight_kg: float, mas: int, rom: int, mrc: int, activity: int) -> float:
    score = 0.0
    score += 1.0 if weight_kg < 60 else (2.5 if weight_kg < 80 else (4.0 if weight_kg < 100 else 5.5))
    score += mas * 0.9
    score += 0 if rom >= 15 else (1.5 if rom >= 5 else (3.0 if rom >= 0 else 4.5))
    score += (5 - mrc) * 0.6
    score -= activity * 0.4
    return round(min(max(score, 0), 10), 1)


# ── Compliance risk ───────────────────────────────────────────────────────────

def compliance_risks(row: dict) -> list:
    risks = []
    if row.get("mas_ankle", 0) >= 3:
        risks.append({
            "factor": "High spasticity",
            "detail": "MAS ≥ 3 makes donning difficult and increases pressure-point pain.",
            "action": "Warm soak or 15 min slow stretch before fitting. Consider hinged AFO.",
        })
    if row.get("activity_level", 1) == 0:
        risks.append({
            "factor": "Very low activity level",
            "detail": "Sedentary patients perceive less functional benefit and discontinue earlier.",
            "action": "Set a concrete daily goal: one walk to the kitchen and back. Track it.",
        })
    if row.get("varus_valgus", 0) >= 2:
        risks.append({
            "factor": "Moderate–severe varus/valgus",
            "detail": "Deformity creates focal pressure points and skin breakdown risk.",
            "action": "Custom mould is mandatory. Add medial/lateral wedging to footplate.",
        })
    if not row.get("skin_sensation"):
        risks.append({
            "factor": "Impaired skin sensation",
            "detail": "Patient cannot feel pressure sores developing under the AFO.",
            "action": "Daily skin inspection of all contact areas. Remove for rest periods.",
        })
    return risks


# ── Reassessment triggers ─────────────────────────────────────────────────────

def reassessment_alerts(days_post: int, mrc: int) -> list:
    alerts = []
    if days_post < 30:
        alerts.append("Early post-stroke: motor recovery is rapid. Reassess in 2 weeks.")
    elif days_post < 180:
        alerts.append(
            "Active recovery phase: MRC scores may improve significantly. "
            "Reassess monthly. Consider FES if not already trialled."
        )
    if mrc >= 3 and days_post > 60:
        alerts.append(
            "MRC dorsiflexors ≥ 3: patient may no longer need an AFO, "
            "or may be able to downgrade to a lighter PLS or carbon fibre design. "
            "Conduct a walking trial without the device."
        )
    return alerts

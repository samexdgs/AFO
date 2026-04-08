"""
store.py
All data persistence for the AFO Clinical Platform.

Uses /tmp/afo_data/ as the base directory. On Streamlit Cloud, /tmp/ survives
page reruns and is shared across reloads within the same deployment.
Data resets when the app redeploys — acceptable for a research platform.
For production, replace each function body with a database call.
"""

import json
import hashlib
import os
from datetime import date, datetime
from typing import Optional

_DIR = "/tmp/afo_data"


def _p(f: str) -> str:
    os.makedirs(_DIR, exist_ok=True)
    return os.path.join(_DIR, f)


def _r(f: str) -> dict:
    path = _p(f)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def _w(f: str, d: dict) -> None:
    with open(_p(f), "w") as fh:
        json.dump(d, fh, indent=2)


def _hash(v: str) -> str:
    return hashlib.sha256(v.encode()).hexdigest()


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def physio_register(username: str, full_name: str, clinic: str, password: str) -> bool:
    users = _r("physios.json")
    key   = username.lower()
    if key in users:
        return False
    users[key] = {"username": key, "full_name": full_name, "clinic": clinic,
                  "password_hash": _hash(password), "created": str(date.today())}
    _w("physios.json", users)
    return True


def physio_authenticate(username: str, password: str) -> Optional[dict]:
    u = _r("physios.json").get(username.lower())
    return u if (u and u["password_hash"] == _hash(password)) else None


def physio_get(username: str) -> Optional[dict]:
    return _r("physios.json").get(username.lower())


def patient_register(physio_username: str, patient_id: str, full_name: str,
                     age: int, sex: str, diagnosis: str, days_post_onset: int,
                     password: str) -> bool:
    patients = _r("patients.json")
    key      = patient_id.lower()
    if key in patients:
        return False
    patients[key] = {"patient_id": key, "physio_username": physio_username.lower(),
                     "full_name": full_name, "age": age, "sex": sex,
                     "diagnosis": diagnosis, "days_post_onset": days_post_onset,
                     "password_hash": _hash(password),
                     "registered": str(date.today()), "active": True}
    _w("patients.json", patients)
    return True


def patient_authenticate(patient_id: str, password: str) -> Optional[dict]:
    p = _r("patients.json").get(patient_id.lower())
    return p if (p and p["password_hash"] == _hash(password)) else None


def patient_get(patient_id: str) -> Optional[dict]:
    return _r("patients.json").get(patient_id.lower())


def patients_for_physio(physio_username: str) -> list:
    return [p for p in _r("patients.json").values()
            if p["physio_username"] == physio_username.lower() and p.get("active", True)]


def assessment_save(patient_id: str, assessment: dict) -> None:
    records = _r("assessments.json")
    key     = patient_id.lower()
    if key not in records:
        records[key] = []
    assessment["timestamp"] = _ts()
    assessment["date"]      = str(date.today())
    records[key].append(assessment)
    _w("assessments.json", records)


def assessments_get(patient_id: str) -> list:
    return _r("assessments.json").get(patient_id.lower(), [])


def assessment_latest(patient_id: str) -> Optional[dict]:
    h = assessments_get(patient_id)
    return h[-1] if h else None


def wearing_log_save(patient_id: str, entry: dict) -> None:
    logs  = _r("wearing_logs.json")
    key   = patient_id.lower()
    today = str(date.today())
    if key not in logs:
        logs[key] = []
    logs[key] = [e for e in logs[key] if e.get("date") != today]
    entry["date"] = today
    entry["timestamp"] = _ts()
    logs[key].append(entry)
    _w("wearing_logs.json", logs)


def wearing_log_get(patient_id: str) -> list:
    return _r("wearing_logs.json").get(patient_id.lower(), [])


def wearing_log_today(patient_id: str) -> Optional[dict]:
    today = str(date.today())
    return next((e for e in wearing_log_get(patient_id) if e.get("date") == today), None)


def satisfaction_save(patient_id: str, scores: dict) -> None:
    records = _r("satisfaction.json")
    key     = patient_id.lower()
    if key not in records:
        records[key] = []
    records[key].append({"date": str(date.today()), "scores": scores, "timestamp": _ts()})
    _w("satisfaction.json", records)


def satisfaction_get(patient_id: str) -> list:
    return _r("satisfaction.json").get(patient_id.lower(), [])

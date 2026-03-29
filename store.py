"""
store.py
Handles all data persistence using JSON files on disk.
In a production deployment, each function here maps 1:1 to a database query.
Replacing JSON with PostgreSQL requires only changes to this file.
"""

import json
import hashlib
import os
from datetime import date, datetime
from typing import Optional

_DIR = "data"


def _path(filename: str) -> str:
    os.makedirs(_DIR, exist_ok=True)
    return os.path.join(_DIR, filename)


def _read(filename: str) -> dict:
    p = _path(filename)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def _write(filename: str, data: dict) -> None:
    with open(_path(filename), "w") as f:
        json.dump(data, f, indent=2)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Physiotherapist accounts ──────────────────────────────────────────────────

def physio_register(username: str, full_name: str, clinic: str, password: str) -> bool:
    users = _read("physios.json")
    key = username.lower()
    if key in users:
        return False
    users[key] = {
        "username": key,
        "full_name": full_name,
        "clinic": clinic,
        "password_hash": _hash(password),
        "created": str(date.today()),
    }
    _write("physios.json", users)
    return True


def physio_authenticate(username: str, password: str) -> Optional[dict]:
    users = _read("physios.json")
    u = users.get(username.lower())
    if u and u["password_hash"] == _hash(password):
        return u
    return None


def physio_get(username: str) -> Optional[dict]:
    return _read("physios.json").get(username.lower())


# ── Patient accounts ──────────────────────────────────────────────────────────

def patient_register(
    physio_username: str,
    patient_id: str,
    full_name: str,
    age: int,
    sex: str,
    diagnosis: str,
    days_post_onset: int,
    password: str,
) -> bool:
    patients = _read("patients.json")
    key = patient_id.lower()
    if key in patients:
        return False
    patients[key] = {
        "patient_id": key,
        "physio_username": physio_username.lower(),
        "full_name": full_name,
        "age": age,
        "sex": sex,
        "diagnosis": diagnosis,
        "days_post_onset": days_post_onset,
        "password_hash": _hash(password),
        "registered": str(date.today()),
        "active": True,
    }
    _write("patients.json", patients)
    return True


def patient_authenticate(patient_id: str, password: str) -> Optional[dict]:
    patients = _read("patients.json")
    p = patients.get(patient_id.lower())
    if p and p["password_hash"] == _hash(password):
        return p
    return None


def patient_get(patient_id: str) -> Optional[dict]:
    return _read("patients.json").get(patient_id.lower())


def patients_for_physio(physio_username: str) -> list:
    patients = _read("patients.json")
    return [
        p for p in patients.values()
        if p["physio_username"] == physio_username.lower() and p.get("active", True)
    ]


# ── Clinical assessments (physiotherapist-entered) ────────────────────────────

def assessment_save(patient_id: str, assessment: dict) -> None:
    records = _read("assessments.json")
    key = patient_id.lower()
    if key not in records:
        records[key] = []
    assessment["timestamp"] = _ts()
    assessment["date"] = str(date.today())
    records[key].append(assessment)
    _write("assessments.json", records)


def assessments_get(patient_id: str) -> list:
    return _read("assessments.json").get(patient_id.lower(), [])


def assessment_latest(patient_id: str) -> Optional[dict]:
    history = assessments_get(patient_id)
    return history[-1] if history else None


# ── Wearing compliance log (patient-entered) ──────────────────────────────────

def wearing_log_save(patient_id: str, entry: dict) -> None:
    logs = _read("wearing_logs.json")
    key = patient_id.lower()
    if key not in logs:
        logs[key] = []
    today = str(date.today())
    logs[key] = [e for e in logs[key] if e.get("date") != today]
    entry["date"] = today
    entry["timestamp"] = _ts()
    logs[key].append(entry)
    _write("wearing_logs.json", logs)


def wearing_log_get(patient_id: str) -> list:
    return _read("wearing_logs.json").get(patient_id.lower(), [])


def wearing_log_today(patient_id: str) -> Optional[dict]:
    today = str(date.today())
    return next((e for e in wearing_log_get(patient_id) if e.get("date") == today), None)


# ── Satisfaction scores (patient-entered, QUEST-based) ────────────────────────

def satisfaction_save(patient_id: str, scores: dict) -> None:
    records = _read("satisfaction.json")
    key = patient_id.lower()
    if key not in records:
        records[key] = []
    records[key].append({"date": str(date.today()), "scores": scores, "timestamp": _ts()})
    _write("satisfaction.json", records)


def satisfaction_get(patient_id: str) -> list:
    return _read("satisfaction.json").get(patient_id.lower(), [])

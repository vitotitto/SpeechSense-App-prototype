"""
SpeechSense Monitor - Patient Store
===================================
Manages the per-patient JSON files for storing longitudinal data.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PATIENT_DATA_DIR = Path(os.environ.get("PATIENT_DATA_DIR", "patient_data"))

def ensure_store():
    PATIENT_DATA_DIR.mkdir(parents=True, exist_ok=True)

def _get_patient_path(patient_id: str) -> Path:
    return PATIENT_DATA_DIR / f"{patient_id}.json"

def list_patients() -> List[str]:
    """Returns a list of all patient IDs in the store."""
    ensure_store()
    patients = []
    for p in PATIENT_DATA_DIR.glob("*.json"):
        patients.append(p.stem)
    return sorted(patients)

def get_patient(patient_id: str) -> Optional[Dict]:
    """Loads a patient's data, or returns None if not found."""
    ext_path = _get_patient_path(patient_id)
    if not ext_path.exists():
        return None
    try:
        with open(ext_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {patient_id}: {e}")
        return None

def save_patient(patient_data: Dict) -> None:
    """Saves a patient's data to disk."""
    ensure_store()
    patient_id = patient_data.get("patient_id")
    if not patient_id:
        raise ValueError("Patient data missing 'patient_id'")
        
    ext_path = _get_patient_path(patient_id)
    with open(ext_path, "w", encoding="utf-8") as f:
        json.dump(patient_data, f, indent=2, ensure_ascii=False)

def create_patient(patient_id: str) -> Dict:
    """Creates a new patient record. Returns existing if already present."""
    existing = get_patient(patient_id)
    if existing:
        return existing
        
    now = datetime.utcnow().isoformat() + "Z"
    new_patient = {
        "patient_id": patient_id,
        "created_at": now,
        "sessions": [],
        "scores": []
    }
    save_patient(new_patient)
    return new_patient

def add_session(patient_id: str, session_data: Dict) -> Dict:
    """Adds a session to a patient's record."""
    patient = get_patient(patient_id)
    if not patient:
        patient = create_patient(patient_id)
        
    patient["sessions"].append(session_data)
    save_patient(patient)
    return patient

def add_score(patient_id: str, score_data: Dict) -> Dict:
    """Adds a new score to a patient's history."""
    patient = get_patient(patient_id)
    if not patient:
         patient = create_patient(patient_id)
         
    patient["scores"].append(score_data)
    save_patient(patient)
    return patient

def _get_embeddings_path(patient_id: str) -> Path:
    return PATIENT_DATA_DIR / f"{patient_id}_embeddings.npy"


def save_embeddings(patient_id: str, new_embeddings: np.ndarray) -> int:
    """Appends new embeddings to the patient's .npy file.

    Args:
        patient_id: Patient identifier.
        new_embeddings: Array of shape (N, 2560) with new chunk embeddings.

    Returns:
        Total number of accumulated embeddings after saving.
    """
    ensure_store()
    new_embeddings = np.asarray(new_embeddings, dtype=np.float32)
    if new_embeddings.ndim == 1:
        new_embeddings = new_embeddings.reshape(1, -1)

    emb_path = _get_embeddings_path(patient_id)
    if emb_path.exists():
        existing = np.load(emb_path)
        combined = np.vstack([existing, new_embeddings])
    else:
        combined = new_embeddings

    np.save(emb_path, combined)
    return combined.shape[0]


def load_embeddings(patient_id: str) -> np.ndarray:
    """Loads all accumulated embeddings for a patient.

    Returns:
        Array of shape (N, 2560), or empty (0, 2560) if none exist.
    """
    emb_path = _get_embeddings_path(patient_id)
    if emb_path.exists():
        return np.load(emb_path)
    return np.empty((0, 2560), dtype=np.float32)


def clear_patient_data(patient_id: str) -> None:
    """Deletes a patient's embeddings and resets their JSON (scores/sessions)."""
    # Remove embeddings file
    emb_path = _get_embeddings_path(patient_id)
    if emb_path.exists():
        os.remove(emb_path)

    # Reset JSON record (keep patient_id and created_at)
    patient = get_patient(patient_id)
    if patient:
        patient["sessions"] = []
        patient["scores"] = []
        save_patient(patient)

def get_score_history(patient_id: str) -> List[Dict]:
    """Retrieves the score history for a patient."""
    patient = get_patient(patient_id)
    if not patient:
        return []
    return patient.get("scores", [])

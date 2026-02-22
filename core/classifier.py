"""
SpeechSense Monitor - Classifier
================================
Loads the pre-trained Logistic Regression model and scaler, aggregates
embeddings, and predicts the cognitive risk score.
"""

import os
import pickle
from typing import Dict, List, Union
from pathlib import Path

import numpy as np

# Path to the models directory (relative to this file)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

class RiskClassifier:
    def __init__(self):
        self.scaler = None
        self.clf = None
        self.feature_spec = None
        self._is_loaded = False
        
    def load(self):
        """Loads the pre-trained multimodal model."""
        if self._is_loaded:
            return
            
        model_path = MODELS_DIR / "text_acoustic_narrative_model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}."
            )
            
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            
        self.scaler = data['scaler']
        self.clf = data['classifier']
        self.feature_spec = data.get('feature_spec', None)
            
        self._is_loaded = True
        
    def score_patient(self, features: Union[Dict, np.ndarray], is_speaker_level: bool = False) -> Dict:
        """
        Takes features for a patient and predicts the cognitive risk score.

        features can be:
          - dict with 'medgemma' (N, 2560) array (from extract())
          - 1D array of shape (5120,) if is_speaker_level=True (pre-aggregated)

        Aggregation follows the training protocol:
          [medgemma_mean(2560), medgemma_std(2560)] = 5120
        """
        if not self._is_loaded:
            self.load()

        # Handle dict input from feature_extractor.extract()
        if isinstance(features, dict):
            medgemma = np.asarray(features["medgemma"], dtype=np.float32)

        elif isinstance(features, np.ndarray) and not is_speaker_level:
            # Raw (N, 2560) accumulated embeddings
            medgemma = np.asarray(features, dtype=np.float32)
            if medgemma.ndim == 1:
                medgemma = medgemma.reshape(1, -1)

        elif is_speaker_level:
            # Pre-aggregated speaker-level features (5120 dims)
            feature_vector = np.asarray(features, dtype=np.float32)
            n_chunks = 1
            medgemma = None
        else:
            return {"score": 0.0, "n_chunks": 0, "confidence_level": "none"}

        if medgemma is not None:
            n_chunks = medgemma.shape[0]

            if n_chunks == 0:
                return {"score": 0.0, "n_chunks": 0, "confidence_level": "none"}

            # Aggregate: [mean, std]
            mg_mean = np.mean(medgemma, axis=0)
            mg_std = np.std(medgemma, axis=0) if n_chunks > 1 else np.zeros_like(mg_mean)

            # Concat: [mean(2560), std(2560)] = 5120
            feature_vector = np.concatenate([mg_mean, mg_std])
        
        # Ensure 2D for scaler
        feature_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # Predict probability of class 1 (Dementia risk)
        probability = self.clf.predict_proba(feature_scaled)[0, 1]
        
        # Determine confidence level based on number of chunks
        if n_chunks < 5:
            confidence = "low"
        elif n_chunks < 15:
            confidence = "moderate"
        else:
            confidence = "good"
            
        return {
            "score": float(probability),
            "n_chunks": n_chunks,
            "confidence_level": confidence
        }

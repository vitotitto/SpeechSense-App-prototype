"""
SpeechSense Monitor - Gradio Application Entry Point
====================================================
"""

import os
import re
import time
from datetime import datetime
from typing import List, Tuple

import gradio as gr
import urllib3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from dotenv import load_dotenv

# Suppress warnings from requests
urllib3.disable_warnings()

from core.transcript_service import transcribe_audio
from core.feature_extractor import MultimodalFeatureExtractor
from core.classifier import RiskClassifier
from core import patient_store

# Load environment variables (API Key)
load_dotenv()

risk_classifier = RiskClassifier()
feature_extractor = MultimodalFeatureExtractor()

PATIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
SESSION_COLUMNS = ["Date/Time", "Recordings", "Score", "Band"]

APP_CSS = """
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap");

:root {
    --bg-0: #f8fafc;
    --bg-1: #f1f5f9;
    --surface: #ffffff;
    --surface-soft: #fbfdff;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --line: #e2e8f0;
    --accent: #2563eb;
    --accent-dark: #1d4ed8;
    --accent-soft: #eff6ff;
    --danger: #dc2626;
    --danger-soft: #fef2f2;
    --warning: #ea580c;
    --warning-soft: #fff7ed;
    --success: #059669;
    --success-soft: #ecfdf5;
    
    --radius-xl: 24px;
    --radius-lg: 16px;
    --radius-md: 12px;
    --radius-sm: 8px;
    
    --space-major: 32px;
    --space-card: 20px;
    --space-inner: 16px;
}

body {
    background: linear-gradient(135deg, var(--bg-0), var(--bg-1));
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
}

.gradio-container {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 2rem 1rem !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: "Outfit", current-font, sans-serif !important;
    letter-spacing: -0.02em;
    color: var(--text-primary);
}

/* Hero Section */
#hero-card {
    border-radius: var(--radius-xl);
    padding: 2.5rem;
    margin-bottom: var(--space-major);
    background: linear-gradient(120deg, var(--surface) 0%, #f1f5f9 100%);
    border: 1px solid rgba(255, 255, 255, 0.4);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 24px 48px -12px rgba(15, 23, 42, 0.08);
    position: relative;
    overflow: hidden;
}

#hero-card::before {
    content: '';
    position: absolute;
    top: 0; right: 0; left: 0;
    height: 4px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4, #10b981);
}

#hero-card h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

#hero-card p {
    margin: 0.75rem 0 0 0;
    color: var(--text-secondary);
    font-size: 1.125rem;
    max-width: 600px;
}

#hero-chip {
    display: inline-flex;
    align-items: center;
    margin-top: 1.5rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-radius: 9999px;
    padding: 0.375rem 0.875rem;
    color: var(--accent-dark);
    background: var(--accent-soft);
    border: 1px solid rgba(59, 130, 246, 0.2);
}

/* Layout Zones */
.zone-header {
    font-size: 0.8125rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    color: #64748b;
    margin: 0 0 1.5rem 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.zone-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--line);
}

#workspace-grid {
    gap: 2.5rem !important;
}

/* Cards */
.section-card {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--line) !important;
    background: var(--surface) !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.03) !important;
    padding: 1.75rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.section-card > div {
    gap: var(--space-inner) !important;
}

.section-card h3 {
    margin: 0.25rem 0 1.25rem 0 !important;
    font-size: 1.25rem !important;
    color: #1e293b;
}

.step-label {
    display: inline-block;
    font-size: 0.84rem; /* 20% larger than 0.70rem */
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.3rem 0.72rem; /* Adjusted padding and line height slightly */
    border-radius: var(--radius-sm);
    color: #64748b;
    background: #f1f5f9;
    margin-bottom: 0.75rem;
}

/* Margins */
#input-zone .section-card,
#output-zone .section-card {
    margin-bottom: var(--space-major);
}
#output-zone #details-card { margin-bottom: 0; }

/* Form Helpers */
.form-help {
    font-size: 0.875rem;
    line-height: 1.5;
    border-radius: var(--radius-md);
    padding: 0.75rem 1rem;
    border: 1px solid var(--line);
    background: var(--surface-soft);
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.form-help.success {
    border-color: #a7f3d0;
    background: var(--success-soft);
    color: var(--success);
}

.form-help.warning {
    border-color: #fed7aa;
    background: var(--warning-soft);
    color: var(--warning);
}

.form-help.error {
    border-color: #fecaca;
    background: var(--danger-soft);
    color: var(--danger);
}

/* Buttons */
.gr-button, button.gr-button, #action-row button, #create-btn button, #confirm-clear-btn button {
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 0.75rem 1.25rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}

#create-btn button {
    background: #0f172a !important;
    border: 1px solid #0f172a !important;
    color: #ffffff !important;
    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.2), 0 2px 4px -2px rgba(15, 23, 42, 0.1) !important;
}
#create-btn button:hover {
    background: #1e293b !important;
    border-color: #1e293b !important;
    transform: translateY(-1px);
    box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.25), 0 4px 6px -4px rgba(15, 23, 42, 0.1) !important;
}

#analyze-btn button {
    background: var(--accent) !important;
    border: 1px solid var(--accent-dark) !important;
    color: white !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2), 0 2px 4px -2px rgba(37, 99, 235, 0.1) !important;
}
#analyze-btn button:hover:not(:disabled) {
    background: var(--accent-dark) !important;
    transform: translateY(-1px);
    box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.25), 0 4px 6px -4px rgba(37, 99, 235, 0.1) !important;
}

#confirm-clear-btn button {
    background: white !important;
    border: 1px solid #fda4af !important;
    color: var(--danger) !important;
}
#confirm-clear-btn button:hover:not(:disabled) {
    background: var(--danger-soft) !important;
    border-color: #f87171 !important;
}
button:disabled {
    opacity: 0.6 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Input Fields */
input, textarea {
    border-radius: var(--radius-md) !important;
    border-color: #cbd5e1 !important;
    padding: 0.625rem 0.875rem !important;
    font-size: 0.875rem !important; /* Slightly smaller to prevent cut-offs */
}
input:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}

/* File Upload Box */
.gr-file {
    border-radius: var(--radius-lg) !important;
    border: 2px dashed #cbd5e1 !important;
    background: #f8fafc !important;
    transition: all 0.2s ease !important;
}
.gr-file:hover {
    border-color: var(--accent) !important;
    background: var(--accent-soft) !important;
}

/* Status Panel & Chips */
#status-card {
    border-style: dashed !important;
    border-width: 2px !important;
    background: var(--surface-soft) !important;
}

.status-chip {
    border-radius: var(--radius-md);
    border: 1px solid var(--line);
    background: var(--surface);
    color: var(--text-secondary);
    font-size: 0.9375rem;
    padding: 0.75rem 1rem;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}
.status-chip.running {
    border-color: #93c5fd;
    background: var(--accent-soft);
    color: var(--accent-dark);
}
.status-chip.error {
    border-color: #fca5a5;
    background: var(--danger-soft);
    color: var(--danger);
}

/* Metric Cards */
.metric-card {
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    background: var(--surface);
    border: 1px solid var(--line);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0; width: 4px;
}
.metric-card.low { background: linear-gradient(180deg, var(--surface), #f8fafc); }
.metric-card.low::before { background: var(--success); }
.metric-card.elevated { background: linear-gradient(180deg, var(--surface), var(--warning-soft)); }
.metric-card.elevated::before { background: var(--warning); }
.metric-card.high { background: linear-gradient(180deg, var(--surface), var(--danger-soft)); }
.metric-card.high::before { background: var(--danger); }

.metric-kicker {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.metric-value {
    font-family: "Outfit", sans-serif;
    font-size: 3.5rem;
    line-height: 1;
    font-weight: 700;
    margin: 0.5rem 0 1rem 0;
    letter-spacing: -0.02em;
}

.signal-pill {
    display: inline-flex;
    align-items: center;
    border-radius: 9999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8125rem;
    font-weight: 600;
    border: 1px solid transparent;
}
.signal-pill.low { color: var(--success); background: var(--success-soft); border-color: #a7f3d0; }
.signal-pill.elevated { color: var(--warning); background: #ffedd5; border-color: #fed7aa; }
.signal-pill.high { color: var(--danger); background: var(--danger-soft); border-color: #fecaca; }

.metric-note {
    margin-top: 1rem;
    color: var(--text-secondary);
    font-size: 0.875rem;
    line-height: 1.5;
    padding-top: 1rem;
    border-top: 1px dashed var(--line);
}

/* Info Cards */
.info-card {
    border-radius: var(--radius-md);
    border: 1px solid var(--line);
    padding: 1.25rem;
    background: #f8fafc;
    color: #334155;
    line-height: 1.6;
    font-size: 0.95rem;
}
.info-card .headline {
    color: #0f172a;
    font-weight: 600;
    font-size: 1.05rem;
    display: block;
    margin-bottom: 0.5rem;
}

/* Empty State */
.empty-state {
    border-radius: var(--radius-lg);
    border: 2px dashed #cbd5e1;
    padding: 2.5rem 1.5rem;
    text-align: center;
    color: #64748b;
    background: #f8fafc;
}
.empty-state strong {
    color: #334155;
    font-size: 1.1rem;
    display: block;
    margin-bottom: 0.25rem;
}

/* Tables */
#history-table table {
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--line);
}
#history-table th {
    background: #f8fafc !important;
    color: #475569 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 1rem !important;
}
#history-table td {
    padding: 1rem !important;
    font-size: 0.9375rem !important;
    color: #334155 !important;
    border-bottom: 1px solid var(--line) !important;
}
#history-table tbody tr:hover td {
    background: #f1f5f9 !important;
}

/* Footer & Extras */
#footer-note {
    margin-top: 2rem;
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    background: var(--surface);
    border: 1px solid #e2e8f0;
    color: #64748b;
    font-size: 0.875rem;
    text-align: center;
}

/* Results Header */
#results-header-card {
    border-radius: var(--radius-md);
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 1rem 1.25rem;
    margin-bottom: var(--space-major);
}
.results-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.results-title {
    font-weight: 600;
    font-size: 1.125rem;
    color: var(--text-primary);
}
.results-status {
    font-size: 0.875rem;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.results-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #94a3b8;
}

@media (max-width: 980px) {
    #workspace-grid { flex-direction: column !important; }
    #hero-card { padding: 1.5rem; }
    .metric-value { font-size: 2.5rem; }
}
"""

# Helper Functions


def _trunc2(value: float) -> str:
    """Truncate to 2 decimal places without rounding (e.g. 0.9997 -> '0.99')."""
    import math
    return f"{math.floor(value * 100) / 100:.2f}"


def risk_style(score: float) -> Tuple[str, str, str, str]:
    """Maps a score to signal language, color, and visual tone."""
    if score < 0.3:
        return "Low signal", "#059669", "low", "Low"
    if score < 0.6:
        return "Moderate signal", "#ea580c", "elevated", "Moderate"
    return "High signal", "#dc2626", "high", "High"


def form_help_html(message: str, tone: str = "default") -> str:
    """Small inline helper message for forms."""
    tone_class = "" if tone == "default" else f" {tone}"
    return f"<div class='form-help{tone_class}'>{message}</div>"


def empty_state_html(title: str, subtitle: str) -> str:
    """Reusable empty-state card."""
    return f"""
    <div class="empty-state">
        <strong>{title}</strong><br>
        <span>{subtitle}</span>
    </div>
    """


def build_score_html(score: float) -> str:
    """Builds model-score card markup with research framing."""
    signal_text, color, tone, badge = risk_style(score)
    # Using generic coloring class and overriding color for inline text if necessary. 
    # The ::before pseudo-element takes care of the accent line in the new CSS.
    return f"""
    <div class="metric-card {tone}">
        <div class="metric-kicker">Model Score</div>
        <div class="metric-value" style="color: {color};">{_trunc2(score)}</div>
        <span class="signal-pill {tone}">{badge}</span>
        <div class="metric-note">{signal_text}. Research-use signal only; not diagnostic.</div>
    </div>
    """


def build_confidence_html(conf_level: str, total_recordings: int, total_sessions: int, total_new_recordings: int | None = None, n_files: int | None = None) -> str:
    """Builds confidence summary markup for current score context."""
    upload_line = ""
    if total_new_recordings is not None and n_files is not None:
        upload_line = f"<br>This run added {total_new_recordings} recording(s) from {n_files} file(s)."

    return f"""
    <div class="info-card">
        <span class="headline">Evidence Summary</span><br>
        Confidence level: {conf_level}.<br>
        Based on {total_recordings} cumulative recording(s) across {total_sessions} scored session(s).{upload_line}
    </div>
    """


def compact_status_html(message: str, tone: str = "running") -> str:
    """Compact status chip."""
    return f"<div class='status-chip {tone}'>{message}</div>"


def _parse_iso_timestamp(ts: str) -> datetime | None:
    """Parses ISO timestamps used in patient files."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Date extraction from filenames — for demonstration purposes
# ---------------------------------------------------------------------------
# Parses dates embedded in filenames (e.g. "CarolBurnett_15Jan2025.wav") so
# that the timeline plot can show separate date-labelled points per recording.

_MONTH_TYPO_MAP = {
    "janury": "january", "januray": "january", "janaury": "january",
    "febuary": "february", "febraury": "february", "febrary": "february",
    "febuary": "february",
    "martch": "march", "marhc": "march",
    "apirl": "april", "aprli": "april",
    "augst": "august", "agust": "august",
    "setpember": "september", "septmber": "september", "spetember": "september",
    "ocotber": "october", "octobr": "october",
    "novmber": "november", "novemeber": "november",
    "decmber": "december", "decemeber": "december",
}

# Pattern 1: named month — e.g. "15Jan2025", "3February2024"
_DATE_NAMED_MONTH_RE = re.compile(r"(\d{1,2})([A-Za-z]{3,9})(\d{4})")

# Pattern 2: numeric underscore-separated DD_MM_YYYY at the end of the stem
# e.g. "patient_speech_part_0001_10_11_2024" → 10/11/2024 (DD/MM/YYYY)
_DATE_NUMERIC_RE = re.compile(r"(\d{1,2})_(\d{1,2})_(\d{4})$")


def _correct_month_typo(month_str: str) -> str:
    """Attempts to correct common month-name misspellings."""
    lower = month_str.lower()
    return _MONTH_TYPO_MAP.get(lower, lower)


def extract_date_from_filename(filename: str) -> datetime | None:
    """
    Extracts a date from a filename.  Supports two formats:

    1. Named month:   'CarolBurnett_15Jan2025.wav'  → 15 January 2025
    2. Numeric DD_MM_YYYY at end: 'part_0001_10_11_2024.wav' → 10 November 2024

    Returns None if no recognisable date pattern is found.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]

    # Try named-month pattern first (e.g. "15Jan2025")
    m = _DATE_NAMED_MONTH_RE.search(stem)
    if m:
        day_str, month_raw, year_str = m.group(1), m.group(2), m.group(3)
        corrected_month = _correct_month_typo(month_raw)
        try:
            return datetime.strptime(f"{day_str} {corrected_month} {year_str}", "%d %B %Y")
        except ValueError:
            pass
        # Try abbreviated month (e.g. "Jan" instead of "January")
        try:
            return datetime.strptime(f"{day_str} {corrected_month} {year_str}", "%d %b %Y")
        except ValueError:
            pass

    # Try numeric DD_MM_YYYY at the end of the stem
    m = _DATE_NUMERIC_RE.search(stem)
    if m:
        day_str, month_str, year_str = m.group(1), m.group(2), m.group(3)
        try:
            return datetime(int(year_str), int(month_str), int(day_str))
        except ValueError:
            pass

    return None


def empty_session_table() -> pd.DataFrame:
    """Creates an empty dataframe with fixed columns for the session list."""
    return pd.DataFrame(columns=SESSION_COLUMNS)


def build_session_table(patient_id: str) -> Tuple[pd.DataFrame, List[str]]:
    """Builds table rows and detail markdown for scored sessions."""
    patient = patient_store.get_patient(patient_id) or {}
    scores = patient.get("scores", [])
    sessions = patient.get("sessions", [])
    if not scores:
        return empty_session_table(), []

    rows = []
    details = []
    prev_session_edge = 0

    for idx, score in enumerate(scores, start=1):
        score_val = float(score.get("score", 0.0))
        _, _, _, badge = risk_style(score_val)
        dt = _parse_iso_timestamp(score.get("timestamp", ""))
        date_str = dt.strftime("%Y-%m-%d %H:%M") if dt else "Unknown"
        new_chunks = int(score.get("n_new_recordings", score.get("n_new_chunks", score.get("n_total_chunks", 0))))
        conf = str(score.get("confidence_level", "unknown")).upper()
        total_chunks = int(score.get("n_total_chunks", 0))

        declared_edge = score.get("n_sessions")
        if isinstance(declared_edge, int) and declared_edge >= prev_session_edge:
            session_edge = min(declared_edge, len(sessions))
        else:
            session_edge = min(prev_session_edge + 1, len(sessions))

        linked_sessions = sessions[prev_session_edge:session_edge]
        if not linked_sessions and prev_session_edge < len(sessions):
            linked_sessions = [sessions[prev_session_edge]]
            session_edge = prev_session_edge + 1
        prev_session_edge = session_edge

        detail_lines = []
        for sess in linked_sessions[:3]:
            transcript = (sess.get("transcript") or "").replace("\n", " ").strip()
            if len(transcript) > 180:
                transcript = transcript[:177] + "..."
            file_name = sess.get("audio_filename", "audio")
            detail_lines.append(f"- **{file_name}**: {transcript or 'Transcript unavailable.'}")

        if len(linked_sessions) > 3:
            detail_lines.append(f"- ... {len(linked_sessions) - 3} more recording(s)")
        if not detail_lines:
            detail_lines.append("- No recording transcript captured for this scored session.")

        details.append(
            "\n".join(
                [
                    f"### Session {idx} | {date_str}",
                    f"- Model score: **{_trunc2(score_val)}** ({badge})",
                    f"- Confidence: **{conf}**",
                    f"- Cumulative chunks: **{total_chunks}**",
                    f"- New chunks this run: **{new_chunks}**",
                    "",
                    "#### Recordings",
                    *detail_lines,
                ]
            )
        )
        rows.append([date_str, new_chunks, _trunc2(score_val), badge])

    rows.reverse()
    details.reverse()
    return pd.DataFrame(rows, columns=SESSION_COLUMNS), details


def plot_score_timeline(patient_id: str):
    """Generates a matplotlib timeline of the patient's scores."""
    history = patient_store.get_score_history(patient_id) if patient_id else []

    fig, ax = plt.subplots(figsize=(8.8, 4.9), facecolor="#f8fcff")
    ax.set_facecolor("#ffffff")

    if not history:
        ax.text(
            0.5,
            0.5,
            "No timeline data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color="#5c6b79",
            fontsize=11,
        )
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_color("#d4e0e9")
        fig.tight_layout()
        return fig

    if len(history) < 2:
        single_score = float(history[0].get("score", 0.0))
        dt = _parse_iso_timestamp(history[0].get("timestamp", ""))
        x_label = dt.strftime("%Y-%m-%d") if dt else "Latest"

        ax.axhspan(0.0, 0.3, color="#0f766e", alpha=0.08, label="Typical control")
        ax.axhspan(0.3, 0.6, color="#b54708", alpha=0.08, label="Elevated")
        ax.axhspan(0.6, 1.0, color="#b42318", alpha=0.08, label="High")
        ax.scatter([0], [single_score], s=120, color="#0f766e", edgecolor="white", linewidth=2, zorder=3)
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0])
        ax.set_xticklabels([x_label], rotation=0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("Model score")
        ax.set_title("Trend chart (insufficient history)", color="#102d41", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.8, color="#d2dee7", alpha=0.9)
        ax.text(
            0.5,
            0.92,
            "Need at least 2 scored sessions to estimate trend.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#576879",
            fontsize=9,
        )
        for spine in ax.spines.values():
            spine.set_color("#d4e0e9")
        fig.tight_layout()
        return fig

    dates = []
    scores = []

    for score_entry in history:
        dt = _parse_iso_timestamp(score_entry.get("timestamp", ""))
        if dt is None:
            continue
        dates.append(pd.to_datetime(dt))
        scores.append(float(score_entry.get("score", 0.0)))

    if not dates:
        return plot_score_timeline("")

    # Plot data points and connecting lines
    ax.plot(
        dates,
        scores,
        marker="o",
        linestyle="-",
        color="#0f766e",
        linewidth=2.25,
        markersize=7,
        markerfacecolor="#ffffff",
        markeredgewidth=2,
    )
    
    # Reference bands based on generic illustrative thresholds
    ax.axhspan(0.0, 0.3, color="#0f766e", alpha=0.08, label="Typical control")
    ax.axhspan(0.3, 0.6, color="#b54708", alpha=0.08, label="Elevated")
    ax.axhspan(0.6, 1.0, color="#b42318", alpha=0.08, label="High")

    # Formatting
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Model score", color="#334654", fontsize=10)
    ax.set_title(f"Trend Chart for {patient_id}", color="#102d41", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.8, color="#d2dee7", alpha=0.9)
    ax.tick_params(axis="x", colors="#526272")
    ax.tick_params(axis="y", colors="#526272")
    for spine in ax.spines.values():
        spine.set_color("#d4e0e9")
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    plt.xticks(rotation=25)
    plt.tight_layout()

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

    return fig


def create_new_patient(new_id: str) -> Tuple[gr.Dropdown, str, gr.Textbox]:
    """Creates a new patient and updates dropdown + inline validation."""
    candidate = (new_id or "").strip()
    if not candidate:
        return gr.update(), form_help_html("Enter a patient ID first.", "warning"), gr.update()

    if not PATIENT_ID_PATTERN.match(candidate):
        return (
            gr.update(),
            form_help_html("Invalid ID. Use letters, numbers, hyphen, or underscore only.", "error"),
            gr.update(value=candidate),
        )

    patient_store.create_patient(candidate)
    patients = patient_store.list_patients()
    return (
        gr.update(choices=patients, value=candidate),
        form_help_html(f"Patient '{candidate}' is ready.", "success"),
        gr.update(value=""),
    )


def has_uploaded_files(audio_files) -> bool:
    """Checks whether Gradio file input contains at least one file."""
    if not audio_files:
        return False
    if isinstance(audio_files, str):
        return True
    if isinstance(audio_files, list):
        return len(audio_files) > 0
    return True


def update_action_controls(patient_id: str, audio_files, clear_confirmation: str):
    """Controls Analyze/Clear button interactivity and clear helper text."""
    can_analyze = bool(patient_id) and has_uploaded_files(audio_files)
    confirm_value = (clear_confirmation or "").strip()
    danger_visible = bool(patient_id)

    if can_analyze:
        analyze_msg = form_help_html("Ready. Click Analyze Recording to compute model score.", "success")
    elif patient_id and not has_uploaded_files(audio_files):
        analyze_msg = form_help_html("Upload one or more files to enable analysis.", "warning")
    else:
        analyze_msg = form_help_html("Select a patient and upload files to enable analysis.", "warning")

    if not patient_id:
        clear_msg = form_help_html("Select a patient before using destructive actions.", "warning")
        can_clear = False
    elif confirm_value != patient_id:
        clear_msg = form_help_html(f"Type '{patient_id}' to enable clear.", "warning")
        can_clear = False
    else:
        clear_msg = form_help_html("Confirmation matched. You can clear patient data.", "error")
        can_clear = True

    return (
        gr.update(interactive=can_analyze),
        gr.update(interactive=can_clear),
        clear_msg,
        analyze_msg,
        gr.update(visible=danger_visible),
    )


def reset_clear_confirmation():
    """Clears the confirmation input when patient context changes."""
    return gr.update(value="")


def start_processing_ui():
    """Shows processing card and prevents double-submit while running."""
    return (
        gr.update(visible=True),
        compact_status_html("Processing audio...", "running"),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def open_session_detail(details: List[str], evt):
    """Shows markdown details for a clicked session row."""
    if not details:
        return gr.update(value="", visible=False)

    idx = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
    if idx is None or idx < 0 or idx >= len(details):
        return gr.update(value="", visible=False)

    return gr.update(value=details[idx], visible=True)


def process_audio(patient_id: str, audio_files, progress=gr.Progress()):
    """
    Main pipeline (On-the-fly Multimodal):
    1. Transcribe each audio file (Pyannote) to get timestamps
    2. Extract Sequential Multimodal Features over chunks
    3. Persist embeddings, then score using ALL accumulated embeddings
    4. Store session & Update UI
    """
    start_time = time.time()

    if not patient_id:
        return (
            gr.HTML(empty_state_html("No patient selected", "Select a patient before analyzing audio.")),
            gr.HTML(empty_state_html("Awaiting analysis", "Upload files to run an analysis.")),
            plot_score_timeline(""),
            empty_session_table(),
            [],
            gr.update(value="", visible=False),
            gr.update(visible=False),
            compact_status_html("Please select a patient first.", "error"),
        )

    if not audio_files:
        return (
            gr.HTML(empty_state_html("No audio files provided", "Upload one or more .wav or .mp3 files.")),
            gr.HTML(empty_state_html("Awaiting analysis", "The model score appears after processing.")),
            plot_score_timeline(patient_id),
            *build_session_table(patient_id),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            compact_status_html("Please upload at least one audio file.", "error"),
        )

    # Normalise input: gr.File returns a single path or list of paths
    if isinstance(audio_files, str):
        audio_files = [audio_files]

    status_updates = []

    def status_callback(msg, progress_val=None, progress_desc=None):
        if progress_val is not None and progress_desc is not None:
             progress(progress_val, desc=progress_desc)
        status_updates.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    try:
        # Check prerequisites
        if not os.environ.get("PYANNOTE_API_KEY"):
            raise ValueError("PYANNOTE_API_KEY is not set. Check your .env file.")

        n_files = len(audio_files)
        all_transcripts = []

        # -----------------------------------------------------------------
        # Phase 1 — Extract: transcribe + embed each file (1 embedding each)
        # -----------------------------------------------------------------
        # Each entry: (file_label, embedding (1,2560), date or None)
        file_results = []

        for file_idx, audio_path in enumerate(audio_files):
            file_label = os.path.basename(audio_path)
            file_progress_base = file_idx / n_files
            file_progress_step = 1.0 / n_files

            # 1a. Transcribe
            status_callback(
                f"[File {file_idx+1}/{n_files}] Transcribing {file_label}...",
                progress_val=file_progress_base + file_progress_step * 0.1,
                progress_desc=f"Transcribing file {file_idx+1}/{n_files}"
            )
            transcript, segments = transcribe_audio(audio_path)

            if not segments:
                status_callback(f"[File {file_idx+1}/{n_files}] No speech segments in {file_label}, skipping.")
                continue

            # 1b. Extract single whole-file embedding
            def inner_callback(msg, progress_val=None, progress_desc=None):
                # Remap inner progress into this file's slice
                if progress_val is not None:
                    remapped = file_progress_base + file_progress_step * (0.2 + 0.7 * progress_val)
                    progress(remapped, desc=f"[{file_idx+1}/{n_files}] {progress_desc or msg}")
                status_updates.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

            features = feature_extractor.extract(audio_path, segments, inner_callback)
            file_embedding = features["medgemma"]  # (1, 2560)

            # 1c. Parse date from filename (for demonstration purposes)
            file_date = extract_date_from_filename(file_label)

            file_results.append({
                "label": file_label,
                "embedding": file_embedding,
                "date": file_date,
                "transcript": transcript,
            })

            all_transcripts.append(f"**{file_label}**:\n> {transcript}")

            # Save session metadata (no embeddings in JSON)
            ts = file_date.isoformat() + "Z" if file_date else (datetime.utcnow().isoformat() + "Z")
            session_data = {
                "session_id": f"sess_{int(time.time())}_{file_idx}",
                "timestamp": ts,
                "audio_filename": file_label,
                "transcript": transcript,
                "n_recordings": 1,
                "processing_time_s": round(time.time() - start_time, 1)
            }
            patient_store.add_session(patient_id, session_data)

        if not file_results:
            raise ValueError("No speech segments found in any of the uploaded files.")

        # -----------------------------------------------------------------
        # Phase 2 — Per-date independent scoring + cumulative overall
        # -----------------------------------------------------------------
        # Each date group is scored independently so the timeline shows
        # how the patient's speech varies across sessions.  The main score
        # card displays the cumulative score across all recordings.
        total_new_recordings = len(file_results)

        # Group files by date, sort chronologically
        from collections import defaultdict
        date_groups = defaultdict(list)
        for fr in file_results:
            key = fr["date"] or datetime.utcnow()
            date_groups[key].append(fr)

        sorted_dates = sorted(date_groups.keys())

        # Persist all new embeddings at once
        new_embeddings = np.vstack([fr["embedding"] for fr in file_results])
        status_callback(
            f"Saving {total_new_recordings} new recording embeddings...",
            progress_val=0.9, progress_desc="Saving embeddings"
        )
        total_accumulated = patient_store.save_embeddings(patient_id, new_embeddings)

        # Score each date group independently (only that group's recordings)
        status_callback(
            f"Scoring {len(sorted_dates)} date group(s) independently ({total_accumulated} total recordings)...",
            progress_val=0.93, progress_desc="Computing per-date scores"
        )

        existing_scores = patient_store.get_score_history(patient_id)

        for group_date in sorted_dates:
            group_files = date_groups[group_date]
            group_embeddings = np.vstack([gf["embedding"] for gf in group_files])

            group_result = risk_classifier.score_patient(
                group_embeddings, is_speaker_level=False
            )

            ts = group_date.isoformat() + "Z"
            score_entry = {
                "timestamp": ts,
                "score": group_result["score"],
                "n_total_chunks": group_result["n_chunks"],
                "n_new_recordings": len(group_files),
                "confidence_level": group_result["confidence_level"],
                "n_sessions": len(existing_scores) + 1,
            }
            patient_store.add_score(patient_id, score_entry)
            existing_scores = patient_store.get_score_history(patient_id)

        # Compute cumulative score across ALL accumulated recordings
        # (displayed on the main score card as the overall assessment)
        accumulated_embeddings = patient_store.load_embeddings(patient_id)
        status_callback(
            f"Computing cumulative score ({accumulated_embeddings.shape[0]} recordings)...",
            progress_val=0.97, progress_desc="Cumulative score"
        )
        cumulative_result = risk_classifier.score_patient(
            accumulated_embeddings, is_speaker_level=False
        )

        status_callback("Processing complete.", progress_val=1.0, progress_desc="Done")

        # -----------------------------------------------------------------
        # Format UI outputs
        # -----------------------------------------------------------------
        # Main score card shows cumulative; timeline shows per-date points
        score_val = cumulative_result["score"]
        conf_level = cumulative_result["confidence_level"].upper()
        total_recordings = cumulative_result["n_chunks"]
        session_table_df, session_details = build_session_table(patient_id)
        total_sessions = len(session_table_df)

        score_html = build_score_html(score_val)
        conf_html = build_confidence_html(
            conf_level=conf_level,
            total_recordings=total_recordings,
            total_sessions=total_sessions,
            total_new_recordings=total_new_recordings,
            n_files=n_files,
        )

        transcript_display = "### Latest Processing Details\n\n" + "\n\n---\n\n".join(all_transcripts)
        timeline_fig = plot_score_timeline(patient_id)

        return (
            gr.HTML(score_html),
            gr.HTML(conf_html),
            timeline_fig,
            session_table_df,
            session_details,
            gr.update(value=transcript_display, visible=True),
            gr.update(visible=False),
            compact_status_html("Processing complete.", "running"),
        )

    except Exception as e:
        status_updates.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {str(e)}")
        session_table_df, session_details = build_session_table(patient_id)
        details_md = "\n".join(
            [
                "### Processing failed",
                f"- Error: `{str(e)}`",
                "",
                "#### Latest status",
                *[f"- {line}" for line in status_updates[-5:]],
            ]
        )
        return (
            gr.HTML(empty_state_html("Analysis failed", "The pipeline could not complete this request.")),
            gr.HTML(f"<div class='info-card'><span class='headline'>Error</span><br>{str(e)}</div>"),
            plot_score_timeline(patient_id),
            session_table_df,
            session_details,
            gr.update(value=details_md, visible=True),
            gr.update(visible=False),
            compact_status_html(f"Processing failed: {str(e)}", "error"),
        )


def clear_data(patient_id: str, clear_confirmation: str):
    """Clears all embeddings, sessions, and scores for the selected patient."""
    if not patient_id:
        return (
            gr.HTML(empty_state_html("No patient selected", "Choose a patient before clearing data.")),
            gr.HTML(empty_state_html("No changes made", "Patient data remains unchanged.")),
            plot_score_timeline(""),
            empty_session_table(),
            [],
            gr.update(value="", visible=False),
            gr.update(visible=False),
            compact_status_html("Select a patient first.", "error"),
            gr.update(value=""),
            form_help_html("Select a patient before clearing data.", "warning"),
        )

    if (clear_confirmation or "").strip() != patient_id:
        session_df, session_details = build_session_table(patient_id)
        return (
            gr.HTML(empty_state_html("Confirmation required", "Type the patient ID before clearing data.")),
            gr.HTML(empty_state_html("No data cleared", "Destructive action is locked until confirmation matches.")),
            plot_score_timeline(patient_id),
            session_df,
            session_details,
            gr.update(value="", visible=False),
            gr.update(visible=False),
            compact_status_html("Clear blocked: confirmation did not match.", "error"),
            gr.update(value=clear_confirmation),
            form_help_html(f"Type '{patient_id}' exactly to enable clear.", "error"),
        )

    patient_store.clear_patient_data(patient_id)

    return (
        gr.HTML(empty_state_html("Data cleared", "Upload new files to restart longitudinal tracking.")),
        gr.HTML(f"<div class='info-card'><span class='headline'>{patient_id}</span><br>All sessions and score history were removed.</div>"),
        plot_score_timeline(patient_id),
        empty_session_table(),
        [],
        gr.update(value="", visible=False),
        gr.update(visible=False),
        compact_status_html(f"Patient '{patient_id}' data cleared.", "running"),
        gr.update(value=""),
        form_help_html(f"Type '{patient_id}' to enable clear.", "warning"),
    )


def refresh_patient_data(patient_id: str):
    """Refreshes the UI when a new patient is selected from the dropdown."""
    if not patient_id:
         return (
             gr.HTML(empty_state_html("No patient selected", "Select a patient profile to view score history.")),
             gr.HTML(empty_state_html("Awaiting analysis", "Model score appears after analysis.")),
             plot_score_timeline(""),
             empty_session_table(),
             [],
             gr.update(value="", visible=False),
             gr.update(visible=False),
             compact_status_html("Select a patient to begin.", "running"),
         )

    history = patient_store.get_score_history(patient_id)
    session_df, session_details = build_session_table(patient_id)

    if not history:
        # No prior data
        return (
             gr.HTML(empty_state_html("No model score yet", "Run the first analysis to generate outputs.")),
             gr.HTML(empty_state_html("Ready for first analysis", "Research-use output appears after scoring.")),
             plot_score_timeline(patient_id),
             session_df,
             session_details,
             gr.update(value="", visible=False),
             gr.update(visible=False),
             compact_status_html("Ready for analysis.", "running"),
        )

    last_score = history[-1]
    score_val = last_score.get("score", 0.0)
    conf_level = last_score.get("confidence_level", "unknown").upper()
    total_chunks = last_score.get("n_total_chunks", 0)
    total_sessions = len(history)

    score_html = build_score_html(score_val)
    conf_html = build_confidence_html(
        conf_level=conf_level,
        total_recordings=total_chunks,
        total_sessions=total_sessions,
    )

    return (
        gr.HTML(score_html),
        gr.HTML(conf_html),
        plot_score_timeline(patient_id),
        session_df,
        session_details,
        gr.update(value="", visible=False),
        gr.update(visible=False),
        compact_status_html("Patient loaded.", "running"),
    )

# --- Define Gradio Interface ---

with gr.Blocks(title="SpeechSense Monitor", css=APP_CSS) as demo:

    session_details_state = gr.State([])

    gr.HTML(
        """
        <section id="hero-card">
            <h1>SpeechSense Monitor</h1>
            <p>Longitudinal cognitive speech monitoring prototype for research workflows.</p>
            <div id="hero-chip">Research use only | Not for diagnosis</div>
        </section>
        """
    )

    with gr.Row(elem_id="workspace-grid"):
        with gr.Column(scale=5, min_width=360, elem_id="input-zone"):
            gr.HTML("<div class='zone-header'>Input Workflow</div>")

            with gr.Group(elem_classes=["section-card", "workflow-step"], elem_id="patient-card"):
                gr.HTML("<span class='step-label'>Patient</span>")
                with gr.Row(elem_id="patient-row"):
                    patient_dropdown = gr.Dropdown(
                        choices=patient_store.list_patients(),
                        label="Select Existing Patient",
                        interactive=True,
                        scale=2,
                    )
                    new_patient_input = gr.Textbox(
                        label="New Patient ID",
                        placeholder="Create new patient ID...",
                        scale=2,
                    )
                    create_btn = gr.Button("Create Patient", variant="primary", scale=1, elem_id="create-btn")

                create_help = gr.HTML(
                    value=form_help_html("Use letters, numbers, hyphen, or underscore only.", "default"),
                    elem_id="create-help",
                )

            with gr.Group(elem_classes=["section-card", "workflow-step"], elem_id="upload-card"):
                gr.HTML("<span class='step-label'>Recording Upload</span>")
                audio_upload = gr.File(
                    file_count="multiple",
                    type="filepath",
                    file_types=[".wav", ".mp3"],
                    label="Upload audio files (.wav/.mp3)",
                )

            with gr.Group(elem_classes=["section-card", "workflow-step"], elem_id="action-card"):
                gr.HTML("<span class='step-label'>Run Analysis</span>")
                with gr.Row(elem_id="action-row"):
                    analyze_btn = gr.Button(
                        "Analyze Recording",
                        variant="primary",
                        size="lg",
                        elem_id="analyze-btn",
                        interactive=False,
                    )

                analyze_help = gr.HTML(
                    value=form_help_html("Select a patient and upload files to enable analysis.", "warning"),
                    elem_id="analyze-help",
                )

                with gr.Accordion("Danger Zone: Clear Patient Data", open=False, visible=False, elem_id="danger-accordion") as danger_accordion:
                    clear_confirm_input = gr.Textbox(
                        label="Type selected patient ID to confirm",
                        placeholder="Type exact patient ID...",
                        elem_id="clear-confirm-input",
                    )
                    clear_guard_note = gr.HTML(
                        value=form_help_html("Select a patient before using destructive actions.", "warning"),
                        elem_id="clear-guard-note",
                    )
                    clear_btn = gr.Button(
                        "Confirm Clear",
                        variant="secondary",
                        elem_id="confirm-clear-btn",
                        interactive=False,
                    )

            with gr.Group(elem_classes=["section-card", "workflow-step"], elem_id="status-card", visible=False) as status_panel:
                gr.HTML("<span class='step-label'>Processing</span>")
                status_html = gr.HTML(value=compact_status_html("Processing...", "running"))

        with gr.Column(scale=7, min_width=420, elem_id="output-zone"):
            gr.HTML("<div class='zone-header'>Results</div>")
            gr.HTML(
                """
                <div id="results-header-card">
                    <div class="results-header-row">
                        <span class="results-title">Analysis Results</span>
                        <span class="results-status"><span class="results-dot"></span>Status in workflow panel</span>
                    </div>
                </div>
                """
            )

            with gr.Group(elem_classes=["section-card"], elem_id="score-card"):
                gr.HTML("<span class='step-label'>Score Summary</span>")
                score_display = gr.HTML(
                    value=empty_state_html("No model score yet", "Run analysis to populate results."),
                )
                confidence_display = gr.HTML(
                    value=empty_state_html("Awaiting evidence", "Confidence summary appears after scoring."),
                )

            with gr.Group(elem_classes=["section-card"], elem_id="timeline-card"):
                gr.HTML("<span class='step-label'>Trend Chart</span>")
                timeline_plot = gr.Plot(label="", value=plot_score_timeline(""))

            with gr.Group(elem_classes=["section-card"], elem_id="history-card"):
                gr.HTML("<span class='step-label'>Session List</span>")
                session_table = gr.Dataframe(
                    headers=SESSION_COLUMNS,
                    value=empty_session_table(),
                    datatype=["str", "number", "str", "str"],
                    row_count=(1, "dynamic"),
                    col_count=(4, "fixed"),
                    interactive=True,
                    elem_id="history-table",
                )

            with gr.Group(elem_classes=["section-card"], elem_id="details-card"):
                gr.HTML("<span class='step-label'>Session Details</span>")
                details_output = gr.Markdown(value="", visible=False, elem_id="details-output")

    gr.HTML(
        """
        <div id="footer-note">
            <strong>Disclaimer:</strong> This prototype is for research only. Outputs are not diagnostic and
            should not be used as standalone clinical decisions.
        </div>
        """
    )

    create_btn.click(
        fn=create_new_patient,
        inputs=[new_patient_input],
        outputs=[patient_dropdown, create_help, new_patient_input],
    ).then(
        fn=refresh_patient_data,
        inputs=[patient_dropdown],
        outputs=[
            score_display,
            confidence_display,
            timeline_plot,
            session_table,
            session_details_state,
            details_output,
            status_panel,
            status_html,
        ],
    ).then(
        fn=reset_clear_confirmation,
        inputs=[],
        outputs=[clear_confirm_input],
    ).then(
        fn=update_action_controls,
        inputs=[patient_dropdown, audio_upload, clear_confirm_input],
        outputs=[analyze_btn, clear_btn, clear_guard_note, analyze_help, danger_accordion],
    )

    patient_dropdown.change(
        fn=refresh_patient_data,
        inputs=[patient_dropdown],
        outputs=[
            score_display,
            confidence_display,
            timeline_plot,
            session_table,
            session_details_state,
            details_output,
            status_panel,
            status_html,
        ],
    ).then(
        fn=reset_clear_confirmation,
        inputs=[],
        outputs=[clear_confirm_input],
    ).then(
        fn=update_action_controls,
        inputs=[patient_dropdown, audio_upload, clear_confirm_input],
        outputs=[analyze_btn, clear_btn, clear_guard_note, analyze_help, danger_accordion],
    )

    audio_upload.change(
        fn=update_action_controls,
        inputs=[patient_dropdown, audio_upload, clear_confirm_input],
        outputs=[analyze_btn, clear_btn, clear_guard_note, analyze_help, danger_accordion],
    )

    clear_confirm_input.change(
        fn=update_action_controls,
        inputs=[patient_dropdown, audio_upload, clear_confirm_input],
        outputs=[analyze_btn, clear_btn, clear_guard_note, analyze_help, danger_accordion],
    )

    analyze_btn.click(
        fn=start_processing_ui,
        inputs=[],
        outputs=[status_panel, status_html, analyze_btn, clear_btn],
        queue=False,
    ).then(
        fn=process_audio,
        inputs=[patient_dropdown, audio_upload],
        outputs=[
            score_display,
            confidence_display,
            timeline_plot,
            session_table,
            session_details_state,
            details_output,
            status_panel,
            status_html,
        ],
    ).then(
        fn=update_action_controls,
        inputs=[patient_dropdown, audio_upload, clear_confirm_input],
        outputs=[analyze_btn, clear_btn, clear_guard_note, analyze_help, danger_accordion],
    )

    clear_btn.click(
        fn=clear_data,
        inputs=[patient_dropdown, clear_confirm_input],
        outputs=[
            score_display,
            confidence_display,
            timeline_plot,
            session_table,
            session_details_state,
            details_output,
            status_panel,
            status_html,
            clear_confirm_input,
            clear_guard_note,
        ],
    ).then(
        fn=update_action_controls,
        inputs=[patient_dropdown, audio_upload, clear_confirm_input],
        outputs=[analyze_btn, clear_btn, clear_guard_note, analyze_help, danger_accordion],
    )

    session_table.select(
        fn=open_session_detail,
        inputs=[session_details_state],
        outputs=[details_output],
    )


if __name__ == "__main__":
    
    # We must ensure MedGemma and the Classifier are reachable before launch.
    # We can pre-load them, but it takes ~30-60s which delays UI startup.
    # Therefore we try to load the classifier here (fast) to fail early if missing,
    # but leave MedGemma to load on first request.
    print("Checking classifier artifact requirements...")
    try:
         risk_classifier.load()
    except Exception as e:
         print(f"Failed to load classifier: {e}")
         print("Please ensure the multimodal model was saved correctly.")
         
         
    print("Starting Gradio Web Server...")
    # Share allows external access if needed, inbrowser opens default browser
    demo.launch(server_name="0.0.0.0", server_port=7861, inbrowser=True)


"""
Quick end-to-end pipeline test (no Gradio UI).
Run: conda activate audio && python test_pipeline.py
"""
import os, sys, time, traceback
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

AUDIO = r"D:\medgemma_data\dementianet\dementia\Donald Sterling\DonaldSterling_5.wav"

print("=" * 60)
print("SpeechSense Pipeline Test")
print("=" * 60)

# --- Step 1: Pyannote Transcription ---
print("\n[1/7] Pyannote transcription...")
try:
    from core.transcript_service import transcribe_audio
    transcript, segments = transcribe_audio(AUDIO)
    print(f"  OK: {len(segments)} segments, {len(transcript)} chars")
    for s in segments[:5]:
        dur = s['end'] - s['start']
        print(f"    seg {s['id']}: {s['start']:.1f}-{s['end']:.1f}s ({dur:.1f}s) speaker={s.get('speaker','')} \"{s.get('text','')[:70]}\"")
    if len(segments) > 5:
        print(f"    ... and {len(segments)-5} more")

    # Show full-transcript concatenation (no more chunk merging in the pipeline)
    full_transcript = " ".join(s.get("text", "") for s in segments).strip()
    print(f"\n  Full transcript: {len(full_transcript)} chars")
    print(f"    \"{full_transcript[:120]}...\"")
    print()
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Step 2: Feature Extraction (single whole-file embedding) ---
print("\n[2/7] Feature extraction (MedGemma text+acoustic narrative, whole file)...")
try:
    from core.feature_extractor import MultimodalFeatureExtractor
    extractor = MultimodalFeatureExtractor()
    t0 = time.time()
    features = extractor.extract(AUDIO, segments)
    elapsed = time.time() - t0
    print(f"  OK in {elapsed:.1f}s")
    print(f"  medgemma shape: {features['medgemma'].shape}")  # Expected: (1, 2560)
    print(f"  medgemma sample: min={features['medgemma'].min():.4f} max={features['medgemma'].max():.4f} mean={features['medgemma'].mean():.4f}")
    # Check for all-zeros (would indicate failed extraction)
    mg_nonzero = (features['medgemma'] != 0).any(axis=1).sum()
    print(f"  non-zero embeddings: medgemma={mg_nonzero}/{features['medgemma'].shape[0]}")
    assert features['medgemma'].shape == (1, 2560), f"Expected (1, 2560), got {features['medgemma'].shape}"
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Step 3: Classification (single-session, dict path) ---
print("\n[3/7] Classification (single-session, dict path)...")
try:
    from core.classifier import RiskClassifier
    clf = RiskClassifier()
    clf.load()
    result = clf.score_patient(features)
    print(f"  OK")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Chunks: {result['n_chunks']}")
    print(f"  Confidence: {result['confidence_level']}")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Step 4: Dimension verification ---
print("\n[4/7] Dimension verification...")
import numpy as np
mg = features['medgemma']
n = mg.shape[0]
mg_mean = np.mean(mg, axis=0)
mg_std = np.std(mg, axis=0) if n > 1 else np.zeros_like(mg_mean)
vec = np.concatenate([mg_mean, mg_std])
print(f"  Aggregated feature vector: {vec.shape[0]} dims")
print(f"  Scaler expects:            {clf.scaler.n_features_in_} dims")
print(f"  Match: {'YES' if vec.shape[0] == clf.scaler.n_features_in_ else 'NO - MISMATCH!'}")

# --- Step 5: Embedding persistence & cumulative scoring ---
print("\n[5/7] Embedding persistence & cumulative scoring...")
try:
    from core import patient_store

    TEST_PATIENT = "__test_pipeline__"

    # Clean up any prior test data
    patient_store.clear_patient_data(TEST_PATIENT)
    patient_store.create_patient(TEST_PATIENT)

    raw_embeddings = features['medgemma']  # (1, 2560) — single embedding per file

    # Save embeddings
    total = patient_store.save_embeddings(TEST_PATIENT, raw_embeddings)
    print(f"  Saved {raw_embeddings.shape[0]} embeddings -> {total} total accumulated")

    # Load and verify
    loaded = patient_store.load_embeddings(TEST_PATIENT)
    assert loaded.shape == raw_embeddings.shape, f"Shape mismatch: {loaded.shape} vs {raw_embeddings.shape}"
    assert np.allclose(loaded, raw_embeddings), "Loaded embeddings don't match saved ones!"
    print(f"  Loaded back: {loaded.shape} — matches original")

    # Score via raw ndarray path (cumulative scoring)
    result_cumulative = clf.score_patient(loaded, is_speaker_level=False)
    print(f"  Cumulative score (ndarray path): {result_cumulative['score']:.4f} ({result_cumulative['n_chunks']} recordings)")

    # Verify it matches the dict path
    assert abs(result_cumulative['score'] - result['score']) < 1e-6, \
        f"Score mismatch: dict={result['score']:.6f} vs ndarray={result_cumulative['score']:.6f}"
    print(f"  Dict path vs ndarray path scores match: YES")

    # Simulate a second upload — append same embeddings again
    total2 = patient_store.save_embeddings(TEST_PATIENT, raw_embeddings)
    loaded2 = patient_store.load_embeddings(TEST_PATIENT)
    print(f"  After 2nd upload: {loaded2.shape[0]} total recordings (was {total})")
    assert loaded2.shape[0] == total * 2, f"Expected {total*2} recordings, got {loaded2.shape[0]}"

    result_double = clf.score_patient(loaded2, is_speaker_level=False)
    print(f"  Score after 2x embeddings: {result_double['score']:.4f} ({result_double['n_chunks']} recordings)")

    # Clean up test data
    patient_store.clear_patient_data(TEST_PATIENT)
    # Remove patient JSON too
    p_path = patient_store._get_patient_path(TEST_PATIENT)
    if p_path.exists():
        os.remove(p_path)
    print(f"  Cleaned up test patient data")

except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Step 6: Single-file embedding (std=0 when n=1) ---
print("\n[6/7] Single-file embedding note...")
try:
    # With 1-embedding-per-file, a single upload yields exactly 1 row, so
    # std is always zero for a single file.  This is expected and matches
    # the holdout evaluation behaviour.
    result_single = clf.score_patient(raw_embeddings, is_speaker_level=False)
    print(f"  Single file (1 embedding, std=0): score={result_single['score']:.4f}")
    print(f"  This matches the holdout evaluation granularity (1 embedding per file).")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Step 7: Date extraction from filenames ---
print("\n[7/7] Date extraction from filenames...")
try:
    from app import extract_date_from_filename

    test_cases = [
        # Named month patterns
        ("CarolBurnett_15Jan2025.wav",    "2025-01-15"),
        ("file_3February2024.mp3",        "2024-02-03"),
        ("recording_1Febraury2023.wav",   "2023-02-01"),  # typo: Febraury
        ("subject_22Decemeber2024.wav",   "2024-12-22"),  # typo: Decemeber
        ("test_7Apr2025_extra.wav",       "2025-04-07"),
        # Numeric DD_MM_YYYY patterns (underscore-separated at end of stem)
        ("patient_speech_part_0001_10_11_2024.wav", "2024-11-10"),
        ("patient_speech_part_0005_10_12_2024.wav", "2024-12-10"),
        ("patient_speech_part_0009_10_02_2025.wav", "2025-02-10"),
        # No date
        ("DonaldSterling_5.wav",           None),
    ]

    all_pass = True
    for filename, expected in test_cases:
        result_dt = extract_date_from_filename(filename)
        actual = result_dt.strftime("%Y-%m-%d") if result_dt else None
        status = "OK" if actual == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {status}: '{filename}' -> {actual} (expected {expected})")

    if all_pass:
        print("  All date extraction tests passed.")
    else:
        print("  WARNING: Some date extraction tests failed.")

except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
score = result['score']
if score < 0.3:
    label = "LOW RISK"
elif score < 0.6:
    label = "ELEVATED RISK"
else:
    label = "HIGH RISK"
print(f"RESULT: {score:.4f} ({label})")
print("=" * 60)

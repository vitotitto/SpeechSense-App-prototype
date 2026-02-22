"""
SpeechSense Monitor - Transcript Service
========================================
Simplified wrapper around the Pyannote API to transcribe a single audio file.
"""

import os
import time
from pathlib import Path
from typing import Optional

from core.pyannote_client import PyannoteApiClient, build_transcript_segments

def transcribe_audio(filepath: str, timeout_s: int = 300) -> str:
    """
    Uploads an audio file to Pyannote API, runs diarization+transcription,
    polls for completion, and returns the full transcript as a single string.
    
    Raises RuntimeError if the API key is missing, the file is invalid, 
    or the job fails/times out.
    """
    api_key = os.environ.get("PYANNOTE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PYANNOTE_API_KEY environment variable is not set. "
            "Please configure it in the .env file."
        )
        
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")
        
    client = PyannoteApiClient(api_key=api_key)
    
    print(f"Uploading {path.name} to Pyannote API...")
    try:
        # 1. Create upload URL and upload file
        # The media_url format doesn't strictly matter for the Pyannote backend processing,
        # it just needs to be a unique 'media://' URI for the API
        media_uri = f"media://speechsense_app/{path.name}_{int(time.time())}.wav"
        
        upload_url = client.create_upload_url(media_uri)
        client.upload_file(upload_url, path)
        
        # 2. Submit job
        print("Submitting diarization & transcription job...")
        payload = {
            "url": media_uri,
            "model": "precision-2",
            "transcription": True,
            "transcriptionConfig": {"model": "parakeet-tdt-0.6b-v3"},
            "exclusive": True
        }
        job_id = client.submit_diarization(payload)
        print(f"Job submitted. ID: {job_id}")
        
        # 3. Poll for completion
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_s:
                raise RuntimeError(f"Pyannote processing timed out after {timeout_s}s")
                
            job = client.get_job(job_id)
            status = str(job.get("status", "")).lower()
            
            if status == "succeeded":
                print("Job succeeded. Extracting transcript...")
                output = job.get("output", {})
                
                # We can reuse the existing helper from the pipeline
                segments = build_transcript_segments(output, prefer_exclusive=True)
                
                # Combine all text segments into a single string
                full_text = " ".join([seg.get("text", "").strip() for seg in segments])
                # Clean up multiple spaces
                full_text = " ".join(full_text.split())
                
                if not full_text:
                    raise RuntimeError(
                        "Job succeeded but produced no transcript text. "
                        "Check the audio quality."
                    )
                    
                return full_text, segments
                
            elif status in ["failed", "canceled"]:
                error_msg = job.get("error", "Unknown error")
                raise RuntimeError(f"Pyannote job {status}: {error_msg}")
                
            # Still processing...
            time.sleep(5)
            
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")

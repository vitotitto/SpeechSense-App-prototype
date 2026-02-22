"""
Pyannote API Client
===================
Lightweight client for the pyannote.ai diarisation + transcription API.
Extracted from the research pipeline's run_pyannote_api.py — contains only
the classes and helpers needed by the SpeechSense Monitor application.
"""

from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class PyannoteApiClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.pyannote.ai/v1",
        timeout_s: int = 120,
        max_retries: int = 5,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = requests.Session()

    def _request_json(
        self,
        method: str,
        url_or_path: str,
        *,
        expected: Tuple[int, ...],
        json_body: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        headers: Optional[Dict[str, str]] = None,
        data: Any = None,
        parse_json: bool = True,
    ) -> Any:
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            url = url_or_path
        else:
            url = f"{self.base_url}{url_or_path}"

        req_headers: Dict[str, str] = {}
        if auth:
            req_headers["Authorization"] = f"Bearer {self.api_key}"
        if headers:
            req_headers.update(headers)

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    headers=req_headers,
                    json=json_body,
                    data=data,
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"{method} {url} request failed: {exc}") from exc
                time.sleep(min(2 ** attempt, 30))
                continue

            if response.status_code in expected:
                if not parse_json:
                    return None
                if not response.content:
                    return {}
                try:
                    return response.json()
                except ValueError as exc:
                    raise RuntimeError(
                        f"{method} {url} returned non-JSON response ({response.status_code})"
                    ) from exc

            retryable = response.status_code in {429, 500, 502, 503, 504}
            if retryable and attempt < self.max_retries - 1:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = int(retry_after)
                else:
                    delay = min(2 ** attempt, 60)
                time.sleep(delay)
                continue

            body_preview = response.text[:1500]
            raise RuntimeError(
                f"{method} {url} failed ({response.status_code}): {body_preview}"
            )

        raise RuntimeError(f"{method} {url} exhausted retries")

    def create_upload_url(self, media_url: str) -> str:
        payload = {"url": media_url}
        resp = self._request_json(
            "POST",
            "/media/input",
            expected=(201,),
            json_body=payload,
        )
        upload_url = resp.get("url")
        if not upload_url:
            raise RuntimeError(f"Upload URL missing in response: {resp}")
        return str(upload_url)

    def upload_file(self, upload_url: str, file_path: Path) -> None:
        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = "application/octet-stream"
        with open(file_path, "rb") as f:
            self._request_json(
                "PUT",
                upload_url,
                expected=(200, 201, 204),
                auth=False,
                headers={"Content-Type": content_type},
                data=f,
                parse_json=False,
            )

    def submit_diarization(self, payload: Dict[str, Any]) -> str:
        resp = self._request_json(
            "POST",
            "/diarize",
            expected=(200,),
            json_body=payload,
        )
        job_id = resp.get("jobId")
        if not job_id:
            raise RuntimeError(f"jobId missing in response: {resp}")
        return str(job_id)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        resp = self._request_json(
            "GET",
            f"/jobs/{job_id}",
            expected=(200,),
        )
        if not isinstance(resp, dict):
            raise RuntimeError(f"Unexpected job response for {job_id}: {resp}")
        return resp


def select_primary_diarization(
    output: Dict[str, Any],
    prefer_exclusive: bool,
) -> List[Dict[str, Any]]:
    if prefer_exclusive:
        exclusive = output.get("exclusiveDiarization")
        if isinstance(exclusive, list) and exclusive:
            return exclusive
    diarization = output.get("diarization")
    if isinstance(diarization, list):
        return diarization
    return []


def build_transcript_segments(
    output: Dict[str, Any],
    prefer_exclusive: bool,
) -> List[Dict[str, Any]]:
    primary_diar = select_primary_diarization(output, prefer_exclusive)
    turns = output.get("turnLevelTranscription")
    diar_conf_by_key: Dict[Tuple[float, float, str], Dict[str, Any]] = {}

    for seg in primary_diar:
        conf = seg.get("confidence")
        if not isinstance(conf, dict):
            continue
        s = round(to_float(seg.get("start"), -1), 3)
        e = round(to_float(seg.get("end"), -1), 3)
        spk = str(seg.get("speaker", ""))
        diar_conf_by_key[(s, e, spk)] = conf

    if isinstance(turns, list) and turns:
        raw_segments = turns
    else:
        raw_segments = [
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": "",
                "speaker": seg.get("speaker"),
            }
            for seg in primary_diar
        ]

    raw_segments.sort(key=lambda x: (to_float(x.get("start")), to_float(x.get("end"))))
    segments: List[Dict[str, Any]] = []
    for idx, seg in enumerate(raw_segments, start=1):
        start = round(to_float(seg.get("start")), 3)
        end = round(to_float(seg.get("end")), 3)
        if end <= start:
            continue
        speaker = str(seg.get("speaker", ""))
        row: Dict[str, Any] = {
            "id": idx,
            "start": start,
            "end": end,
            "text": str(seg.get("text", "")).strip(),
            "speaker": speaker,
        }
        conf_map = diar_conf_by_key.get((start, end, speaker))
        if conf_map:
            row["speaker_confidence"] = conf_map
        segments.append(row)
    return segments

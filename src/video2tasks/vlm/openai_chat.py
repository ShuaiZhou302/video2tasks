"""OpenAI-compatible chat completions backend for vision requests."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests

from .base import VLMBackend


def _encode_png_data_url(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    s = t.find("{")
    e = t.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(t[s : e + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def _normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text", "")))
        return "\n".join(x for x in chunks if x)
    return str(content)


class OpenAIChatBackend(VLMBackend):
    """OpenAI-compatible backend, e.g. AIHubMix."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_sec: float = 120.0,
        max_tokens: int = 2048,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_sec = float(timeout_sec)
        self.max_tokens = int(max_tokens)
        self.headers = headers or {}

    @property
    def name(self) -> str:
        return "openai_chat"

    def infer(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            print("[OpenAIChat] Missing api_key")
            return {}
        if not self.model:
            print("[OpenAIChat] Missing model")
            return {}

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            data_url = _encode_png_data_url(img)
            if not data_url:
                continue
            content.append(
                {"type": "image_url", "image_url": {"url": data_url}}
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
        }
        req_headers: Dict[str, str] = {
            "Content-Type": "application/json",
            **self.headers,
        }
        if not any(k.lower() == "authorization" for k in req_headers):
            req_headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        try:
            r = requests.post(url, json=payload, headers=req_headers, timeout=self.timeout_sec)
        except requests.RequestException as e:
            print(f"[OpenAIChat] Request failed: {e}")
            return {}

        if r.status_code != 200:
            print(f"[OpenAIChat] HTTP {r.status_code}: {r.text[:500]}")
            return {}

        try:
            data = r.json()
        except json.JSONDecodeError as e:
            print(f"[OpenAIChat] Invalid JSON response: {e}")
            return {}

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return {}

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            return {}

        raw = _normalize_message_content(message.get("content"))
        parsed = _extract_json(raw)
        if isinstance(parsed, dict) and ("transitions" in parsed or "instructions" in parsed):
            return parsed
        return {}

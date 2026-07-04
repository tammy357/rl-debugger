"""The only module that touches the network. OpenAI-compatible chat
completions; llama-server is the primary backend (default port 8080), LM
Studio the fallback (set GEMMA_BASE_URL=http://localhost:1234/v1)."""

import os

import requests

from .errors import AnalyzeRunError

DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_MODEL = "local-gemma"


class GemmaClient:
    def __init__(self, base_url=None, model=None, timeout=120.0,
                 temperature=0.2, max_tokens=1200):
        self.base_url = (base_url or os.environ.get("GEMMA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or os.environ.get("GEMMA_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_usage = None

    def chat(self, messages, temperature=None):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            resp = requests.post(f"{self.base_url}/chat/completions",
                                 json=body, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise AnalyzeRunError("timeout", f"inference exceeded {self.timeout}s")
        except requests.exceptions.RequestException as exc:
            raise AnalyzeRunError("backend", f"cannot reach Gemma server at {self.base_url}: {exc}")
        if resp.status_code >= 400:
            raise AnalyzeRunError("backend", f"Gemma server returned HTTP {resp.status_code}")
        data = resp.json()
        self.last_usage = data.get("usage")
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        if not content and message.get("reasoning_content"):
            # Live failure seen 2026-07-04: Gemma 4's chat template enables
            # thinking by default; the whole max_tokens budget goes to
            # reasoning_content and content stays empty. Fail loud + actionable
            # instead of letting extract_json report a misleading bad_json.
            raise AnalyzeRunError(
                "backend",
                "model spent its entire token budget on thinking and returned no "
                "answer — launch llama-server with --reasoning-budget 0 to "
                "disable thinking mode",
            )
        return content

import pytest
import requests

from stream2_gemma_inference.client import GemmaClient
from stream2_gemma_inference.errors import AnalyzeRunError
from stream2_gemma_inference.mock_client import MockClient

MSGS = [{"role": "user", "content": "hi"}]


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


def _payload(content="hello", usage=None):
    return {"choices": [{"message": {"content": content}}], "usage": usage or {"prompt_tokens": 7}}


def test_chat_returns_content_and_records_usage(monkeypatch):
    client = GemmaClient(base_url="http://x/v1", model="m")
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(url=url, body=json, timeout=timeout)
        return FakeResp(payload=_payload("the answer"))

    monkeypatch.setattr(requests, "post", fake_post)
    assert client.chat(MSGS) == "the answer"
    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["body"]["temperature"] == 0.2 and captured["body"]["max_tokens"] == 1200
    assert captured["timeout"] == 120.0
    assert client.last_usage == {"prompt_tokens": 7}


def test_per_call_temperature_override(monkeypatch):
    client = GemmaClient(base_url="http://x/v1", model="m")
    captured = {}
    monkeypatch.setattr(
        requests, "post",
        lambda url, json=None, timeout=None: captured.update(body=json) or FakeResp(payload=_payload()),
    )
    client.chat(MSGS, temperature=0)
    assert captured["body"]["temperature"] == 0


def test_timeout_maps_to_kind_timeout(monkeypatch):
    client = GemmaClient(base_url="http://x/v1", model="m")
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.Timeout()))
    with pytest.raises(AnalyzeRunError) as exc:
        client.chat(MSGS)
    assert exc.value.kind == "timeout"


def test_connection_error_maps_to_kind_backend(monkeypatch):
    client = GemmaClient(base_url="http://x/v1", model="m")
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.ConnectionError()))
    with pytest.raises(AnalyzeRunError) as exc:
        client.chat(MSGS)
    assert exc.value.kind == "backend"


def test_http_error_maps_to_kind_backend(monkeypatch):
    client = GemmaClient(base_url="http://x/v1", model="m")
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp(status=500))
    with pytest.raises(AnalyzeRunError) as exc:
        client.chat(MSGS)
    assert exc.value.kind == "backend"


def test_thinking_only_response_raises_actionable_backend_error(monkeypatch):
    # Live-failure regression (2026-07-04): Gemma 4 thinking is ON by default at
    # the template level; ALL tokens go to reasoning_content, content stays
    # empty, and analyze_run dies with an unhelpful bad_json two calls later.
    client = GemmaClient(base_url="http://x/v1", model="m")
    payload = {"choices": [{"message": {"content": "", "reasoning_content": "hmm " * 500}}],
               "usage": {"completion_tokens": 1200}}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp(payload=payload))
    with pytest.raises(AnalyzeRunError) as exc:
        client.chat(MSGS)
    assert exc.value.kind == "backend"
    assert "--reasoning-budget 0" in str(exc.value)  # actionable message


def test_env_defaults(monkeypatch):
    monkeypatch.setenv("GEMMA_BASE_URL", "http://env-host:9999/v1")
    monkeypatch.setenv("GEMMA_MODEL", "env-model")
    client = GemmaClient()
    assert client.base_url == "http://env-host:9999/v1"
    assert client.model == "env-model"


def test_mock_client_replays_and_records():
    mock = MockClient(["one", "two"])
    assert mock.chat(MSGS) == "one"
    assert mock.chat(MSGS, temperature=0) == "two"
    assert mock.calls[1]["temperature"] == 0
    assert mock.calls[0]["messages"] == MSGS


def test_max_tokens_env_override(monkeypatch):
    # Deep-analysis mode needs a bigger completion budget without code edits.
    monkeypatch.setenv("GEMMA_MAX_TOKENS", "2048")
    client = GemmaClient(base_url="http://x/v1", model="m")
    captured = {}
    monkeypatch.setattr(
        requests, "post",
        lambda url, json=None, timeout=None: captured.update(body=json) or FakeResp(payload=_payload()),
    )
    client.chat(MSGS)
    assert captured["body"]["max_tokens"] == 2048

"""Canned-response stand-in for GemmaClient — lets everything below the HTTP
line run in tests without a model."""


class MockClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.last_usage = None

    def chat(self, messages, temperature=None):
        self.calls.append({"messages": messages, "temperature": temperature})
        return self._responses.pop(0)

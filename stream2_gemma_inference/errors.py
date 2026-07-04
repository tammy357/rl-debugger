class AnalyzeRunError(Exception):
    """Raised when analyze_run cannot produce a valid hypothesis.

    kind: "timeout"  — inference call exceeded the client timeout
          "backend"  — server unreachable / HTTP error / model not loaded
          "bad_json" — model output failed validation even after one repair round
    Stream 3 can branch on .kind; .raw_response holds the model's last output.
    """

    def __init__(self, kind, message, raw_response=None):
        super().__init__(message)
        self.kind = kind
        self.raw_response = raw_response

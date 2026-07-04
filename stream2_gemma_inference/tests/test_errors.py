from stream2_gemma_inference.errors import AnalyzeRunError


def test_error_carries_kind_and_raw_response():
    err = AnalyzeRunError("bad_json", "no JSON found", raw_response="blah")
    assert err.kind == "bad_json"
    assert err.raw_response == "blah"
    assert "no JSON found" in str(err)


def test_raw_response_optional():
    err = AnalyzeRunError("timeout", "gave up after 120s")
    assert err.kind == "timeout"
    assert err.raw_response is None

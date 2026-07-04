from stream2_gemma_inference import AnalyzeRunError, analyze_run


def test_package_exports_stream2_public_api():
    assert analyze_run is not None
    assert AnalyzeRunError is not None

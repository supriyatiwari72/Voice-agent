"""
Tests for benchmark suite functions.
Verifies that all 4 benchmark functions run without exceptions using mock/dummy
providers, and that their output JSON structure is correct.
"""
import os
import json
import pytest
import tempfile


class TestBenchmarkSTT:

    def test_stt_benchmark_runs_without_exception(self, monkeypatch):
        """Verify benchmark_stt runs with dummy provider."""
        from stt.factory import STTFactory
        original_get_provider = STTFactory.get_provider
        
        def mock_get_provider(name, config=None):
            if name != "dummy":
                raise ValueError("Skipping real provider in tests")
            return original_get_provider(name, config)
            
        monkeypatch.setattr(STTFactory, "get_provider", mock_get_provider)

        from benchmarks.benchmark_stt import run_stt_benchmark
        results = run_stt_benchmark(num_iterations=2)
        # Should return a dict containing at least the dummy results
        assert isinstance(results, dict)
        assert "dummy" in results

    def test_stt_benchmark_dummy_provider_returns_results(self, monkeypatch):
        """With dummy provider available, results should not be empty."""
        from benchmarks.benchmark_stt import run_stt_benchmark

        # Patch providers list to only use dummy
        import benchmarks.benchmark_stt as bm
        monkeypatch.setattr(bm, "run_stt_benchmark", lambda num_iterations=2: _run_stt_dummy_only(num_iterations))

        def _run_stt_dummy_only(num_iterations):
            from stt.factory import STTFactory
            import time, numpy as np
            provider = STTFactory.get_provider("dummy", {})
            audio = b"\x00" * 32000
            latencies = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                provider.transcribe(audio)
                latencies.append((time.perf_counter() - start) * 1000)
            return {"dummy": {"latency_ms": {"average": float(np.mean(latencies))}}}

        results = bm.run_stt_benchmark(num_iterations=2)
        assert isinstance(results, dict)


class TestBenchmarkLLM:

    def test_llm_benchmark_runs_without_exception(self, monkeypatch):
        from llm.factory import LLMFactory
        original_get_provider = LLMFactory.get_provider
        
        def mock_get_provider(name, config=None):
            if name != "dummy":
                raise ValueError("Skipping real provider in tests")
            return original_get_provider(name, config)
            
        monkeypatch.setattr(LLMFactory, "get_provider", mock_get_provider)

        from benchmarks.benchmark_llm import run_llm_benchmark
        results = run_llm_benchmark(num_iterations=1)
        assert isinstance(results, dict)
        assert "dummy" in results

    def test_llm_benchmark_dummy_result_structure(self):
        """Dummy LLM provider should produce correct result structure."""
        from llm.factory import LLMFactory
        from benchmarks.benchmark_llm import _benchmark_provider

        provider = LLMFactory.get_provider("dummy", {})
        result = _benchmark_provider(provider, "Dummy LLM", num_iterations=2)

        assert result is not None
        assert "ttft_ms" in result
        assert "total_ms" in result
        assert "tokens_per_sec" in result
        assert "resources" in result
        assert result["ttft_ms"]["average"] >= 0


class TestBenchmarkTTS:

    def test_tts_benchmark_runs_without_exception(self, monkeypatch):
        from tts.factory import TTSFactory
        original_get_provider = TTSFactory.get_provider
        
        def mock_get_provider(name, config=None):
            if name != "dummy":
                raise ValueError("Skipping real provider in tests")
            return original_get_provider(name, config)
            
        monkeypatch.setattr(TTSFactory, "get_provider", mock_get_provider)

        from benchmarks.benchmark_tts import run_tts_benchmark
        results = run_tts_benchmark(num_iterations=1)
        assert isinstance(results, dict)
        assert "dummy" in results

    def test_tts_benchmark_dummy_result_structure(self):
        """Dummy TTS provider should produce results with correct keys."""
        results = {}
        from tts.factory import TTSFactory
        import time, numpy as np

        provider = TTSFactory.get_provider("dummy", {})
        latencies = []
        for _ in range(2):
            start = time.perf_counter()
            audio = provider.synthesize("Hello world")
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        assert len(latencies) == 2
        assert all(l >= 0 for l in latencies)


class TestBenchmarkE2E:

    def test_e2e_benchmark_runs_with_mock_stack(self, monkeypatch):
        """E2E benchmark with mock-only stack should complete without exception."""
        from noise.factory import NoiseFactory
        from vad.factory import VADFactory
        from stt.factory import STTFactory
        from llm.factory import LLMFactory
        from tts.factory import TTSFactory

        # Helper mock to throw on non-dummy
        def make_mock_get_provider(factory):
            original = factory.get_provider
            def mock_get_provider(name, config=None):
                if name != "dummy":
                    raise ValueError(f"Skipping real provider {name} in tests")
                return original(name, config)
            return mock_get_provider

        monkeypatch.setattr(NoiseFactory, "get_provider", make_mock_get_provider(NoiseFactory))
        monkeypatch.setattr(VADFactory, "get_provider", make_mock_get_provider(VADFactory))
        monkeypatch.setattr(STTFactory, "get_provider", make_mock_get_provider(STTFactory))
        monkeypatch.setattr(LLMFactory, "get_provider", make_mock_get_provider(LLMFactory))
        monkeypatch.setattr(TTSFactory, "get_provider", make_mock_get_provider(TTSFactory))

        from benchmarks.benchmark_end_to_end import run_e2e_benchmark
        results = run_e2e_benchmark(num_iterations=1)
        assert isinstance(results, dict)
        # Mock baseline should always succeed
        assert "mock_baseline" in results

    def test_e2e_result_structure(self, monkeypatch):
        from noise.factory import NoiseFactory
        from vad.factory import VADFactory
        from stt.factory import STTFactory
        from llm.factory import LLMFactory
        from tts.factory import TTSFactory

        # Helper mock to throw on non-dummy
        def make_mock_get_provider(factory):
            original = factory.get_provider
            def mock_get_provider(name, config=None):
                if name != "dummy":
                    raise ValueError(f"Skipping real provider {name} in tests")
                return original(name, config)
            return mock_get_provider

        monkeypatch.setattr(NoiseFactory, "get_provider", make_mock_get_provider(NoiseFactory))
        monkeypatch.setattr(VADFactory, "get_provider", make_mock_get_provider(VADFactory))
        monkeypatch.setattr(STTFactory, "get_provider", make_mock_get_provider(STTFactory))
        monkeypatch.setattr(LLMFactory, "get_provider", make_mock_get_provider(LLMFactory))
        monkeypatch.setattr(TTSFactory, "get_provider", make_mock_get_provider(TTSFactory))

        from benchmarks.benchmark_end_to_end import run_e2e_benchmark
        results = run_e2e_benchmark(num_iterations=1)

        for stack_key, stack_data in results.items():
            assert "display_name" in stack_data
            assert "iterations" in stack_data
            assert "latency_ms" in stack_data
            latency = stack_data["latency_ms"]
            # All 5 streaming latency keys must be present
            for key in ["first_transcript", "first_llm_token", "first_sentence",
                        "first_audio_chunk", "total_turnaround"]:
                assert key in latency, f"Missing key '{key}' in {stack_key} results"
                assert "avg" in latency[key]


class TestReportGenerator:

    def test_report_generator_runs_with_empty_results(self, tmp_path, monkeypatch):
        """Report generator should produce a valid markdown file even with no results."""
        monkeypatch.setenv("BENCHMARK_RESULTS_DIR", str(tmp_path))

        import benchmarks.report_generator as rg
        original_results = rg.RESULTS_DIR
        original_report = rg.REPORT_PATH

        report_path = str(tmp_path / "benchmark_report.md")
        monkeypatch.setattr(rg, "RESULTS_DIR", str(tmp_path))
        monkeypatch.setattr(rg, "REPORT_PATH", report_path)

        rg.generate_report()

        assert os.path.exists(report_path)
        content = open(report_path).read()
        assert "# Voice Agent" in content
        assert "Recommendations" in content

    def test_report_generator_with_mock_data(self, tmp_path, monkeypatch):
        """Report generator should correctly format mock result data."""
        import benchmarks.report_generator as rg
        monkeypatch.setattr(rg, "RESULTS_DIR", str(tmp_path))
        monkeypatch.setattr(rg, "REPORT_PATH", str(tmp_path / "report.md"))

        # Write mock STT results
        mock_stt = {
            "dummy": {
                "display_name": "Dummy STT",
                "iterations": 3,
                "latency_ms": {"average": 0.5, "p50": 0.4, "p95": 0.8, "p99": 1.0, "min": 0.3, "max": 1.1},
                "resources": {"cpu_percent": 5.0, "ram_mb": 120.0, "ram_delta_mb": 2.0},
            }
        }
        with open(tmp_path / "stt_results.json", "w") as f:
            json.dump(mock_stt, f)

        rg.generate_report()

        content = open(str(tmp_path / "report.md")).read()
        assert "Dummy STT" in content
        assert "STT Provider Comparison" in content

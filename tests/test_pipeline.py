"""
tests/test_pipeline.py
──────────────────────
Smoke tests for the Supernan Hindi Dubbing Pipeline.

Run with:
    pytest tests/ -v

These tests use either real ffmpeg on a synthetic audio file (no internet),
or mocks for the GPU-heavy stages (Whisper, XTTS, GFPGAN).

Environment skip rules
-----------------------
- Tests that require ffmpeg are marked `needs_ffmpeg` and skipped automatically
  when ffmpeg is not found on PATH.
- Tests that require openai-whisper are marked `needs_whisper` and skipped when
  the package is not installed (installed separately via `pip install openai-whisper`).
On Google Colab / CI with full deps these tests all run.
"""

import json
import os
import shutil
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Environment skip markers ─────────────────────────────────────────────────

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not found on PATH (install via brew/apt-get)",
)

try:
    import whisper as _whisper  # noqa: F401
    _has_whisper = True
except ImportError:
    _has_whisper = False

needs_whisper = pytest.mark.skipif(
    not _has_whisper,
    reason="openai-whisper not installed (pip install openai-whisper)",
)

# Ensure root is on path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sine_wav(path: str, duration: float = 5.0, sr: int = 16000) -> str:
    """Write a minimal sine-wave WAV to *path*."""
    import math, array
    n_samples = int(sr * duration)
    samples = array.array("h", [
        int(32767 * math.sin(2 * math.pi * 440 * t / sr))
        for t in range(n_samples)
    ])
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


def _make_test_mp4(path: str, duration: float = 5.0) -> str:
    """
    Create a tiny synthetic MP4 using ffmpeg (color video + sine audio).
    Requires ffmpeg installed.
    """
    import subprocess
    subprocess.run(
        [
            "ffmpeg",
            "-f", "lavfi", "-i", f"color=c=blue:s=320x240:r=25:d={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={duration}",
            "-c:v", "libx264", "-c:a", "aac",
            "-t", str(duration),
            "-y", path,
        ],
        check=True,
        capture_output=True,
    )
    return path


# ── Stage 1: Extract ──────────────────────────────────────────────────────────

class TestExtract:
    @needs_ffmpeg
    def test_creates_clip_and_audio(self, tmp_path):
        input_mp4 = str(tmp_path / "input.mp4")
        _make_test_mp4(input_mp4, duration=35.0)

        from modules.extract import extract

        result = extract(
            input_mp4,
            start=15.0,
            end=30.0,
            tmp_dir=str(tmp_path / "tmp"),
        )

        assert os.path.exists(result["clip"]), "clip.mp4 not created"
        assert os.path.exists(result["audio"]), "clip_audio.wav not created"
        assert os.path.exists(result["silent_clip"]), "silent clip not created"
        assert os.path.getsize(result["clip"]) > 0
        assert os.path.getsize(result["audio"]) > 0

    @needs_ffmpeg
    def test_clip_duration(self, tmp_path):
        """Output clip should be close to (end - start) seconds."""
        import subprocess
        input_mp4 = str(tmp_path / "input.mp4")
        _make_test_mp4(input_mp4, duration=35.0)

        from modules.extract import extract
        result = extract(input_mp4, start=15.0, end=30.0, tmp_dir=str(tmp_path / "tmp"))

        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                result["clip"],
            ],
            capture_output=True, text=True, check=True,
        )
        duration = float(probe.stdout.strip())
        assert 14.0 <= duration <= 16.0, f"Unexpected clip duration: {duration}s"


# ── Stage 2: Transcribe (mocked Whisper) ─────────────────────────────────────

class TestTranscribe:
    def _mock_whisper_result(self):
        return {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 3.5, "text": "Hello, welcome to Supernan."},
                {"start": 3.5, "end": 7.0, "text": "We are looking for talented interns."},
            ],
        }

    @needs_whisper
    def test_output_json_schema(self, tmp_path):
        wav_path = str(tmp_path / "audio.wav")
        _make_sine_wav(wav_path)

        fake_whisper_model = MagicMock()
        fake_whisper_model.transcribe.return_value = self._mock_whisper_result()

        with patch("whisper.load_model", return_value=fake_whisper_model):
            from modules.transcribe import transcribe
            result = transcribe(wav_path, model_size="base", tmp_dir=str(tmp_path / "tmp"))

        assert "segments" in result
        assert "language" in result
        for seg in result["segments"]:
            assert "start" in seg and "end" in seg and "text" in seg

    @needs_whisper
    def test_transcript_json_saved(self, tmp_path):
        wav_path = str(tmp_path / "audio.wav")
        _make_sine_wav(wav_path)

        fake_whisper_model = MagicMock()
        fake_whisper_model.transcribe.return_value = self._mock_whisper_result()

        with patch("whisper.load_model", return_value=fake_whisper_model):
            from modules.transcribe import transcribe
            transcribe(wav_path, model_size="base", tmp_dir=str(tmp_path / "tmp"))

        out = tmp_path / "tmp" / "transcript.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data["segments"]) == 2


# ── Stage 3: Translate (mocked deep-translator) ───────────────────────────────

class TestTranslate:
    TRANSCRIPT = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 3.5, "text": "Hello, welcome to Supernan."},
            {"start": 3.5, "end": 7.0, "text": "We are looking for talented interns."},
        ],
    }

    def test_segment_count_matches(self, tmp_path):
        mock_translator = MagicMock()
        mock_translator.translate.side_effect = [
            "नमस्ते, सुपरनान में आपका स्वागत है।",
            "हम प्रतिभाशाली इंटर्न की तलाश कर रहे हैं।",
        ]

        with patch("deep_translator.GoogleTranslator", return_value=mock_translator):
            from modules.translate import translate
            result = translate(
                self.TRANSCRIPT, tmp_dir=str(tmp_path), use_indictrans=False
            )

        assert len(result["segments"]) == len(self.TRANSCRIPT["segments"])

    def test_hindi_key_present(self, tmp_path):
        mock_translator = MagicMock()
        mock_translator.translate.side_effect = [
            "नमस्ते, सुपरनान में आपका स्वागत है।",
            "हम प्रतिभाशाली इंटर्न की तलाश कर रहे हैं।",
        ]

        with patch("deep_translator.GoogleTranslator", return_value=mock_translator):
            from modules.translate import translate
            result = translate(
                self.TRANSCRIPT, tmp_dir=str(tmp_path), use_indictrans=False
            )

        for seg in result["segments"]:
            assert "hindi" in seg
            assert len(seg["hindi"]) > 0


# ── Stage 4: TTS duration matching ────────────────────────────────────────────

class TestTTSDurationMatch:
    def test_stretch_within_tolerance(self):
        import numpy as np
        from modules.tts import _stretch_audio

        sr = 24000
        # 3-second audio, target 4 seconds
        audio = np.zeros(sr * 3, dtype=np.float32)
        stretched = _stretch_audio(audio, sr, target_duration=4.0)
        actual_duration = len(stretched) / sr
        assert abs(actual_duration - 4.0) < 0.2, f"Duration off: {actual_duration}s"

    def test_no_stretch_within_tolerance(self):
        import numpy as np
        from modules.tts import _stretch_audio

        sr = 24000
        # 3.01s audio, target 3.0s → within 5% tolerance → no stretch
        audio = np.zeros(int(sr * 3.01), dtype=np.float32)
        stretched = _stretch_audio(audio, sr, target_duration=3.0, tolerance=0.05)
        assert len(stretched) == len(audio)


# ── Stage 7: Assemble ─────────────────────────────────────────────────────────

class TestAssemble:
    @needs_ffmpeg
    def test_creates_output_file(self, tmp_path):
        video_path = str(tmp_path / "lipsynced.mp4")
        audio_path = str(tmp_path / "hindi_dubbed.wav")
        output_path = str(tmp_path / "output.mp4")

        _make_test_mp4(video_path, duration=5.0)
        _make_sine_wav(audio_path, duration=5.0)

        from modules.assemble import assemble
        result = assemble(video_path, audio_path, output_path, normalize_audio=False)

        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    @needs_ffmpeg
    def test_output_has_video_and_audio(self, tmp_path):
        import subprocess
        video_path = str(tmp_path / "lipsynced.mp4")
        audio_path = str(tmp_path / "hindi_dubbed.wav")
        output_path = str(tmp_path / "output.mp4")

        _make_test_mp4(video_path, duration=5.0)
        _make_sine_wav(audio_path, duration=5.0)

        from modules.assemble import assemble
        result = assemble(video_path, audio_path, output_path, normalize_audio=False)

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", result],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(probe.stdout)["streams"]
        codec_types = {s["codec_type"] for s in streams}
        assert "video" in codec_types
        assert "audio" in codec_types

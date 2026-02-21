"""
modules/transcribe.py
─────────────────────
Stage 2 – Speech-to-Text with OpenAI Whisper

Strategy for long audio (full-video mode):
  - Split audio on silence using pydub to create manageable chunks
  - Transcribe each chunk independently and re-align timestamps
  - This avoids Whisper's context-window limits and GPU OOM on free Colab

Model selection:
  - "base"    → fast, free Colab T4       (WER ~10%)
  - "small"   → balanced
  - "large-v3"→ best quality, needs ~10 GB VRAM (Colab Pro / A100)
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Silence-based chunking (for long audio) ──────────────────────────────────

def _split_audio_on_silence(
    audio_path: str,
    silence_thresh_db: int = -40,
    min_silence_ms: int = 500,
    keep_silence_ms: int = 250,
) -> list[tuple[float, "AudioSegment"]]:  # noqa: F821
    """
    Return a list of (offset_seconds, AudioSegment) chunks
    split on detected silent regions.
    """
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    audio = AudioSegment.from_wav(audio_path)
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
        keep_silence=keep_silence_ms,
    )
    if not chunks:
        # No silence detected → treat whole file as one chunk
        return [(0.0, audio)]

    offsets: list[tuple[float, "AudioSegment"]] = []
    cursor_ms = 0
    for chunk in chunks:
        offsets.append((cursor_ms / 1000.0, chunk))
        cursor_ms += len(chunk)
    return offsets


# ── Core transcription ────────────────────────────────────────────────────────

def transcribe(
    audio_path: str,
    model_size: str = "base",
    language: str = "en",
    tmp_dir: str = "tmp",
    batch: bool = False,
) -> dict:
    """
    Transcribe *audio_path* with Whisper.

    Parameters
    ----------
    audio_path  : path to a WAV / MP3 file
    model_size  : whisper model (tiny | base | small | medium | large-v3)
    language    : source language code (None = auto-detect)
    tmp_dir     : directory to write transcript JSON
    batch       : if True, split on silence first (for long audio)

    Returns
    -------
    dict  – {"segments": [{start, end, text}, ...], "language": str}
    """
    import whisper

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(tmp_dir, "transcript.json")

    logger.info(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    if batch:
        logger.info("Batch mode: splitting audio on silence")
        from pydub import AudioSegment
        import tempfile

        chunks = _split_audio_on_silence(audio_path)
        all_segments: list[dict] = []

        for offset_sec, chunk in chunks:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                chunk.export(tmp_f.name, format="wav")
                result = model.transcribe(
                    tmp_f.name,
                    language=language,
                    task="transcribe",
                    verbose=False,
                )
            # Re-align timestamps
            for seg in result["segments"]:
                all_segments.append(
                    {
                        "start": round(seg["start"] + offset_sec, 3),
                        "end": round(seg["end"] + offset_sec, 3),
                        "text": seg["text"].strip(),
                    }
                )
        detected_lang = language or "en"

    else:
        logger.info(f"Transcribing {audio_path}")
        result = model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            verbose=False,
        )
        all_segments = [
            {
                "start": round(s["start"], 3),
                "end": round(s["end"], 3),
                "text": s["text"].strip(),
            }
            for s in result["segments"]
        ]
        detected_lang = result.get("language", language)

    output = {"language": detected_lang, "segments": all_segments}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Transcription complete: {len(all_segments)} segments → {out_path}"
    )
    return output


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    audio = sys.argv[1] if len(sys.argv) > 1 else "tmp/clip_audio.wav"
    result = transcribe(audio, model_size="base")
    for seg in result["segments"]:
        print(f"[{seg['start']:.1f}s – {seg['end']:.1f}s] {seg['text']}")

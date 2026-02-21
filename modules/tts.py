"""
modules/tts.py
──────────────
Stage 4 – Hindi Voice Cloning & Synthesis (Coqui XTTS v2)

Key design decisions:
  • Voice is cloned from the original English audio clip so the output voice
    matches the speaker's timbre, not a generic Hindi TTS voice.
  • Each translated segment is synthesised independently, then each chunk is
    time-stretched (pitch-preserving via pyrubberband) to fill exactly the
    original segment duration. This keeps lips in sync without gaps/overlaps.
  • For long videos the same strategy applies: process one segment at a time,
    concat with ffmpeg at the end.

Dependencies:
    pip install TTS pyrubberband
    apt-get install -y rubberband-cli   (Colab)
"""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # XTTS v2 native sample rate


# ── Duration matching ─────────────────────────────────────────────────────────

def _stretch_audio(
    audio: np.ndarray,
    sr: int,
    target_duration: float,
    tolerance: float = 0.05,
) -> np.ndarray:
    """
    Time-stretch *audio* to *target_duration* seconds (pitch-preserving).
    Clamps ratio to [0.5, 2.0] to prevent extreme distortion.
    """
    current_duration = len(audio) / sr
    ratio = current_duration / target_duration  # >1 → speed up, <1 → slow down

    # Clamp: never stretch more than 2× slower or 2× faster
    ratio = max(0.5, min(ratio, 2.0))

    if abs(ratio - 1.0) < tolerance:
        # Already close — just trim/pad to exact length
        return _hard_trim_pad(audio, sr, target_duration)

    try:
        import pyrubberband as pyrb
        stretched = pyrb.time_stretch(audio, sr, rate=ratio)
    except Exception as e:
        logger.warning(f"pyrubberband unavailable ({e}); using naive resampling")
        target_len = int(target_duration * sr)
        stretched = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(audio.dtype)

    # Hard trim/pad to guarantee exact duration
    return _hard_trim_pad(stretched, sr, target_duration)


def _hard_trim_pad(audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
    """Trim or zero-pad audio to exactly target_duration seconds."""
    target_samples = int(round(target_duration * sr))
    if len(audio) >= target_samples:
        return audio[:target_samples]
    # Pad with silence
    pad = np.zeros(target_samples - len(audio), dtype=audio.dtype)
    return np.concatenate([audio, pad])


# ── XTTS synthesis ────────────────────────────────────────────────────────────

def _load_xtts():
    """Load Coqui XTTS v2 model (download on first run ~1.8 GB)."""
    from TTS.api import TTS  # type: ignore
    logger.info("Loading Coqui XTTS v2 model …")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=_has_gpu())
    return tts


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def synthesise(
    hindi_segments: dict,
    speaker_wav: str,
    tmp_dir: str = "tmp",
) -> str:
    """
    Synthesise Hindi speech for each segment, time-stretch to match original
    duration, and concatenate into a single WAV.

    Parameters
    ----------
    hindi_segments : output from translate() – {"segments": [{start,end,text,hindi}]}
    speaker_wav    : path to the reference (original) audio for voice cloning
    tmp_dir        : working directory for intermediate WAVs

    Returns
    -------
    str – path to the final concatenated Hindi WAV (tmp/hindi_dubbed.wav)
    """
    import soundfile as sf

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    tts = _load_xtts()

    segments = hindi_segments["segments"]
    chunk_paths: list[str] = []

    for i, seg in enumerate(segments):
        hindi_text = seg.get("hindi", "").strip()
        if not hindi_text:
            logger.warning(f"Segment {i} has empty Hindi text; inserting silence")
            target_dur = seg["end"] - seg["start"]
            silence = np.zeros(int(target_dur * SAMPLE_RATE), dtype=np.float32)
            chunk_path = os.path.join(tmp_dir, f"seg_{i:04d}.wav")
            sf.write(chunk_path, silence, SAMPLE_RATE)
            chunk_paths.append(chunk_path)
            continue

        target_duration = seg["end"] - seg["start"]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
            raw_path = tmp_f.name

        logger.info(
            f"[{i+1}/{len(segments)}] Synthesising: '{hindi_text[:60]}…'"
            if len(hindi_text) > 60 else
            f"[{i+1}/{len(segments)}] Synthesising: '{hindi_text}'"
        )

        # speed=1.2 – Hindi text is typically 20-30% longer than source;
        # speaking slightly faster keeps the output inside the segment window.
        try:
            tts.tts_to_file(
                text=hindi_text,
                speaker_wav=speaker_wav,
                language="hi",
                file_path=raw_path,
                speed=1.2,
            )
        except TypeError:
            # Older TTS API without speed param
            tts.tts_to_file(
                text=hindi_text,
                speaker_wav=speaker_wav,
                language="hi",
                file_path=raw_path,
            )

        # Load, stretch, save
        audio, sr = sf.read(raw_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono
        audio = audio.astype(np.float32)

        stretched = _stretch_audio(audio, sr, target_duration)

        chunk_path = os.path.join(tmp_dir, f"seg_{i:04d}.wav")
        sf.write(chunk_path, stretched, sr)
        chunk_paths.append(chunk_path)

    # ── Concatenate all segment WAVs ─────────────────────────────────────────
    out_path = os.path.join(tmp_dir, "hindi_dubbed.wav")
    _concat_wavs(chunk_paths, out_path)
    logger.info(f"Hindi dubbed audio → {out_path}")
    return out_path


def _concat_wavs(paths: list[str], out_path: str) -> None:
    """Concatenate a list of WAV files into one using pydub."""
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    for p in paths:
        combined += AudioSegment.from_wav(p)
    combined.export(out_path, format="wav")


if __name__ == "__main__":
    import json
    import sys
    logging.basicConfig(level=logging.INFO)

    seg_path = sys.argv[1] if len(sys.argv) > 1 else "tmp/hindi_segments.json"
    spk_wav = sys.argv[2] if len(sys.argv) > 2 else "tmp/clip_audio.wav"

    with open(seg_path, encoding="utf-8") as f:
        segs = json.load(f)

    out = synthesise(segs, spk_wav)
    print(f"Output: {out}")

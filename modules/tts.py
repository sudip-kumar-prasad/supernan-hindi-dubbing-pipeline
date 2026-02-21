"""
modules/tts.py
──────────────
Stage 4 – Hindi Voice Cloning & Synthesis (Coqui XTTS v2)

Key design decisions:
  • XTTS v2 has a 150-character limit per call — long Hindi text must be split
    into sentences BEFORE synthesis or the model truncates to first sentence only.
  • Each sentence is synthesised independently and WAVs are concatenated.
  • The concatenated audio is then fitted to the target segment duration:
      - If ratio is in [0.85, 1.5]: use pyrubberband atempo stretch
      - If audio is much shorter (ratio < 0.85): pad with silence — better than
        a 2× slow-motion voice
      - Hard trim/pad to exact sample count after any stretching

Dependencies:
    pip install TTS pyrubberband soundfile pydub
    apt-get install -y rubberband-cli   (Colab)
"""

import logging
import os
import re
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # XTTS v2 native sample rate
XTTS_CHAR_LIMIT = 140  # stay safely below the 150-char tokenizer limit


# ── Sentence splitting ────────────────────────────────────────────────────────

def _split_into_sentences(text: str, max_chars: int = XTTS_CHAR_LIMIT) -> list[str]:
    """
    Split *text* into chunks each ≤ *max_chars* characters.
    Splits on sentence-ending punctuation first, then on commas/clauses,
    then hard-splits at max_chars if no good split point is found.
    """
    # Normalize whitespace
    text = " ".join(text.split())

    if len(text) <= max_chars:
        return [text]

    # Try splitting on Hindi/Latin sentence boundaries: ।  .  !  ?
    # Split on commas/clauses for finer granularity if needed
    raw_parts = re.split(r'(?<=[।.!?])\s+|(?<=[,،])\s+', text)

    chunks: list[str] = []
    current = ""
    for part in raw_parts:
        candidate = (current + " " + part).strip() if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If a single part is still too long, hard-split it
            if len(part) > max_chars:
                words = part.split()
                buf = ""
                for w in words:
                    trial = (buf + " " + w).strip() if buf else w
                    if len(trial) <= max_chars:
                        buf = trial
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = w
                current = buf
            else:
                current = part

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


# ── Duration matching ─────────────────────────────────────────────────────────

def _hard_trim_pad(audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
    """Trim or zero-pad audio to exactly target_duration seconds."""
    target_samples = int(round(target_duration * sr))
    if len(audio) >= target_samples:
        return audio[:target_samples]
    pad = np.zeros(target_samples - len(audio), dtype=audio.dtype)
    return np.concatenate([audio, pad])


def _fit_audio(
    audio: np.ndarray,
    sr: int,
    target_duration: float,
) -> np.ndarray:
    """
    Fit *audio* to *target_duration* seconds.

    Strategy:
      • ratio = audio_duration / target_duration
      • 0.85 ≤ ratio ≤ 1.5  → pyrubberband stretch (natural speed change)
      • ratio > 1.5           → speed up, clamped to 2.0x max
      • ratio < 0.85          → speech is shorter than slot; return as-is
                                 (DO NOT PAD — assemble.py will trim the video
                                 to match the audio length instead)
    """
    current_duration = len(audio) / sr
    if target_duration <= 0:
        return audio

    ratio = current_duration / target_duration

    logger.info(f"_fit_audio: audio={current_duration:.2f}s target={target_duration:.2f}s ratio={ratio:.3f}")

    if 0.85 <= ratio <= 1.5:
        # Normal speed adjustment — sounds good
        try:
            import pyrubberband as pyrb
            audio = pyrb.time_stretch(audio, sr, rate=ratio)
            # Hard trim/pad to exact length after stretching
            audio = _hard_trim_pad(audio, sr, target_duration)
        except Exception as e:
            logger.warning(f"pyrubberband failed ({e}); using naive resample")
            target_len = int(target_duration * sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, target_len),
                np.arange(len(audio)),
                audio,
            ).astype(audio.dtype)
    elif ratio > 1.5:
        # Speech is much longer than slot; must speed up
        # Clamp to max 2.0 to avoid chipmunk effect
        clamped = min(ratio, 2.0)
        try:
            import pyrubberband as pyrb
            audio = pyrb.time_stretch(audio, sr, rate=clamped)
        except Exception as e:
            logger.warning(f"pyrubberband failed ({e}); using naive resample")
            target_len = int(target_duration * sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, target_len),
                np.arange(len(audio)),
                audio,
            ).astype(audio.dtype)
    else:
        # ratio < 0.85 → speech is much shorter than the segment
        # Return audio at its NATURAL length — do NOT pad with silence.
        # assemble.py will detect the short audio and trim the video to match.
        logger.info(
            f"Audio ({current_duration:.1f}s) much shorter than target ({target_duration:.1f}s); "
            "returning at natural length — assemble.py will trim video to match"
        )
        return audio  # ← natural length, no padding

    return audio  # for ratio > 1.5 branch (already trimmed above if needed)


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


def _synth_sentences(
    tts,
    sentences: list[str],
    speaker_wav: str,
    tmp_dir: str,
    prefix: str,
) -> np.ndarray:
    """
    Synthesise each sentence independently (respecting XTTS char limit),
    then concatenate the audio arrays.
    """
    import soundfile as sf

    parts: list[np.ndarray] = []

    for j, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tmp_dir) as f:
            sent_path = f.name

        logger.info(f"  sentence {j+1}/{len(sentences)}: '{sentence[:80]}'")

        try:
            tts.tts_to_file(
                text=sentence,
                speaker_wav=speaker_wav,
                language="hi",
                file_path=sent_path,
                speed=1.15,   # slight speed-up to help fit long translations
            )
        except TypeError:
            # older TTS API without speed param
            tts.tts_to_file(
                text=sentence,
                speaker_wav=speaker_wav,
                language="hi",
                file_path=sent_path,
            )

        audio, sr = sf.read(sent_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        parts.append(audio.astype(np.float32))
        os.unlink(sent_path)

    if not parts:
        return np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1s silence fallback

    return np.concatenate(parts)


# ── Public API ────────────────────────────────────────────────────────────────

def synthesise(
    hindi_segments: dict,
    speaker_wav: str,
    tmp_dir: str = "tmp",
) -> str:
    """
    Synthesise Hindi speech for each segment and concatenate into a single WAV.

    For each segment:
      1. Split Hindi text into ≤140-char sentences (XTTS limit).
      2. Synthesise each sentence separately.
      3. Concatenate sentence WAVs.
      4. Fit to target duration (stretch if close, pad if too short).
      5. Concatenate all segment WAVs into final dubbed track.

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
        target_duration = seg["end"] - seg["start"]

        if not hindi_text:
            logger.warning(f"Segment {i} has empty Hindi text; inserting silence")
            silence = np.zeros(int(target_duration * SAMPLE_RATE), dtype=np.float32)
            chunk_path = os.path.join(tmp_dir, f"seg_{i:04d}.wav")
            sf.write(chunk_path, silence, SAMPLE_RATE)
            chunk_paths.append(chunk_path)
            continue

        sentences = _split_into_sentences(hindi_text)
        logger.info(
            f"[{i+1}/{len(segments)}] Synthesising {len(sentences)} sentence(s) "
            f"(target: {target_duration:.1f}s): '{hindi_text[:60]}…'"
            if len(hindi_text) > 60 else
            f"[{i+1}/{len(segments)}] Synthesising '{hindi_text}'"
        )

        # Synthesise all sentences and concatenate
        raw_audio = _synth_sentences(tts, sentences, speaker_wav, tmp_dir, f"seg{i}")
        raw_duration = len(raw_audio) / SAMPLE_RATE
        logger.info(f"  raw synthesis: {raw_duration:.2f}s → fitting to {target_duration:.2f}s")

        # Fit to target duration
        fitted = _fit_audio(raw_audio, SAMPLE_RATE, target_duration)

        chunk_path = os.path.join(tmp_dir, f"seg_{i:04d}.wav")
        sf.write(chunk_path, fitted, SAMPLE_RATE)
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

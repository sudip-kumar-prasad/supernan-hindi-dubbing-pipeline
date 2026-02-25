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
    pip install TTS pyrubberband soundfile pydub moviepy
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


# ── Audio utilities ───────────────────────────────────────────────────────────

def _create_silence(duration: float, sr: int) -> np.ndarray:
    """Create silence of exactly target_duration seconds."""
    if duration <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.zeros(int(round(duration * sr)), dtype=np.float32)


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

    # moviepy expects stereo tracks, stack mono into stereo
    combined = np.concatenate(parts)
    return np.column_stack((combined, combined))


# ── Public API ────────────────────────────────────────────────────────────────

def synthesise(
    hindi_segments: dict,
    speaker_wav: str,
    tmp_dir: str = "tmp",
) -> str:
    """
    Synthesise Hindi speech for each segment and composite into a single WAV using exact timestamps.

    For each segment:
      1. Split Hindi text into ≤140-char sentences (XTTS limit).
      2. Synthesise each sentence separately.
      3. Composite sentences into a moviepy AudioArrayClip.
      4. Shift clip to the segment's exact start_time using CompositeAudioClip.
      5. Output the single timeline audio.

    Parameters
    ----------
    hindi_segments : output from translate() – {"segments": [{start,end,text,hindi}]}
    speaker_wav    : path to the reference (original) audio for voice cloning
    tmp_dir        : working directory for intermediate WAVs

    Returns
    -------
    str – path to the final concatenated Hindi WAV (tmp/hindi_dubbed.wav)
    """
    try:
        from moviepy import AudioArrayClip, CompositeAudioClip
    except ImportError:
        from moviepy.editor import AudioArrayClip, CompositeAudioClip

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    tts = _load_xtts()

    segments = hindi_segments["segments"]
    audio_clips = []

    for i, seg in enumerate(segments):
        start_time = seg.get("start", 0.0)
        hindi_text = seg.get("hindi", "").strip()
            
        if not hindi_text:
            logger.warning(f"Segment {i} has empty Hindi text; skipping synthesis")
            continue

        sentences = _split_into_sentences(hindi_text)
        logger.info(f"[{i+1}/{len(segments)}] Synthesising '{hindi_text[:60]}…'")

        # 2. Synthesise all sentences and concatenate
        raw_audio_stereo = _synth_sentences(tts, sentences, speaker_wav, tmp_dir, f"seg{i}")
        raw_duration = len(raw_audio_stereo) / SAMPLE_RATE
        logger.info(f"  → Natural speech duration: {raw_duration:.2f}s")

        # 3. Create AudioArrayClip strictly anchored to its absolute timestamp
        clip = AudioArrayClip(raw_audio_stereo, fps=SAMPLE_RATE)
        clip = clip.with_start(start_time) if hasattr(clip, "with_start") else clip.set_start(start_time)
        audio_clips.append(clip)

    # ── Composite all segment clips into one final WAV timeline ────────────────
    if not audio_clips:
        logger.warning("No audio segments synthesized. Creating silent audio.")
        silence = _create_silence(1.0, SAMPLE_RATE)
        silence_stereo = np.column_stack((silence, silence))
        clip = AudioArrayClip(silence_stereo, fps=SAMPLE_RATE)
        clip = clip.with_start(0.0) if hasattr(clip, "with_start") else clip.set_start(0.0)
        audio_clips.append(clip)
        
    final_audio = CompositeAudioClip(audio_clips)
    out_path = os.path.join(tmp_dir, "hindi_dubbed.wav")
    final_audio.write_audiofile(out_path, fps=SAMPLE_RATE, logger=None)
    
    logger.info(f"Hindi dubbed composite audio → {out_path}")
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

"""
modules/extract.py
──────────────────
Stage 1 – Video Clipping & Audio Extraction

Uses ffmpeg to:
  1. Trim the source video to the target [start, end] window → tmp/clip.mp4
  2. Extract a mono 16-kHz WAV for Whisper ASR  → tmp/clip_audio.wav
  3. Produce a silent (no-audio) video stream     → tmp/clip_silent.mp4
     (used later by the lip-sync stage)
"""

import logging
import os
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)


def extract(
    input_path: str,
    start: float = 15.0,
    end: float = 30.0,
    tmp_dir: str = "tmp",
) -> dict[str, str]:
    """
    Clip video and extract audio.

    Returns
    -------
    dict with keys:
        clip        – trimmed video with audio (mp4)
        audio       – mono 16-kHz WAV
        silent_clip – video stream only (mp4, no audio)
    """
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    duration = end - start
    clip_path = os.path.join(tmp_dir, "clip.mp4")
    audio_path = os.path.join(tmp_dir, "clip_audio.wav")
    silent_path = os.path.join(tmp_dir, "clip_silent.mp4")

    # ── 1. Trim to clip ──────────────────────────────────────────────────────
    logger.info(f"Trimming {input_path} [{start}s – {end}s] → {clip_path}")
    (
        ffmpeg
        .input(input_path, ss=start, t=duration)
        .output(clip_path, c="copy", avoid_negative_ts="make_zero")
        .overwrite_output()
        .run(quiet=True)
    )

    # ── 2. Extract mono 16-kHz audio for Whisper ─────────────────────────────
    logger.info(f"Extracting audio → {audio_path}")
    (
        ffmpeg
        .input(clip_path)
        .output(audio_path, ac=1, ar=16000, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )

    # ── 3. Video-only (silent) stream for lip-sync ───────────────────────────
    logger.info(f"Creating silent clip → {silent_path}")
    (
        ffmpeg
        .input(clip_path)
        .output(silent_path, an=None, vcodec="copy")
        .overwrite_output()
        .run(quiet=True)
    )

    result = {
        "clip": clip_path,
        "audio": audio_path,
        "silent_clip": silent_path,
    }
    logger.info(f"Extraction complete: {result}")
    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    inp = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    print(extract(inp))

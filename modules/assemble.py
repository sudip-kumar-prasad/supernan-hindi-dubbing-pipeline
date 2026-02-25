"""
modules/assemble.py
───────────────────
Stage 7 – Final Assembly

Merges the video with the Hindi dubbed audio using ffmpeg.
Strategy:
- Pad dubbed audio with silence to video length.
- Normalize loudness.
- NO background audio (prevents Kannada bleed-through).
- NO atempo stretching (prevents slow-motion audio).
"""

import logging
import os
import subprocess
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_duration(path: str) -> float | None:
    """Return media duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", path,
            ],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        for s in streams:
            if "duration" in s:
                return float(s["duration"])
    except Exception as e:
        logger.warning(f"ffprobe failed for {path}: {e}")
    return None


def assemble(
    video_path: str,
    dubbed_audio_path: str,
    bg_audio_path: str | None = None,   # kept for API compat – ignored
    output_path: str = "output.mp4",
    normalize_audio: bool = True,
) -> str:
    """
    Mux video_path (video stream) + dubbed_audio_path (Hindi audio) into output_path.

    Strategy:
    - Pad the dubbed audio with silence to match the video length.
    - Apply loudnorm to normalize loudness.
    - NO background audio mixing (was causing Kannada bleed-through).
    - NO atempo stretching (was causing extreme slow-motion audio).
    """
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    logger.info(f"Assembling final video → {output_path}")

    vid_dur = _get_duration(video_path)
    aud_dur = _get_duration(dubbed_audio_path)

    logger.info(f"Video duration: {vid_dur:.3f}s | Dubbed audio: {aud_dur:.3f}s")

    # Build audio filter: pad with silence to fill video length, then normalize
    if normalize_audio:
        af = f"apad,atrim=0:{vid_dur:.6f},loudnorm=I=-14:TP=-1.5:LRA=11"
    else:
        af = f"apad,atrim=0:{vid_dur:.6f}"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,         # input 0: video
        "-i", dubbed_audio_path,  # input 1: dubbed Hindi audio
        "-filter_complex", f"[1:a:0]{af}[aout]",
        "-map", "0:v:0",          # take video from input 0
        "-map", "[aout]",         # take processed Hindi audio
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-t", str(vid_dur),
        output_path,
    ]

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Output file: {output_path} ({size_mb:.1f} MB)")
    return output_path


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    vid = sys.argv[1] if len(sys.argv) > 1 else "tmp/enhanced.mp4"
    aud = sys.argv[2] if len(sys.argv) > 2 else "tmp/hindi_dubbed.wav"
    out = sys.argv[3] if len(sys.argv) > 3 else "output.mp4"
    print(assemble(vid, aud, output_path=out))

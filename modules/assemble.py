"""
modules/assemble.py
───────────────────
Stage 7 – Final Assembly

Merges the enhanced (or lip-synced) video with the Hindi dubbed audio using
ffmpeg. Also normalises audio loudness to broadcast standard (–14 LUFS).
"""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def assemble(
    video_path: str,
    audio_path: str,
    output_path: str = "output.mp4",
    normalize_audio: bool = True,
) -> str:
    """
    Mux *video_path* (video stream) + *audio_path* (audio) into *output_path*.

    Parameters
    ----------
    video_path      : enhanced / lip-synced video (may or may not have audio)
    audio_path      : Hindi dubbed WAV
    output_path     : final output file path
    normalize_audio : apply EBU R128 loudness normalization (–14 LUFS)

    Returns
    -------
    str – path to the final output mp4
    """
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)

    logger.info(f"Assembling final video → {output_path}")

    # Get exact video duration so audio is trimmed to match
    import subprocess as _sp, json as _json
    probe = _sp.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
        capture_output=True, text=True,
    )
    try:
        streams = _json.loads(probe.stdout).get("streams", [])
        vid_duration = next(
            (float(s["duration"]) for s in streams if s["codec_type"] == "video"), None
        )
    except Exception:
        vid_duration = None

    audio_filter = "loudnorm=I=-14:TP=-1.5:LRA=11" if normalize_audio else "anull"

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",      # video from first input
        "-map", "1:a:0",      # audio from second input
        "-af", audio_filter,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
    ]

    # Hard-trim both streams to video duration to guarantee A/V sync
    if vid_duration:
        cmd += ["-t", str(vid_duration)]

    cmd += ["-y", output_path]

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
    print(assemble(vid, aud, out))

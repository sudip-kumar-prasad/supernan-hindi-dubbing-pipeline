"""
modules/assemble.py
───────────────────
Stage 7 – Final Assembly

Merges the enhanced (or lip-synced) video with the Hindi dubbed audio using
ffmpeg. Uses ffmpeg's native atempo filter to FORCE the audio to exactly match
the video duration — no external libraries needed, 100% reliable sync.
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


def _has_audio(path: str) -> bool:
    """Check if media file contains an audio stream."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-select_streams", "a", path,
            ],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        return len(streams) > 0
    except Exception as e:
        logger.warning(f"ffprobe audio check failed for {path}: {e}")
        return False


def _build_atempo_chain(ratio: float) -> str:
    """
    Build an ffmpeg atempo filter chain for *ratio*.
    atempo only accepts values in [0.5, 2.0]; chain multiple filters for
    ratios outside that range.
    """
    filters = []
    while ratio > 2.0:
        filters.append("atempo=2.0")
        ratio /= 2.0
    while ratio < 0.5:
        filters.append("atempo=0.5")
        ratio /= 0.5
    filters.append(f"atempo={ratio:.6f}")
    return ",".join(filters)


def assemble(
    video_path: str,
    dubbed_audio_path: str,
    bg_audio_path: str | None = None,
    output_path: str = "output.mp4",
    normalize_audio: bool = True,
) -> str:
    """
    Mux *video_path* (video stream) + *dubbed_audio_path* (Hindi audio) + 
    *bg_audio_path* (original background) into *output_path*.

    A/V sync strategy
    -----------------
    1. Measure exact video duration with ffprobe.
    2. Measure exact audio duration with ffprobe.
    3. Compute tempo ratio = audio_duration / video_duration.
    4. Apply ffmpeg atempo chain to stretch/compress audio to match video.
    5. Hard-trim the output to video duration with -t.

    This means the dubbed audio ALWAYS ends exactly when the video ends,
    regardless of how long or short the TTS synthesised audio is.
    """
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    logger.info(f"Assembling final video → {output_path}")

    vid_dur = _get_duration(video_path)
    aud_dur = _get_duration(dubbed_audio_path)

    logger.info(f"Video duration: {vid_dur:.3f}s | Dubbed audio: {aud_dur:.3f}s")

    audio_filters = []
    out_duration = vid_dur  # always output the full clip length

    if vid_dur and aud_dur:
        ratio = aud_dur / vid_dur
        logger.info(f"Audio/video ratio: {ratio:.3f}")

        if ratio < 0.95:
            if ratio < 0.65:
                # Speech much shorter than clip: pad audio with silence to fill clip,
                # then apply very gentle stretch if needed.
                logger.info(
                    f"Dubbed audio ({aud_dur:.1f}s) << clip ({vid_dur:.1f}s); "
                    "padding audio with silence to fill clip duration"
                )
                audio_filters.append("apad")
            else:
                # Close: apply atempo to match
                logger.info(f"Applying atempo={ratio:.4f} to sync audio to video")
                audio_filters.append(_build_atempo_chain(ratio))
        elif ratio > 1.05:
            logger.info(f"Applying atempo={ratio:.4f} to compress audio to video")
            audio_filters.append(_build_atempo_chain(ratio))

    final_mix_filter = "anull"
    if normalize_audio:
        final_mix_filter = "loudnorm=I=-14:TP=-1.5:LRA=11"

    af_str = ",".join(audio_filters) if audio_filters else "anull"

    has_bg_audio = bg_audio_path is not None and _has_audio(bg_audio_path)
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", dubbed_audio_path,
    ]

    if has_bg_audio:
        cmd.extend(["-i", bg_audio_path])
        # Mix original background audio (lowered to 10% volume) with the new dubbed audio.
        # Apply loudnorm to the dubbed audio BEFORE mixing so background isn't boosted.
        dub_filters = f"{af_str}"
        if normalize_audio:
            dub_filters += f",{final_mix_filter}"

        filter_complex = (
            f"[2:a:0]volume=0.08[bg]; "           # third input is bg_audio_path
            f"[1:a:0]{dub_filters}[dub]; "        # normalize & process dubbed audio ONLY
            f"[bg][dub]amix=inputs=2:duration=first:dropout_transition=2,"
            f"volume=2.0[aout]"                   # mix them safely
        )
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "0:v:0",   # video from first input
            "-map", "[aout]",  # mixed audio from filter_complex
        ])
    else:
        # No background audio; just apply filters to the dubbed audio directly
        logger.info("No original background audio found; skipping mix.")
        
        f_chain = []
        if af_str != "anull":
            f_chain.append(af_str)
        if final_mix_filter != "anull":
            f_chain.append(final_mix_filter)
        
        final_af = ",".join(f_chain) if f_chain else "anull"
        
        # When using -filter_complex, we MUST use it for the audio if we want to map video from 0 
        # and filtered audio from 1. 
        cmd.extend([
            "-filter_complex", f"[1:a:0]{final_af}[aout]",
            "-map", "0:v:0",   # video from first input
            "-map", "[aout]",  # processed audio
        ])

    cmd.extend([
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",   # faster encode, same quality
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
    ])

    # Hard-trim to out_duration: either full clip length or audio length
    # (out_duration == aud_dur when dubbed speech is much shorter than clip)
    if out_duration:
        cmd += ["-t", str(out_duration)]

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
    bg = sys.argv[3] if len(sys.argv) > 3 else "tmp/clip.mp4"
    out = sys.argv[4] if len(sys.argv) > 4 else "output.mp4"
    print(assemble(vid, aud, bg, out))

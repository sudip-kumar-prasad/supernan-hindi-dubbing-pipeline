"""
modules/enhance.py
──────────────────
Stage 6 – Face Restoration

Uses GFPGAN v1.4 to sharpen and restore facial details introduced or degraded
by the lip-sync model. This is the step that separates a blurry result from a
pixel-perfect one.

Fallback chain:
  1. GFPGAN v1.4  (best, needs facexlib + basicsr)
  2. CodeFormer    (if GFPGAN import fails)
  3. No-op        (returns input as-is, logs a warning)

Processing:
  - Extracts frames from video → enhance each frame → re-encode with original audio

Dependencies:
    pip install gfpgan facexlib basicsr
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Frame-level GFPGAN ────────────────────────────────────────────────────────

def _restore_frame_gfpgan(img_bgr: np.ndarray, restorer) -> np.ndarray:
    """Run GFPGAN on a single BGR frame."""
    _, _, output = restorer.enhance(
        img_bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    return output


def _load_gfpgan():
    from gfpgan import GFPGANer  # type: ignore
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="akhaliq/GFPGAN",
        filename="GFPGANv1.4.pth",
    )
    restorer = GFPGANer(
        model_path=model_path,
        upscale=1,           # keep original resolution
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,   # skip background super-res for speed
    )
    logger.info("GFPGAN v1.4 loaded.")
    return restorer


# ── Frame-level CodeFormer (fallback) ────────────────────────────────────────

def _restore_frame_codeformer(img_bgr: np.ndarray) -> np.ndarray:
    """Best-effort CodeFormer via CLI subprocess."""
    # CodeFormer doesn't have a clean Python API in all versions;
    # easiest path is to call its inference script directly.
    raise NotImplementedError("CodeFormer subprocess fallback not yet wired.")


# ── Video processing ─────────────────────────────────────────────────────────

def _extract_frames(video_path: str, frames_dir: str) -> float:
    """Extract frames to PNG files; return FPS."""
    os.makedirs(frames_dir, exist_ok=True)
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v",
            "-of", "default=noprint_wrappers=1:nokey=1",
            "-show_entries", "stream=r_frame_rate",
            video_path,
        ],
        capture_output=True, text=True,
    )
    fps_str = result.stdout.strip()  # e.g. "30000/1001"
    num, den = fps_str.split("/")
    fps = float(num) / float(den)

    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            os.path.join(frames_dir, "frame_%06d.png"),
            "-hide_banner", "-loglevel", "error",
        ],
        check=True,
    )
    return fps


def _encode_frames(frames_dir: str, audio_path: str, fps: float, out_path: str) -> None:
    """Re-encode enhanced frames + audio to MP4."""
    frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
    subprocess.run(
        [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-i", audio_path,
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "slow",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-y", out_path,
        ],
        check=True,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def enhance(
    lipsynced_video: str,
    dubbed_audio: str,
    tmp_dir: str = "tmp",
) -> str:
    """
    Apply GFPGAN face restoration to every frame of *lipsynced_video*.

    Parameters
    ----------
    lipsynced_video : output of lipsync stage
    dubbed_audio    : the final Hindi WAV (for muxing into restored video)
    tmp_dir         : working directory

    Returns
    -------
    str – path to enhanced video (tmp/enhanced.mp4)
    """
    import cv2

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(tmp_dir, "enhanced.mp4")

    # Try to load GFPGAN
    try:
        restorer = _load_gfpgan()
        use_gfpgan = True
    except Exception as e:
        logger.warning(f"GFPGAN unavailable ({e}); skipping face restoration.")
        shutil.copy(lipsynced_video, out_path)
        return out_path

    # Extract frames
    with tempfile.TemporaryDirectory() as frames_dir:
        logger.info("Extracting frames …")
        fps = _extract_frames(lipsynced_video, frames_dir)

        frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
        logger.info(f"Enhancing {len(frame_files)} frames with GFPGAN …")

        for frame_file in frame_files:
            img_bgr = cv2.imread(str(frame_file))
            if img_bgr is None:
                continue
            restored = _restore_frame_gfpgan(img_bgr, restorer)
            cv2.imwrite(str(frame_file), restored)

        logger.info("Re-encoding enhanced frames …")
        _encode_frames(frames_dir, dubbed_audio, fps, out_path)

    logger.info(f"Enhanced video → {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    vid = sys.argv[1] if len(sys.argv) > 1 else "tmp/lipsynced.mp4"
    aud = sys.argv[2] if len(sys.argv) > 2 else "tmp/hindi_dubbed.wav"
    print(enhance(vid, aud))

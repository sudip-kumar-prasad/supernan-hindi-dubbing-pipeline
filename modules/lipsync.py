"""
modules/lipsync.py
──────────────────
Stage 5 – Lip Synchronisation

Preferred: VideoReTalking
    • Better face quality, doesn't blur the face region as aggressively as Wav2Lip
    • GitHub: https://github.com/vinthony/video-retalking
    • Installed via git clone in the Colab notebook cell

Fallback: Wav2Lip
    • Classic, widely available
    • GitHub: https://github.com/Rudrabha/Wav2Lip
    • Installed via git clone in the Colab notebook cell

Both tools are invoked as subprocesses (they don't have clean pip-installable
Python APIs). Model checkpoints are downloaded from HuggingFace Hub on first run.

On CPU-only machines (no CUDA) this stage exits gracefully with a warning and
returns the original silent video so the rest of the pipeline can still complete.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Model download helpers ────────────────────────────────────────────────────

def _ensure_videoretalking(repo_dir: str = "video-retalking") -> str | None:
    """Return path to VideoReTalking run.py, or None if unavailable."""
    run_py = os.path.join(repo_dir, "inference.py")
    if os.path.exists(run_py):
        return run_py

    logger.info("Cloning VideoReTalking …")
    result = subprocess.run(
        [
            "git", "clone",
            "https://github.com/vinthony/video-retalking.git",
            repo_dir,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        logger.warning("Could not clone VideoReTalking: %s", result.stderr.decode())
        return None

    # Download checkpoints via HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        ckpt_dir = os.path.join(repo_dir, "checkpoints")
        Path(ckpt_dir).mkdir(exist_ok=True)
        snapshot_download(
            "vinthony/video-retalking",
            local_dir=ckpt_dir,
            ignore_patterns=["*.git*"],
        )
        logger.info("VideoReTalking checkpoints downloaded.")
    except Exception as e:
        logger.warning(f"Checkpoint download failed: {e}")

    return run_py if os.path.exists(run_py) else None


def _ensure_wav2lip(repo_dir: str = "Wav2Lip") -> str | None:
    """Return path to Wav2Lip inference.py, or None if unavailable."""
    inf_py = os.path.join(repo_dir, "inference.py")
    if os.path.exists(inf_py):
        return inf_py

    logger.info("Cloning Wav2Lip …")
    result = subprocess.run(
        [
            "git", "clone",
            "https://github.com/Rudrabha/Wav2Lip.git",
            repo_dir,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        logger.warning("Could not clone Wav2Lip: %s", result.stderr.decode())
        return None

    # Download wav2lip_gan.pth from HuggingFace
    try:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id="numz/wav2lip_studio",
            filename="Wav2Lip/wav2lip_gan.pth",
            local_dir=repo_dir,
        )
        logger.info(f"Wav2Lip checkpoint: {ckpt_path}")
    except Exception as e:
        logger.warning(f"Wav2Lip checkpoint download failed: {e}")

    return inf_py if os.path.exists(inf_py) else None


# ── VideoReTalking runner ─────────────────────────────────────────────────────

def _run_videoretalking(
    face_video: str,
    audio: str,
    output: str,
    repo_dir: str = "video-retalking",
) -> bool:
    inf_py = _ensure_videoretalking(repo_dir)
    if not inf_py:
        return False

    cmd = [
        "python", inf_py,
        "--face", face_video,
        "--audio", audio,
        "--outfile", output,
    ]
    logger.info("Running VideoReTalking: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


# ── Wav2Lip runner ────────────────────────────────────────────────────────────

def _run_wav2lip(
    face_video: str,
    audio: str,
    output: str,
    repo_dir: str = "Wav2Lip",
) -> bool:
    inf_py = _ensure_wav2lip(repo_dir)
    if not inf_py:
        return False

    checkpoint = os.path.join(repo_dir, "Wav2Lip", "wav2lip_gan.pth")
    if not os.path.exists(checkpoint):
        checkpoint = os.path.join(repo_dir, "wav2lip_gan.pth")

    cmd = [
        "python", inf_py,
        "--checkpoint_path", checkpoint,
        "--face", face_video,
        "--audio", audio,
        "--outfile", output,
        "--resize_factor", "1",
    ]
    logger.info("Running Wav2Lip: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


# ── Public API ────────────────────────────────────────────────────────────────

def lipsync(
    face_video: str,
    dubbed_audio: str,
    tmp_dir: str = "tmp",
    prefer: str = "videoretalking",
) -> str:
    """
    Generate a lip-synced video.

    Parameters
    ----------
    face_video    : video with the speaker's face (silent or with original audio)
    dubbed_audio  : path to the Hindi dubbed WAV
    tmp_dir       : working directory
    prefer        : 'videoretalking' | 'wav2lip'

    Returns
    -------
    str – path to lip-synced video (tmp/lipsynced.mp4)
    """
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(tmp_dir, "lipsynced.mp4")

    if not _has_cuda():
        logger.warning(
            "No CUDA GPU detected. Lip-sync skipped. "
            "Run this stage on Google Colab with a T4/A100 GPU for best results."
        )
        # Return original video so pipeline can still assemble something
        shutil.copy(face_video, out_path)
        return out_path

    success = False

    if prefer == "videoretalking":
        success = _run_videoretalking(face_video, dubbed_audio, out_path)
        if not success:
            logger.info("VideoReTalking failed; trying Wav2Lip fallback …")
            success = _run_wav2lip(face_video, dubbed_audio, out_path)
    else:
        success = _run_wav2lip(face_video, dubbed_audio, out_path)
        if not success:
            logger.info("Wav2Lip failed; trying VideoReTalking fallback …")
            success = _run_videoretalking(face_video, dubbed_audio, out_path)

    if not success:
        logger.error(
            "Both lip-sync backends failed. "
            "Using original video with dubbed audio (no lip-sync)."
        )
        shutil.copy(face_video, out_path)

    return out_path


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    face = sys.argv[1] if len(sys.argv) > 1 else "tmp/clip_silent.mp4"
    audio = sys.argv[2] if len(sys.argv) > 2 else "tmp/hindi_dubbed.wav"
    print(lipsync(face, audio))

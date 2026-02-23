#!/usr/bin/env python3
"""
dub_video.py
────────────
Supernan AI Intern – Hindi Dubbing Pipeline
Orchestrator CLI

Usage
-----
# Full run (GPU required for lipsync & enhance):
python dub_video.py --input input.mp4 --output output.mp4 --start 15 --end 30

# CPU-only / local test (skips GPU stages):
python dub_video.py --input input.mp4 --output output_test.mp4 \
    --start 15 --end 30 --skip-lipsync --skip-enhance

# Batch mode for long audio:
python dub_video.py --input input.mp4 --output output.mp4 --batch

Architecture
------------
Stage 1  extract.py   – ffmpeg clip + audio/video stream separation
Stage 2  transcribe.py – Whisper ASR → JSON segments
Stage 3  translate.py  – IndicTrans2 → Hindi segments
Stage 4  tts.py        – Coqui XTTS v2 voice-clone + duration matching
Stage 5  lipsync.py    – VideoReTalking / Wav2Lip
Stage 6  enhance.py    – GFPGAN face restoration
Stage 7  assemble.py   – ffmpeg final mux + loudness normalization
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import colorlog  # type: ignore

# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(levelname)s]%(reset)s %(cyan)s%(name)s%(reset)s: %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


# ── Stage runner ──────────────────────────────────────────────────────────────

logger = logging.getLogger("dub_video")


class Stage:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        logger.info(f"{'─' * 50}")
        logger.info(f"▶  Stage: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.t0
        if exc_type:
            logger.error(f"✗  {self.name} failed after {elapsed:.1f}s: {exc_val}")
        else:
            logger.info(f"✓  {self.name} completed in {elapsed:.1f}s")
        return False  # re-raise exceptions


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save_checkpoint(tmp_dir: str, stage: str, data: dict) -> None:
    path = os.path.join(tmp_dir, f"checkpoint_{stage}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug(f"Checkpoint saved: {path}")


def _load_checkpoint(tmp_dir: str, stage: str) -> dict | None:
    path = os.path.join(tmp_dir, f"checkpoint_{stage}.json")
    if os.path.exists(path):
        logger.info(f"Resuming from checkpoint: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    input_path: str,
    output_path: str = "output.mp4",
    start: float | None = None,
    end: float | None = None,
    tmp_dir: str = "tmp",
    model_size: str = "base",
    source_lang: str | None = None,
    skip_lipsync: bool = False,
    skip_enhance: bool = False,
    use_indictrans: bool = True,
    batch: bool = False,
    resume: bool = True,
) -> str:
    """
    Run the full dubbing pipeline.

    Parameters
    ----------
    input_path     : Source video file
    output_path    : Where to write the final dubbed video
    start / end    : Clip window in seconds (default 45–60, confirmed speech in source video)
    tmp_dir        : Folder for intermediate files
    model_size     : Whisper model size ('tiny','base','small','medium','large-v3')
    source_lang    : Source language code (None = auto-detect via Whisper)
    skip_lipsync   : Skip lip-sync stage (CPU testing)
    skip_enhance   : Skip face-restoration stage (faster run)
    use_indictrans : Use IndicTrans2 for translation; fall back to deep-translator
    batch          : Enable silence-based audio batching for long audio
    resume         : Resume from last checkpoint if available

    Returns
    -------
    str – path to the output video
    """
    from modules.extract import extract
    from modules.transcribe import transcribe
    from modules.translate import translate
    from modules.tts import synthesise
    from modules.lipsync import lipsync
    from modules.enhance import enhance
    from modules.assemble import assemble

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    total_t0 = time.perf_counter()

    # ── Stage 1: Extract ──────────────────────────────────────────────────────
    with Stage("1 · Extract clip & audio"):
        ckpt = _load_checkpoint(tmp_dir, "extract") if resume else None
        if ckpt:
            paths = ckpt
        else:
            paths = extract(input_path, start=start, end=end, tmp_dir=tmp_dir)
            _save_checkpoint(tmp_dir, "extract", paths)

    clip_path = paths["clip"]
    audio_path = paths["audio"]
    silent_clip = paths["silent_clip"]

    # ── Stage 2: Transcribe ────────────────────────────────────────────────────
    with Stage("2 · Transcribe (Whisper)"):
        ckpt = _load_checkpoint(tmp_dir, "transcribe") if resume else None
        if ckpt:
            transcript = ckpt
        else:
            transcript = transcribe(
                audio_path,
                model_size=model_size,
                language=source_lang,   # None = auto-detect
                tmp_dir=tmp_dir,
                batch=batch,
            )
            _save_checkpoint(tmp_dir, "transcribe", transcript)

    logger.info(
        f"   Detected: {transcript.get('language', '?')} | "
        f"{len(transcript['segments'])} segments"
    )
    if len(transcript['segments']) == 0:
        logger.warning(
            "⚠  No speech segments detected! "
            "Try a different --start/--end window or use --model large-v3 for better accuracy."
        )

    # ── Stage 3: Translate ────────────────────────────────────────────────────
    with Stage("3 · Translate → Hindi"):
        ckpt = _load_checkpoint(tmp_dir, "translate") if resume else None
        if ckpt:
            hindi_segments = ckpt
        else:
            hindi_segments = translate(
                transcript, tmp_dir=tmp_dir, use_indictrans=use_indictrans
            )
            _save_checkpoint(tmp_dir, "translate", hindi_segments)

    # ── Stage 4: TTS / Voice Cloning ──────────────────────────────────────────
    with Stage("4 · Synthesise Hindi audio (XTTS v2)"):
        dubbed_wav_path = os.path.join(tmp_dir, "hindi_dubbed.wav")
        if resume and os.path.exists(dubbed_wav_path):
            logger.info(f"Using existing dubbed audio: {dubbed_wav_path}")
        else:
            dubbed_wav_path = synthesise(
                hindi_segments, speaker_wav=audio_path, tmp_dir=tmp_dir
            )

    # ── Stage 5: Lip Sync ─────────────────────────────────────────────────────
    with Stage("5 · Lip-sync (VideoReTalking / Wav2Lip)"):
        lipsynced_path = os.path.join(tmp_dir, "lipsynced.mp4")
        if skip_lipsync:
            logger.warning("Lip-sync skipped (--skip-lipsync flag).")
            import shutil
            shutil.copy(silent_clip, lipsynced_path)
        elif resume and os.path.exists(lipsynced_path):
            logger.info(f"Using existing lip-synced video: {lipsynced_path}")
        else:
            lipsynced_path = lipsync(silent_clip, dubbed_wav_path, tmp_dir=tmp_dir)

    # ── Stage 6: Face Restoration ─────────────────────────────────────────────
    with Stage("6 · Face restoration (GFPGAN)"):
        enhanced_path = os.path.join(tmp_dir, "enhanced.mp4")
        if skip_enhance:
            logger.warning("Face restoration skipped (--skip-enhance flag).")
            import shutil
            shutil.copy(lipsynced_path, enhanced_path)
        elif resume and os.path.exists(enhanced_path):
            logger.info(f"Using existing enhanced video: {enhanced_path}")
        else:
            enhanced_path = enhance(lipsynced_path, dubbed_wav_path, tmp_dir=tmp_dir)

    # ── Stage 7: Assemble ─────────────────────────────────────────────────────
    with Stage("7 · Assemble final output"):
        output = assemble(
            video_path=enhanced_path, 
            dubbed_audio_path=dubbed_wav_path, 
            bg_audio_path=clip_path,
            output_path=output_path
        )

    total_elapsed = time.perf_counter() - total_t0
    logger.info(f"{'═' * 50}")
    logger.info(f"🎬  Pipeline complete in {total_elapsed:.1f}s")
    logger.info(f"    Output: {output}")
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supernan Hindi Dubbing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", "-i", required=True, help="Source video file (e.g. input.mp4)")
    p.add_argument("--output", "-o", default="output.mp4", help="Output file (default: output.mp4)")
    p.add_argument("--start", "-s", type=float, default=None, help="Clip start (seconds). Default: Full video")
    p.add_argument("--end", "-e", type=float, default=None, help="Clip end (seconds). Default: Full video")
    p.add_argument("--tmp-dir", default="tmp", help="Directory for intermediate files (default: tmp/)")
    p.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base; use large-v3 on Colab GPU for best results)",
    )
    p.add_argument("--source-lang", default=None, help="Source language code (default: auto-detect). E.g. kn, en, mr")
    p.add_argument("--skip-lipsync", action="store_true", help="Skip lip-sync stage (CPU-only testing)")
    p.add_argument("--skip-enhance", action="store_true", help="Skip face restoration stage")
    p.add_argument("--no-indictrans", action="store_true", help="Use deep-translator instead of IndicTrans2")
    p.add_argument("--batch", action="store_true", help="Enable silence-based batching for long audio")
    p.add_argument("--no-resume", action="store_true", help="Ignore checkpoints; rerun all stages")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    run(
        input_path=args.input,
        output_path=args.output,
        start=args.start,
        end=args.end,
        tmp_dir=args.tmp_dir,
        model_size=args.model,
        source_lang=args.source_lang,
        skip_lipsync=args.skip_lipsync,
        skip_enhance=args.skip_enhance,
        use_indictrans=not args.no_indictrans,
        batch=args.batch,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

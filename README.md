# 🎬 Supernan Hindi Dubbing Pipeline

A modular, **₹0-cost** Python pipeline that takes an Indian-language training video and produces a Hindi-dubbed, lip-synced, face-restored clip — designed to run on **Google Colab Free Tier (T4 GPU)** and scale to 500-hour production batches.

> **Repo**: [sudip-kumar-prasad/supernan-hindi-dubbing-pipeline](https://github.com/sudip-kumar-prasad/supernan-hindi-dubbing-pipeline)
> **Source language**: Kannada (auto-detected by Whisper). Optimal clip: **0:45 – 1:00**.

---

## 📋 Table of Contents
- [Pipeline Overview](#pipeline-overview)
- [Setup](#setup)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Cost Analysis](#cost-analysis)
- [Known Limitations](#known-limitations)
- [Scaling to 500 Hours](#scaling-to-500-hours)
- [What I'd Improve](#what-id-improve)

---

## Pipeline Overview

```
Input Video
    │
    ▼
[Stage 1] extract.py      ffmpeg clip + audio/video separation
    │
    ▼
[Stage 2] transcribe.py   Whisper ASR → Kannada/Indic segments (auto-detect, JSON)
    │
    ▼
[Stage 3] translate.py    IndicTrans2-indic-indic-1B (+ deep-translator fallback) → Hindi segments
    │
    ▼
[Stage 4] tts.py          Coqui XTTS v2 voice cloning + pyrubberband duration matching
    │
    ▼
[Stage 5] lipsync.py      VideoReTalking (→ Wav2Lip fallback) lip-sync
    │
    ▼
[Stage 6] enhance.py      GFPGAN v1.4 face restoration
    │
    ▼
[Stage 7] assemble.py     ffmpeg mux + EBU R128 loudness normalization
    │
    ▼
Output Hindi-Dubbed MP4
```

Intermediate files and per-stage checkpoints are saved to `tmp/`. If a run is interrupted, re-running resumes from the last successful stage automatically.

---

## Setup

### Prerequisites
- Python 3.10+
- `ffmpeg` installed and on `$PATH`
- For lip-sync & face-restore: **NVIDIA GPU with CUDA** (GPU stages are skipped gracefully on CPU)

### Local (CPU – good for testing stages 1–4)

```bash
git clone https://github.com/sudip-kumar-prasad/supernan-hindi-dubbing-pipeline.git
cd supernan-hindi-dubbing-pipeline
pip install -r requirements.txt
apt-get install -y rubberband-cli   # macOS: brew install rubberband
```

### Google Colab (Full GPU Pipeline)

Open [colab_notebook.ipynb](colab_notebook.ipynb) and run all cells.
Set **Runtime → T4 GPU** before running.

---

## Usage

### Basic (process 0:45 – 1:00 segment — confirmed Kannada speech)

```bash
python dub_video.py --input input.mp4 --output output.mp4 --start 45 --end 60
```

### CPU-only / local testing (skips lip-sync & face restoration)

```bash
python dub_video.py --input input.mp4 --output output_test.mp4 \
    --start 45 --end 60 --skip-lipsync --skip-enhance
```

### Long video with silence-based batching

```bash
python dub_video.py --input input.mp4 --output output.mp4 \
    --start 0 --end 600 --batch --model large-v3
```

### All options

```
--input      / -i    Source video (required)
--output     / -o    Output path (default: output.mp4)
--start      / -s    Clip start in seconds (default: 45)
--end        / -e    Clip end in seconds (default: 60)
--tmp-dir           Intermediate file directory (default: tmp/)
--model             Whisper model: tiny|base|small|medium|large-v3 (default: base)
--source-lang       Source language code, e.g. kn, en, mr (default: auto-detect)
--skip-lipsync      Skip lip-sync (CPU testing)
--skip-enhance      Skip face restoration
--no-indictrans     Use deep-translator instead of IndicTrans2
--batch             Enable silence-based audio batching for long audio
--no-resume         Ignore checkpoints and rerun all stages
--verbose           Enable debug logging
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ffmpeg-python` | ≥0.2.0 | Video/audio manipulation |
| `pydub` | ≥0.25.1 | Audio concatenation, silence detection |
| `openai-whisper` | ≥20231117 | Speech-to-text ASR |
| `deep-translator` | ≥1.11.4 | Free Google Translate fallback |
| `IndicTrans2` | git | Context-aware Indic→Hindi translation (indic-indic-1B) |
| `TTS` (Coqui) | ≥0.22.0 | XTTS v2 voice cloning |
| `pyrubberband` | ≥0.3.0 | Pitch-preserving time-stretch |
| `gfpgan` | ≥1.3.8 | Face restoration |
| `facexlib` + `basicsr` | ≥0.3.0 / ≥1.4.2 | GFPGAN dependencies |
| `huggingface_hub` | ≥0.19.4 | Model checkpoint download |
| `transformers` | ≥4.36.0 | IndicTrans2 runtime |
| `colorlog` | ≥6.7.0 | Coloured stage logs |
| `soundfile` | any | WAV read/write for TTS |
| `opencv-python-headless` | any | Frame extraction for GFPGAN |

**System**: `ffmpeg`, `rubberband-cli` (via `apt-get` / `brew`)

---

## Cost Analysis

| Compute Tier | Cost / 15-sec clip | Cost / min of video | Monthly (100h/day) |
|---|---|---|---|
| **Colab Free (T4)** | ₹0 | ₹0 | ₹0 (limited GPU hours) |
| **Colab Pro+ (A100)** | ~₹0.5 | ~₹2 | ~₹90,000 |
| **Vast.ai RTX 4090** | ~₹0.1 | ~₹0.4 | ~₹17,000 |
| **AWS g4dn.xlarge** | ~₹0.2 | ~₹0.8 | ~₹35,000 |
| **Runpod A100** | ~₹0.15 | ~₹0.6 | ~₹26,000 |

**Why ₹0 is achievable**: All models (Whisper, XTTS v2, IndicTrans2, VideoReTalking, GFPGAN) are open-source and self-hosted. No paid API calls required. The only cost on free tier is the clip compute time (15-sec clip runs in ~10 min on T4).

---

## Known Limitations

1. **Lip-sync blurriness**: Wav2Lip (fallback) can blur the face region. VideoReTalking is significantly better but requires ~8 GB VRAM.
2. **Long Hindi text**: XTTS v2 has a max token limit per inference; very long segments are silently truncated. Mitigation: sentence-split before synthesis.
3. **Whisper hallucinations**: On noisy audio, Whisper `base` can produce incorrect segments. Use `large-v3` for production.
4. **IndicTrans2 memory**: The 1B IndicTrans2 model needs ~8 GB RAM; falls back gracefully to Google Translate on OOM.
5. **Speaker overlap**: Pipeline assumes a single speaker. Multi-speaker diarization (pyannote.audio) not yet implemented.
6. **No subtitles**: Pipeline only produces dubbed audio; SRT subtitle output is a planned extension.

---

## Scaling to 500 Hours

To process 500 hours of video overnight (assuming ~8 hrs wall time):

```
500 hrs × 3600s/hr ÷ 8 hrs = ~62,500 minutes/hour throughput needed
```

### Architecture

```
                    ┌─────────────────────────────────┐
Video Files (S3)    │    Task Queue (Celery + Redis)   │
      │             └──────────────┬──────────────────┘
      │                            │ tasks
      ▼                            ▼
┌─────────┐    ┌──────────────────────────────────────────┐
│ Ingestion│    │  Worker Pool (Kubernetes / Ray)          │
│ Service │    │  ┌─────────┐ ┌─────────┐ ┌─────────┐    │
│ (upload │    │  │ GPU-0   │ │ GPU-1   │ │ GPU-N   │    │
│  to S3) │───▶│  │(T4/A100)│ │(T4/A100)│ │(T4/A100)│    │
└─────────┘    │  └─────────┘ └─────────┘ └─────────┘    │
               └──────────────────────────────────────────┘
                            │ results
                            ▼
                    ┌───────────────┐
                    │  Output Store │
                    │  (S3 + DB)    │
                    └───────────────┘
```

### Code Changes Required

1. **Batch worker** (`worker.py`): wrap `dub_video.run()` as a Celery task
2. **Queue** (`queue_videos.py`): scan input S3 bucket, enqueue all files
3. **GPU fleet**: spin up 10–20× Runpod A100 instances with the Docker image
4. **Docker image**: `Dockerfile` with all deps pre-installed, models pre-cached
5. **Monitoring**: Flower (Celery dashboard) + Grafana for throughput/ETA
6. **Cost**: ~50 A100s × ₹300/hr × 8h = **₹1,20,000** for 500 hours

---

## What I'd Improve With More Time

- [ ] **Multi-speaker diarization** (pyannote.audio) for videos with multiple speakers
- [ ] **SRT subtitle syncing** for hearing-impaired audience
- [ ] **Whisper large-v3 + VAD** (Voice Activity Detection) for cleaner segments
- [ ] **Better duration matching**: use Montreal Forced Alignment for per-phoneme sync
- [ ] **CodeFormer** face restoration as a secondary pass after GFPGAN
- [ ] **Docker image** with all models pre-baked for zero cold-start on Runpod
- [ ] **Web UI** (Gradio / Streamlit) for non-technical users to upload and download
- [ ] **Unit tests with real audio fixtures** (currently mocked for CI speed)

---

## Project Structure

```
supernan-hindi-dubbing-pipeline/
├── dub_video.py           # Orchestrator CLI (7 stages)
├── modules/
│   ├── __init__.py
│   ├── extract.py         # Stage 1: ffmpeg clip + audio
│   ├── transcribe.py      # Stage 2: Whisper ASR
│   ├── translate.py       # Stage 3: IndicTrans2 / Google Translate
│   ├── tts.py             # Stage 4: Coqui XTTS v2 voice cloning
│   ├── lipsync.py         # Stage 5: VideoReTalking / Wav2Lip
│   ├── enhance.py         # Stage 6: GFPGAN v1.4
│   └── assemble.py        # Stage 7: ffmpeg mux + loudness normalize
├── tests/
│   └── test_pipeline.py   # Pytest smoke tests (mocked GPU stages)
├── colab_notebook.ipynb   # End-to-end Colab runbook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Author

Built for the **Supernan AI Automation Intern Challenge**.
Contact: [ganesh@supernan.app](mailto:ganesh@supernan.app)

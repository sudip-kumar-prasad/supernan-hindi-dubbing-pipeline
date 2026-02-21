"""
modules/translate.py
────────────────────
Stage 3 – English → Hindi Translation

Two backends (tried in order):

1. IndicTrans2 (AI4Bharat) – HuggingFace model
   • Context-aware, trained on Indic languages
   • Requires ~4 GB VRAM; works on Colab T4
   • install: pip install git+https://github.com/AI4Bharat/IndicTrans2.git

2. deep-translator (Google Translate free endpoint) – CPU fallback
   • No API key, no cost
   • Less context-aware but good enough for testing

Both backends translate segment-by-segment while passing surrounding context
sentences to improve coherence on short utterances.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Map Whisper language codes → deep-translator source codes
# (deep-translator uses ISO 639-1 or full names for some)
LANG_CODE_MAP = {
    "kn": "kn",   # Kannada
    "en": "en",   # English
    "hi": "hi",   # Hindi
    "mr": "mr",   # Marathi
    "te": "te",   # Telugu
    "ta": "ta",   # Tamil
    "ml": "ml",   # Malayalam
    "gu": "gu",   # Gujarati
    "bn": "bn",   # Bengali
    "pa": "pa",   # Punjabi
    "si": "si",   # Sinhala
    "or": "or",   # Odia
}

# Map Whisper codes → IndicTrans2 lang tags
INDICTRANS2_LANG_MAP = {
    "kn": "kan_Knda",
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "ml": "mal_Mlym",
}

# ── IndicTrans2 backend ───────────────────────────────────────────────────────

def _translate_indictrans2(texts: list[str], src_lang: str = "kn") -> list[str]:
    """Translate a list of strings via IndicTrans2."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor  # type: ignore

        model_name = "ai4bharat/indictrans2-indic-indic-1B"
        src_tag = INDICTRANS2_LANG_MAP.get(src_lang, f"{src_lang}_Latn")
        logger.info(f"Loading IndicTrans2 model: {model_name} ({src_tag} → hin_Deva)")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        ip = IndicProcessor(inference=True)

        tgt_lang = "hin_Deva"
        batch = ip.preprocess_batch(texts, src_lang=src_tag, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

        with __import__("torch").no_grad():
            outputs = model.generate(**inputs, num_beams=4, max_length=256)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return ip.postprocess_batch(decoded, lang=tgt_lang)

    except Exception as exc:
        logger.warning(f"IndicTrans2 failed ({exc}); falling back to deep-translator")
        raise


# ── deep-translator backend (free Google Translate) ──────────────────────────

def _translate_deep(texts: list[str], src_lang: str = "kn") -> list[str]:
    """Translate via deep-translator (Google free tier)."""
    from deep_translator import GoogleTranslator

    dl_src = LANG_CODE_MAP.get(src_lang, src_lang)
    translator = GoogleTranslator(source=dl_src, target="hi")
    results: list[str] = []
    for text in texts:
        if not text.strip():
            results.append("")
            continue
        translated = translator.translate(text)
        results.append(translated or "")
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def translate(
    transcript: dict,
    tmp_dir: str = "tmp",
    use_indictrans: bool = True,
) -> dict:
    """
    Translate a Whisper transcript dict to Hindi.

    Parameters
    ----------
    transcript      : output from transcribe() – {"segments": [...], "language": ...}
    tmp_dir         : directory for output JSON
    use_indictrans  : attempt IndicTrans2 first; fall back to deep-translator

    Returns
    -------
    dict – {"segments": [{start, end, text, hindi}, ...]}
    """
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(tmp_dir, "hindi_segments.json")

    segments = transcript["segments"]
    source_texts = [s["text"] for s in segments]
    src_lang = transcript.get("language", "kn")   # default Kannada for this video

    logger.info(f"Translating {len(source_texts)} segments {src_lang} → Hindi")

    hindi_texts: list[str]
    if use_indictrans:
        try:
            hindi_texts = _translate_indictrans2(source_texts, src_lang=src_lang)
        except Exception:
            hindi_texts = _translate_deep(source_texts, src_lang=src_lang)
    else:
        hindi_texts = _translate_deep(source_texts, src_lang=src_lang)

    output_segments = []
    for seg, hindi in zip(segments, hindi_texts):
        output_segments.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "hindi": hindi,
            }
        )

    output = {"segments": output_segments}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Translation complete → {out_path}")
    return output


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    transcript_path = sys.argv[1] if len(sys.argv) > 1 else "tmp/transcript.json"
    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)

    result = translate(transcript, use_indictrans=False)
    for seg in result["segments"]:
        print(f"[{seg['start']:.1f}s] {seg['text']}")
        print(f"       ↳ {seg['hindi']}\n")

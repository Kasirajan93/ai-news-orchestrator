from transformers import pipeline
import math

# prefer smaller model for speed, larger model for long docs
MODEL_SHORT = "sshleifer/distilbart-cnn-12-6"
MODEL_LONG = "facebook/bart-large-cnn"

# initialize models lazily (avoid huge start-up cost if not used)
_summarizers = {}

def get_summarizer(model_name):
    if model_name not in _summarizers:
        try:
            _summarizers[model_name] = pipeline("summarization", model=model_name, tokenizer=model_name, device=-1)
        except Exception as e:
            print("Could not load summarizer model", model_name, e)
            _summarizers[model_name] = None
    return _summarizers[model_name]

def is_summary_valid(original, summary, min_overlap=8):
    if not original or not summary:
        return False
    o_words = set(original.lower().split())
    s_words = set(summary.lower().split())
    overlap = o_words & s_words
    return len(overlap) >= min_overlap

def summarize_text(text, max_length=120, min_length=30):
    if not text:
        return ""
    words = len(text.split())
    # very short -> no summarization
    if words < 60:
        return text[:400]

    model_name = MODEL_SHORT if words < 600 else MODEL_LONG
    summarizer = get_summarizer(model_name)
    if summarizer is None:
        return text[:500]  # fallback

    try:
        # chunking for very long texts (simple approach)
        if words > 1200:
            # split into approximate 800-word chunks
            parts = []
            tokens_per_chunk = 800
            words_list = text.split()
            for i in range(0, words, tokens_per_chunk):
                chunk = " ".join(words_list[i:i+tokens_per_chunk])
                out = summarizer(chunk, max_length=max_length, min_length=min_length, truncation=True)
                parts.append(out[0]["summary_text"])
            summary = " ".join(parts)
        else:
            out = summarizer(text, max_length=max_length, min_length=min_length, truncation=True)
            summary = out[0]["summary_text"]

        # anti-hallucination check: ensure some overlap
        if not is_summary_valid(text, summary):
            # fallback: return the first ~400 chars of original cleaned text
            return text[:500]
        return summary
    except Exception as e:
        print("Summarization error:", e)
        return text[:500]

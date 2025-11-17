# summarizer.py — CLOUD SAFE VERSION (NO Transformers)

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def summarize_text(text, max_length=180, min_length=60):
    """
    Lightweight summarizer that works on Streamlit Cloud.
    Uses LexRank extractive summarization (Sumy).
    """

    if not text:
        return ""

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()

        # Get top 4 summary sentences
        sentences = summarizer(parser.document, 4)
        summary = " ".join([str(s) for s in sentences])

        if len(summary.split()) < 15:
            return text[:400]

        return summary
    except:
        # fallback
        return text[:400]

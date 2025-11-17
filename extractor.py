import requests
from bs4 import BeautifulSoup
from newspaper import Article

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

def fallback_extract(url):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Extract paragraphs
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
        # attempt small cleanup
        text = " ".join(text.split())
        return text or None
    except Exception as e:
        raise Exception(f"Fallback extraction failed: {e}")

def extract_thumbnail_from_entry(entry):
    # feedparser entry may contain media_thumbnail or media_content
    try:
        if hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
            return entry.media_thumbnail[0].get("url")
        if hasattr(entry, "media_content") and entry.media_content:
            return entry.media_content[0].get("url")
    except Exception:
        pass
    return None

def extract(url):
    try:
        a = Article(url)
        a.download()
        a.parse()
        text = a.text or ""
        title = a.title or None

        if not text.strip():  # fallback if newspaper3k extracted nothing
            text = fallback_extract(url)

        if not text:
            raise Exception("No text extracted")

        # basic cleaning
        text = " ".join(text.split())
        return {"title": title, "content": text}
    except Exception as e:
        # try fallback once more
        text = fallback_extract(url)
        if text:
            return {"title": None, "content": text}
        raise Exception(f"Extraction failed entirely: {e}")

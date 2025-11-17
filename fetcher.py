import feedparser
from extractor import extract, extract_thumbnail_from_entry
from summarizer import summarize_text
from db import upsert_article, article_exists
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time
import logging
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# -----------------------------
# NEWSAPI INTEGRATION (ADDED)
# -----------------------------
from newsapi import NewsApiClient
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None

analyzer = SentimentIntensityAnalyzer()

# configure logging
logging.basicConfig(filename="logs/fetcher.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------------
# Existing Feature: Tag Extraction
# ----------------------------------
def simple_tags(text, top_n=5):
    try:
        vec = TfidfVectorizer(stop_words='english', max_features=2000)
        X = vec.fit_transform([text])
        scores = np.asarray(X.sum(axis=0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        top = terms[scores.argsort()[-top_n:]][::-1]
        return list(top)
    except Exception:
        return []


# ----------------------------------
# Quality, Sentiment, Category
# ----------------------------------
def compute_quality(text):
    try:
        return min(1000, len(text.split()))
    except Exception:
        return 0

def compute_sentiment(text):
    try:
        s = analyzer.polarity_scores(text)
        return float(s["compound"])
    except Exception:
        return 0.0

def basic_category_from_text(text):
    text = text.lower()
    if "sport" in text or "match" in text or "score" in text:
        return "Sports"
    if "election" in text or "vote" in text or "minister" in text or "president" in text:
        return "Politics"
    if "technology" in text or "tech" in text or "ai " in text:
        return "Technology"
    if "covid" in text or "health" in text or "hospital" in text:
        return "Health"
    if "entertainment" in text or "movie" in text or "film" in text:
        return "Entertainment"
    return "World"


# ===========================================================
#   ✨ NEW FUNCTION: Topic-Based Article Fetch Using NewsAPI
# ===========================================================
def fetch_topic_news(topic, page_size=15):
    """
    Fetch article list based on user topic input.
    Returns normalized list of dicts for DB insertion.
    """
    if not newsapi:
        print("NEWSAPI_KEY missing — skipping topic-based fetch.")
        return []

    try:
        response = newsapi.get_everything(
            q=topic,
            language="en",
            sort_by="relevancy",
            page_size=page_size
        )
    except Exception as e:
        print("NewsAPI error:", e)
        return []

    articles = []
    for a in response.get("articles", []):
        articles.append({
            "url": a.get("url"),
            "title": a.get("title"),
            "published": a.get("publishedAt"),
            "source": a.get("source", {}).get("name"),
            "content": a.get("content") or a.get("description") or "",
            "image": a.get("urlToImage"),
            "category": "NewsAPI",
            "tags": ""
        })
    return articles


# ===========================================================
#   ✨ NEW FUNCTION: Process Topic-Based Articles
# ===========================================================
def process_topic(topic, max_items=15):
    """
    Fetch topic-specific articles using NewsAPI & save to DB.
    Returns count of new articles saved.
    """
    print(f"\n[Topic Fetch] Fetching articles for topic: {topic}")
    articles = fetch_topic_news(topic, page_size=max_items)

    saved = 0
    for a in articles:

        # Skip duplicates
        if article_exists(a["url"]):
            continue

        content = a["content"] or ""
        summary = summarize_text(content)
        sentiment = compute_sentiment(content)
        quality = compute_quality(content)
        tags = simple_tags(content)
        category = basic_category_from_text(content)

        article = {
            "url": a["url"],
            "title": a["title"],
            "published": a["published"],
            "source": a["source"],
            "content": content,
            "summary": summary,
            "tags": tags,
            "image": a["image"],
            "quality": quality,
            "sentiment": sentiment,
            "category": category
        }

        upsert_article(article)
        saved += 1

    print(f"[Topic Fetch] Saved {saved} new topic articles.\n")
    return saved


# ===========================================================
# Existing RSS Feed Processing (unchanged)
# ===========================================================
def process_feed(feed_url, max_items=30):
    logging.info(f"Processing feed: {feed_url}")
    print(f"Processing feed: {feed_url}")
    d = feedparser.parse(feed_url)
    if 'entries' not in d or len(d.entries) == 0:
        logging.info("No entries in feed")
        print("No entries found in feed.")
        return 0

    saved = 0
    for entry in d.entries[:max_items]:
        url = entry.link
        if article_exists(url):
            logging.info(f"Skipping duplicate: {url}")
            continue

        # extraction attempt
        for attempt in range(1, 2 + 1):
            try:
                data = extract(url)
                break
            except Exception as e:
                logging.warning(f"Extract attempt {attempt} failed for {url}: {e}")
                time.sleep(1)
                if attempt == 2:
                    logging.error(f"Extraction failed for {url}")
                    data = None

        if not data or not data.get("content"):
            continue

        summary = summarize_text(data["content"])
        tags = simple_tags(data["content"])
        quality = compute_quality(data["content"])
        sentiment = compute_sentiment(data["content"])
        category = basic_category_from_text(data["content"])
        image = extract_thumbnail_from_entry(entry) or None

        article = {
            "url": url,
            "title": data.get("title") or entry.get("title"),
            "published": entry.get("published", ""),
            "source": urlparse(url).netloc,
            "content": data.get("content"),
            "summary": summary,
            "tags": tags,
            "image": image,
            "quality": quality,
            "sentiment": sentiment,
            "category": category
        }

        try:
            upsert_article(article)
            saved += 1
            logging.info(f"Saved: {article['title']}")
            print(f"Saved: {article['title']}")
        except Exception as e:
            logging.error(f"DB save error for {url}: {e}")

    logging.info(f"Feed done: {feed_url} new_saved={saved}")
    return saved

import sqlite3
from datetime import datetime

DB_PATH = "ainews.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY,
        url TEXT UNIQUE,
        title TEXT,
        published TEXT,
        source TEXT,
        content TEXT,
        summary TEXT,
        tags TEXT,
        image TEXT,
        quality INTEGER,
        sentiment REAL,
        category TEXT,
        fetched_at TEXT
    )
    ''')
    conn.commit()
    conn.close()

def article_exists(url):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM articles WHERE url=?", (url,))
    result = c.fetchone()
    conn.close()
    return result is not None

def upsert_article(article):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    INSERT OR REPLACE INTO articles (url, title, published, source, content, summary, tags, image, quality, sentiment, category, fetched_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        article['url'],
        article.get('title'),
        article.get('published'),
        article.get('source'),
        article.get('content'),
        article.get('summary'),
        ",".join(article.get('tags', [])) if article.get('tags') else None,
        article.get('image'),
        article.get('quality'),
        article.get('sentiment'),
        article.get('category'),
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

def fetch_latest(limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM articles ORDER BY fetched_at DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

import streamlit as st
import pandas as pd
import os
import json
import threading
import time
from pathlib import Path
from collections import Counter, defaultdict
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
from ai_addons import translate_text

def apply_translation(result, summary_lang):
    # Translate main summaries
    result["combined_summary"] = translate_text(result.get("combined_summary",""), summary_lang)
    result["story_reconstruction"] = translate_text(result.get("story_reconstruction",""), summary_lang)

    # Translate timeline events
    for entry in result.get("timeline", []):
        for ev in entry.get("events", []):
            ev["text"] = translate_text(ev.get("text",""), summary_lang)

    # store translation info
    result["_translated"] = True
    result["_translated_lang"] = summary_lang
    st.session_state["_last_result"] = result
    return result


@st.cache_data
def cached_timeline(topic):
    return build_timeline_from_topic(topic)


# Your project imports (these must exist in your project)
# They were referenced in your pasted app; if their APIs differ adapt accordingly.
try:
    from db import fetch_latest, init_db
except Exception:
    # stub fallback if db module missing (keeps app running)
    def init_db():
        return None

    def fetch_latest(n=200):
        return []

try:
    from fetcher import process_feed
except Exception:
    def process_feed(feed, max_items=10):
        # fallback no-op
        return

try:
    from orchestrator import build_timeline_from_topic
except Exception:
    def build_timeline_from_topic(topic):
        # minimal demo result if orchestrator missing
        now = datetime.date.today().isoformat()
        demo = {
            "combined_summary": f"Demo combined summary for '{topic}'",
            "sources": ["https://example.com/news"],
            "articles_count": 3,
            "reliability": 0.78,
            "timeline": [
                {"date": now, "importance": 8, "events": [{"text": "Demo event A", "source": "https://example.com", "title": "Demo A"}], "consensus": ["Demo consensus"], "conflict": []}
            ],
            "mermaid_gantt": "dateFormat  YYYY-MM-DD\naxisFormat  %Y-%m-%d\nsection Demo\nTaskA :done, a1, 2025-01-01, 10d",
            "clusters": {"Cluster 1": ["Demo event A", "Demo event B"]}
        }
        return demo

# ---------------------------
# Optional NLP & viz libraries (graceful)
# ---------------------------
SPACY_AVAILABLE = False
NLP = None
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

TEXTBLOB_AVAILABLE = False
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

WORDCLOUD_AVAILABLE = False
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# ---------------------------
# Source reputation (expandable)
# ---------------------------
SOURCE_REPUTATION = {
    "bbc.co.uk": 0.92,
    "nytimes.com": 0.88,
    "theguardian.com": 0.84,
    "reuters.com": 0.91,
    "hindustantimes.com": 0.70,
    "indiatoday.in": 0.68,
    "sky.com": 0.76,
    "timesofindia.indiatimes.com": 0.66,
}

def get_source_reputation(url: str) -> float:
    if not url:
        return 0.5
    url = url.lower()
    for k, v in SOURCE_REPUTATION.items():
        if k in url:
            return v
    if ".gov" in url or ".edu" in url:
        return 0.85
    return 0.50

# ---------------------------
# Simple sentiment & NER helpers
# ---------------------------
POS_WORDS = set(["good","great","positive","win","winning","support","approve","benefit","improve","clear","success","up"])
NEG_WORDS = set(["bad","worse","negative","loss","problem","attack","criticize","concern","decline","down","fail","controversy"])

def simple_sentiment(text: str) -> float:
    if not text:
        return 0.0
    if TEXTBLOB_AVAILABLE:
        try:
            s = TextBlob(text).sentiment.polarity
            return max(-1.0, min(1.0, s))
        except Exception:
            pass
    t = text.lower()
    words = t.split()
    pos = sum(1 for w in words if w.strip(".,!?:;\"'") in POS_WORDS)
    neg = sum(1 for w in words if w.strip(".,!?:;\"'") in NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total

def extract_ner_counts(events):
    persons = Counter()
    orgs = Counter()
    if SPACY_AVAILABLE and NLP:
        for e in events:
            text = e.get("text") or e.get("title") or ""
            doc = NLP(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    persons[ent.text.strip()] += 1
                elif ent.label_ == "ORG":
                    orgs[ent.text.strip()] += 1
    else:
        # fallback simple heuristic
        fallback_tokens = Counter()
        for e in events:
            text = e.get("text") or e.get("title") or ""
            for tok in text.split():
                if tok.istitle() and len(tok) > 2:
                    fallback_tokens[tok] += 1
        most_common = fallback_tokens.most_common(50)
        for i, (k, c) in enumerate(most_common):
            if i % 3 == 0:
                persons[k] = c
            else:
                orgs[k] = c
    return persons, orgs

# compression & alignment utilities
def compression_score(original: str, summary: str) -> float:
    o = len(original.split()) if original else 0
    s = len(summary.split()) if summary else 0
    if o == 0:
        return 0.0
    return round(100 * (1 - (s / o)), 2)

def alignment_label(alignment_ratio: float) -> str:
    if alignment_ratio >= 0.75:
        return f"🟩 High ({alignment_ratio:.2f})"
    if alignment_ratio >= 0.40:
        return f"🟧 Medium ({alignment_ratio:.2f})"
    return f"🟥 Low ({alignment_ratio:.2f})"

def safe_date(d):
    if isinstance(d, datetime.date):
        return d.isoformat()
    if isinstance(d, datetime.datetime):
        return d.date().isoformat()
    try:
        parsed = pd.to_datetime(d, errors="coerce")
        if pd.isna(parsed):
            return "unknown"
        return parsed.date().isoformat()
    except Exception:
        return "unknown"

# ---------------------------
# App initialization
# ---------------------------
init_db()

st.set_page_config(page_title="AI News Orchestrator", layout="wide")
st.markdown("""
<style>
    .stApp { padding: 1.5rem !important; }
    .stMarkdown { line-height: 1.8 !important; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="
    background-color:#1c1f26;
    padding:15px;
    border-radius:10px;
    border:1px solid #333;
    margin-bottom:15px;">
    <h2 style="margin:0; color:#00c2ff;">📰 AI News Orchestrator</h2>
    <p style="margin-top:5px; font-size:15px; color:#d0d0d0;">
        Reconstruct stories by aggregating, verifying, and summarizing multi-source news into a single event timeline.
    </p>
</div>
""", unsafe_allow_html=True)

# Locking & folders
LOCK_FILE = Path("locks/refresh.lock")
os.makedirs("locks", exist_ok=True)
os.makedirs("stats", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def is_refresh_running():
    return LOCK_FILE.exists()

def acquire_lock():
    try:
        LOCK_FILE.write_text(str(time.time()))
        return True
    except:
        return False

def release_lock():
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except:
        pass

def read_feeds():
    if not os.path.exists("feeds.txt"):
        return []
    with open("feeds.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh.readlines() if line.strip()]

def run_fetch_all(feeds, max_items=10):
    try:
        for f in feeds:
            process_feed(f, max_items=max_items)
    finally:
        release_lock()

def run_fetch_single(feed, max_items=15):
    try:
        process_feed(feed, max_items=max_items)
    finally:
        release_lock()

# Sidebar: top-level page switch (Option 2)
st.sidebar.markdown("""
<div style="font-size:20px; font-weight:700; margin-bottom:10px; color:#00c2ff;">
🔧 Control Panel
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Go to", ["Home / News Feed", "Generate Timeline", "Analytics Dashboard (Deluxe)"])

# Sidebar feed controls (shared)
feeds = read_feeds()
st.sidebar.write(f"Feeds detected: **{len(feeds)}**")
feed_choice = st.sidebar.selectbox("Pick feed to refresh (optional)", ["-- All feeds --"] + feeds)
max_items = st.sidebar.number_input("Max items per feed", min_value=1, max_value=50, value=10)

col1, col2 = st.sidebar.columns([1,1])

if col1.button("🔁 Refresh All Feeds"):
    if is_refresh_running():
        st.sidebar.warning("A refresh is already running…")
    else:
        if acquire_lock():
            st.sidebar.info("Started background refresh for ALL feeds.")
            threading.Thread(target=run_fetch_all, args=(feeds, max_items), daemon=True).start()

if col2.button("🔁 Refresh Selected Feed"):
    if feed_choice == "-- All feeds --":
        st.sidebar.warning("Select a specific feed first.")
    else:
        if is_refresh_running():
            st.sidebar.warning("Refresh already running…")
        else:
            if acquire_lock():
                st.sidebar.info(f"Refreshing: **{feed_choice}**")
                threading.Thread(target=run_fetch_single, args=(feed_choice, max_items), daemon=True).start()

if is_refresh_running():
    st.sidebar.success("Refresh Running…")
else:
    st.sidebar.info("No refresh running.")

with st.sidebar.expander("📊 Recent Metrics"):
    metrics_path = "stats/metrics.json"
    def safe_load_metrics(path):
        if not os.path.exists(path):
            return {"runs": []}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except:
            with open(path, "w") as fh:
                fh.write(json.dumps({"runs": []}, indent=2))
            return {"runs": []}
    metrics = safe_load_metrics(metrics_path)
    runs = metrics.get("runs", [])[-10:]
    for r in reversed(runs):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.get("timestamp", time.time())))
        st.write(f"{r.get('feed','?')} → {r.get('new_items',0)} new @ {ts}")

# ---------------------------
# PAGE: Home / News Feed
# ---------------------------
if page == "Home / News Feed":
    st.header("📰 Main News Feed")
    st.write("Browse latest fetched articles.")

    search = st.text_input("🔍 Search (title or summary)", key="search_home")
    tag_filter = st.text_input("🏷 Filter tags (comma separated)", key="tags_home")
    sort_by = st.selectbox("Sort by", ["fetched_at", "quality", "sentiment"], key="sort_home")

    @st.cache_data(ttl=30)
    def get_articles():
        return fetch_latest(200)
    articles = get_articles()

    if search:
        q = search.lower()
        articles = [
            a for a in articles
            if q in (a.get("title") or "").lower()
            or q in (a.get("summary") or "").lower()
        ]

    if tag_filter:
        wanted = [t.strip().lower() for t in tag_filter.split(",")]
        articles = [
            a for a in articles if any(w in (a.get("tags") or "").lower() for w in wanted)
        ]

    if sort_by == "quality":
        articles = sorted(articles, key=lambda x: (x.get("quality") or 0), reverse=True)
    elif sort_by == "sentiment":
        articles = sorted(articles, key=lambda x: (x.get("sentiment") or 0), reverse=True)
    else:
        try:
            articles = sorted(articles, key=lambda x: x.get("fetched_at") or "", reverse=True)
        except Exception:
            articles = articles

    # Display news cards
    for a in articles:
        st.markdown(
            "<div style='padding:15px; background:#1c1f26; border-radius:10px; border:1px solid #333;'>",
            unsafe_allow_html=True
        )

        # LEFT (sentiment badge) + CENTER (content + image)
        left_col, main_col = st.columns([0.6, 5])

        # ----------------------------------------
        # LEFT COLUMN — SENTIMENT BADGE
        # ----------------------------------------
        with left_col:
            s = a.get("sentiment", 0) or 0

            if s > 0.2:
                st.markdown(
                    "<span style='background:#162c16;color:#4ff266;padding:6px 10px;border-radius:8px;'>▲ Positive</span>",
                    unsafe_allow_html=True
                )
            elif s < -0.2:
                st.markdown(
                    "<span style='background:#2c1616;color:#ff4f4f;padding:6px 10px;border-radius:8px;'>▼ Negative</span>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<span style='background:#2a2a2a;color:#cccccc;padding:6px 10px;border-radius:8px;'>● Neutral</span>",
                    unsafe_allow_html=True
                )

        # ----------------------------------------
        # MAIN COLUMN — TITLE, SUMMARY, TAGS + GLASS BOX
        # ----------------------------------------
        with main_col:

            # Title
            st.subheader(a.get("title") or "Untitled")

            # Metadata
            st.caption(
                f"Source: {a.get('source')} | Category: {a.get('category')} | Quality: {a.get('quality')}"
            )

            # Summary text
            st.write(a.get("summary") or (a.get("content") or "")[:400])

            # --------------------------
            # TAGS + IMAGE SIDE BY SIDE
            # --------------------------

            tag_col, img_col = st.columns([1, 2])

            # LEFT COLUMN → TAGS (each tag separate)
            with tag_col:
                tags = (a.get("tags") or "")
                if tags:
                    for t in tags.split(","):
                        t = t.strip()   # important: remove spaces
                        st.markdown(
                              f"""
                             <div style="margin-bottom:6px;">
                                  <span style="
                                      background:rgba(0, 194, 255, 0.15);
                                      color:#00c2ff;
                                      padding:5px 10px;
                                      border-radius:12px;
                                      font-size:12px;
                                      display:inline-block;">
                                      {t}
                                  </span>
                             </div>
                             """,
                             unsafe_allow_html=True
                         )



        # RIGHT COLUMN → IMAGE
        with img_col:
             if a.get("image"):
                st.markdown(
                     f"""
                     <div style="text-align:center; margin-top:5px;">
                          <img src="{a.get('image')}" style="
                             width:85%;
                             border-radius:12px;
                              object-fit:cover;
                         ">
                     </div>
                     """,
                     unsafe_allow_html=True
                 )
             else:
                 st.write("")




# ---------------------------
# PAGE: Generate Timeline
# ---------------------------
elif page == "Generate Timeline":

    st.header("🕒 Event Timeline Generator")

    topic_input = st.text_input(
        "Enter topic / event (e.g., 'Trump messaging', 'COP30')",
        key="timeline_topic"
    )

    # Handle topic change (clear previous results)
    if "last_topic" not in st.session_state:
        st.session_state["last_topic"] = None

    if topic_input and topic_input.strip() != st.session_state["last_topic"]:
        st.session_state["last_topic"] = topic_input.strip()
        st.session_state.pop("_last_result", None)

    # Language selector
    summary_lang = st.selectbox(
        "🌍 Output Summary Language",
        ["English", "Tamil", "Hindi", "French", "Spanish", "Chinese", "Arabic", "German"],
        index=0
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        max_lookback = st.number_input(
            "Lookback articles (N)", min_value=20, max_value=1000, value=200
        )

    # ----------------- BUTTON TO GENERATE TIMELINE -----------------
    with col_b:
        if st.button("Generate Timeline"):
            if not topic_input or len(topic_input.strip()) < 3:
                st.warning("Enter at least 3 characters.")
            else:
                with st.spinner("Building timeline…"):
                    result = cached_timeline(topic_input.strip())

                # >>>>>>>>>>>>>>> INSERTED BLOCK START <<<<<<<<<<<<<<<<
                # ----------- LIVE MULTILINGUAL SWITCH (GOOGLETRANS) -----------
                if summary_lang != "English":
                    result["combined_summary"] = translate_text(result.get("combined_summary",""), summary_lang)
                    result["story_reconstruction"] = translate_text(result.get("story_reconstruction",""), summary_lang)

                    for entry in result.get("timeline", []):
                        for ev in entry.get("events", []):
                            ev["text"] = translate_text(ev.get("text",""), summary_lang)

                # Save result
                st.session_state["_last_result"] = result
                # >>>>>>>>>>>>>>> INSERTED BLOCK END <<<<<<<<<<<<<<<<

                st.session_state["_last_topic"] = topic_input.strip()

    # ----------- EXISTING LIVE MULTILINGUAL SWITCHING (KEEP AS IS) -----------
    if "_last_result" in st.session_state:
        result = st.session_state["_last_result"]

        if summary_lang != "English":
            result = apply_translation(result, summary_lang)
            st.session_state["_last_result"] = result
        else:
            # back to English
            result = cached_timeline(topic_input.strip())
            st.session_state["_last_result"] = result

        # ------------- RENDER RESULTS -------------
        st.subheader("🧠 Combined Event Summary")
        st.write(result.get("combined_summary", ""))

        st.caption(
            f"Sources: {', '.join(result.get('sources', [])[:10])} | "
            f"Articles matched: {result.get('articles_count')}"
        )

        st.metric("Authenticity / Reliability", f"{result.get('reliability',0):.2f}")

        st.markdown("## 🧭 Chronological Timeline")
        for entry in result.get("timeline", []):
            date_label = entry.get("date") or "Undated"
            st.markdown(f"<h4 style='color:#00c2ff'>{date_label}</h4>", unsafe_allow_html=True)
            st.markdown(
                f"<span style='background:#ffd84d33;color:#ffd84d;padding:4px 8px;border-radius:8px;'>"
                f"⭐ Importance Score: {entry.get('importance',0)}/10</span>",
                unsafe_allow_html=True
            )

            for ev in entry.get("events", []):
                st.write(f"- {ev.get('text')}")
                st.caption(f"Source: {ev.get('source')} — {ev.get('title')}")

        # Mermaid chart
        st.markdown("---")
        st.markdown("## 📊 Visual Timeline (Mermaid.js Gantt)")
        mermaid = result.get("mermaid_gantt","")
        if mermaid:
            st.markdown(f"```mermaid\n{mermaid}\n```")
        else:
            st.write("No mermaid timeline.")

        # Clusters
        st.markdown("## 🧩 Event Clusters")
        for cname, items in result.get("clusters", {}).items():
            st.markdown(f"### 🔹 {cname}")
            for t in items:
                st.write(f"- {t}")

        # Story
        with st.expander("📝 Story Reconstruction"):
            st.write(result.get("story_reconstruction",""))

        # Conflicts
        conflicts = result.get("conflicting_claims", [])
        if conflicts:
            with st.expander(f"⚠ Conflicting Claims ({len(conflicts)})"):
                for c in conflicts:
                    st.write(c)
        else:
            st.info("No conflicting claims detected.")

        # Bias
        bias_scores = result.get("bias_scores", {})
        if bias_scores:
            with st.expander("🔎 Bias / Clickbait Scores"):
                for url, b in bias_scores.items():
                    st.write(b)
        else:
            st.info("No bias scoring available.")





# ---------------------------
# PAGE: Analytics Dashboard (Deluxe)
# ---------------------------
elif page == "Analytics Dashboard (Deluxe)":
    st.header("📊 Deluxe Analytics Dashboard")
    st.write("This dashboard runs the 10 deluxe add-ons on the last generated timeline or a demo timeline.")

    # Load timeline result from temp cache if present
    timeline_result = None
    if Path("tmp/last_timeline.json").exists():
        try:
            timeline_result = json.loads(Path("tmp/last_timeline.json").read_text())
        except Exception:
            timeline_result = None

    demo_mode = False
    if not timeline_result:
        st.info("No cached timeline found. Using demo timeline for analytics. Generate a timeline first to analyze real results.")
        demo_mode = True
        # small demo
        now = datetime.date.today().isoformat()
        timeline_result = {
            "combined_summary": "Demo combined summary",
            "sources": ["https://www.bbc.co.uk/news", "https://www.nytimes.com"],
            "articles_count": 3,
            "reliability": 0.82,
            "timeline": [
                {"date": now, "importance": 8,
                 "events": [{"text": "Govt announces relief for small businesses.", "source": "https://www.bbc.co.uk/news", "title": "Relief announced"},
                            {"text": "Opposition calls it insufficient.", "source": "https://www.nytimes.com", "title": "Opposition criticizes"}],
                 "consensus": ["Relief package announced"], "conflict": []}
            ]
        }

    # Flatten events list for many operations
    events_flat = []
    for entry in timeline_result.get("timeline", []):
        for ev in entry.get("events", []):
            item = {
                "date": entry.get("date"),
                "text": ev.get("text", ""),
                "title": ev.get("title", ""),
                "source": ev.get("source", "unknown"),
                "original_text": ev.get("text", "")
            }
            # compute sentiment & reliability
            item["sentiment"] = simple_sentiment(item["text"])
            item["reliability"] = get_source_reputation(item["source"])
            events_flat.append(item)

    # 1) Source Credibility Scoreboard
    st.subheader("⭐ Source Credibility Scoreboard")
    sources = sorted({e.get("source", "unknown") for e in events_flat})
    rows = []
    for s in sources:
        rep = get_source_reputation(s)
        bias = "Low" if rep >= 0.8 else "Low-Mid" if rep >= 0.7 else "Mid" if rep >= 0.6 else "Unknown"
        rows.append({"Source": s, "Reliability": round(rep,2), "Bias": bias})
    score_df = pd.DataFrame(rows).sort_values("Reliability", ascending=False)
    st.table(score_df)

    # 2) Key Actors (NER)
    st.subheader("🧩 Key Actors (Named Entities)")
    persons, orgs = extract_ner_counts(events_flat)
    top_persons = persons.most_common(10)
    top_orgs = orgs.most_common(10)
    colp, colo = st.columns(2)
    with colp:
        st.write("People")
        if top_persons:
            for n, c in top_persons:
                st.write(f"- **{n}** ({c})")
        else:
            st.write("No PERSON entities found.")
    with colo:
        st.write("Organizations")
        if top_orgs:
            for n, c in top_orgs:
                st.write(f"- **{n}** ({c})")
        else:
            st.write("No ORG entities found.")

    # 3) Emotion Trend Chart
    st.subheader("📈 Emotion Trend Over Time")
    sent_series = pd.DataFrame([{"date": e["date"], "sentiment": e["sentiment"]} for e in events_flat])
    if not sent_series.empty:
        trend = sent_series.groupby("date").sentiment.mean().sort_index()
        st.line_chart(trend)
    else:
        st.write("No sentiment data available.")

    # 4) Headline Compression Meter
    st.subheader("✂️ Headline Compression Meter")
    orig_text = " ".join([e.get("original_text","") for e in events_flat])
    summary_text = timeline_result.get("combined_summary", "")
    comp = compression_score(orig_text, summary_text)
    st.write(f"Original text length (words): {len(orig_text.split())}")
    st.write(f"Summary length (words): {len(summary_text.split())}")
    st.metric("Compression Score", f"{comp}%")

    # 5) Event Density Insights
    st.subheader("📊 Event Density Insights")
    dates = [safe_date(e["date"]) for e in events_flat]
    date_counts = Counter(dates)
    if date_counts:
        peak_day = max(date_counts, key=date_counts.get)
        low_day = min(date_counts, key=date_counts.get)
        st.write(f"Total events: {len(events_flat)}")
        st.write(f"Days covered: {len(date_counts)}")
        st.write(f"Peak day: {peak_day} ({date_counts[peak_day]} events)")
        st.write(f"Quietest day: {low_day} ({date_counts[low_day]} events)")
        density_df = pd.DataFrame(list(date_counts.items()), columns=["date", "count"]).sort_values("date")
        st.bar_chart(density_df.set_index("date"))
    else:
        st.write("No date information available.")

    # 6) Cross-Source Alignment Percentage
    st.subheader("🔁 Cross-Source Alignment")
    summary_to_sources = defaultdict(set)
    for e in events_flat:
        s = (e.get("text") or e.get("title") or "").strip().lower()
        if s:
            summary_to_sources[s].add(e.get("source", "unknown"))
    consensus_counts = sum(1 for k,v in summary_to_sources.items() if len(v) > 1)
    total_unique_chunks = len(summary_to_sources) if summary_to_sources else 1
    alignment_ratio = consensus_counts / total_unique_chunks
    st.write(f"Consensus chunks: {consensus_counts} / {total_unique_chunks}")
    st.write("Alignment status:", alignment_label(alignment_ratio))

    # 7) Claim Heatmap (simple table)
    st.subheader("🔥 Claim Heatmap (Source contributions)")
    source_counts = Counter([e.get("source", "unknown") for e in events_flat])
    heat_df = pd.DataFrame({"source": list(source_counts.keys()), "events": list(source_counts.values())}).sort_values("events", ascending=False)
    st.table(heat_df)

    # 8) Auto-Generate Judge Summary
    st.subheader("📝 System Summary (Judge View)")
    avg_reliability = sum(e["reliability"] for e in events_flat) / (len(events_flat) or 1)
    judge_summary = {
        "Total sources used": len(set([e.get("source", "unknown") for e in events_flat])),
        "Total events extracted": len(events_flat),
        "Consensus level": alignment_label(alignment_ratio),
        "Conflict level": ("Low" if alignment_ratio >= 0.7 else "Medium" if alignment_ratio >= 0.4 else "High"),
        "Average reliability score": round(avg_reliability, 2),
        "Peak day": peak_day if date_counts else "N/A",
        "Quietest day": low_day if date_counts else "N/A",
    }
    st.json(judge_summary)

    # 9) Download Timeline as TXT / JSON
    st.subheader("📥 Downloads")
    combined_text = "\n\n".join([f"{e['date']} | {e['title']}\n{e['text']}\nSource: {e['source']}" for e in events_flat])
    st.download_button("Download Timeline (TXT)", combined_text, file_name="timeline.txt", mime="text/plain")
    st.download_button("Download Timeline (JSON)", json.dumps(timeline_result, indent=2), file_name="timeline.json", mime="application/json")

    # 10) Wordcloud (optional)
    st.subheader("☁️ Event Wordcloud")
    all_text = " ".join([e.get("text") or e.get("original_text") or "" for e in events_flat])
    if WORDCLOUD_AVAILABLE and all_text.strip():
        wc = WordCloud(width=800, height=300, background_color="white").generate(all_text)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        if not WORDCLOUD_AVAILABLE:
            st.info("WordCloud package not installed — skipping wordcloud. Install `wordcloud` to enable.")
        else:
            st.write("No text available for wordcloud.")

    st.markdown("---")
    st.caption("Tip: Generate a real timeline from the 'Generate Timeline' page to analyze live results. This demo uses cached timeline data.")

# ---------------------------
# End of app
# ---------------------------


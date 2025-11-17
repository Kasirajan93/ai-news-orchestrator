from apscheduler.schedulers.background import BackgroundScheduler
from fetcher import process_feed
import time
import logging
import json
import os

# logger
logging.basicConfig(filename="logs/scheduler.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

METRICS_FILE = "stats/metrics.json"
os.makedirs("logs", exist_ok=True)
os.makedirs("stats", exist_ok=True)

def safe_process(feed):
    for i in range(3):
        try:
            new_items = process_feed(feed)
            return new_items
        except Exception as e:
            logging.warning(f"Attempt {i+1} failed for {feed}: {e}")
            time.sleep(2)
    logging.error(f"All attempts failed for {feed}")
    return 0

def update_metrics(feed, new_items):
    metrics = {"runs": []}
    try:
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, "r", encoding="utf-8") as fh:
                    metrics = json.load(fh)
            except json.JSONDecodeError:
                logging.warning("Corrupted metrics.json detected. Resetting.")
                metrics = {"runs": []}

        if "runs" not in metrics:
            metrics["runs"] = []

        metrics_entry = {
            "feed": feed,
            "new_items": new_items,
            "timestamp": int(time.time())
        }

        metrics["runs"].append(metrics_entry)

        with open(METRICS_FILE, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

    except Exception as e:
        logging.error(f"Failed updating metrics: {e}")


def start_scheduler():
    scheduler = BackgroundScheduler()
    with open("feeds.txt", "r") as f:
        feeds = [line.strip() for line in f.readlines() if line.strip()]

    # keep adaptive delays per feed
    last_new = {f: 1 for f in feeds}

    for feed in feeds:
        def job_fn(f=feed):
            logging.info(f"Running scheduled job for {f}")
            new_items = safe_process(f)
            update_metrics(f, new_items)
            # adaptive: if 0 new items previously, delay next run a bit by rescheduling
            if new_items == 0:
                logging.info(f"No new items for {f}; next interval will be longer.")
        scheduler.add_job(job_fn, 'interval', minutes=30, id=f"job_{feed}", replace_existing=True)

    scheduler.start()
    logging.info("Scheduler started")
    print("Scheduler running... Press CTRL+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user.")
        scheduler.shutdown()

if __name__ == "__main__":
    start_scheduler()

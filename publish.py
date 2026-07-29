#!/usr/bin/env python3
"""
Market Intelligence pipeline - publish to Power Automate.

Sends scored articles to a Power Automate HTTP trigger as a JSON POST, then
marks them published so they are never sent twice.

Configure with an environment variable (NEVER hardcode this - the URL contains
a 'sig=' token that lets anyone trigger your flow):

    export MI_WEBHOOK_URL="https://<your-flow-trigger-url>"

Usage:
    python3 publish.py --dry-run          build the payload, print it, send nothing
    python3 publish.py --dry-run --save   also write it to sample_payload.json
    python3 publish.py --limit 3          send three articles as a test
    python3 publish.py                    send everything pending
    python3 publish.py --min-score 50     only send articles scoring 50+
    python3 publish.py --relevant-only    skip articles judged Not relevant
    python3 publish.py --resend           re-send articles already published

Why this sends application/json: Power Automate's HTTP trigger parses a JSON
body natively into usable fields. Anything else (text/html, octet-stream)
arrives base64-encoded inside a $content property and has to be decoded with
expressions inside the flow.
"""

import argparse
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "articles.db")

WEBHOOK = os.environ.get("MI_WEBHOOK_URL", "")
BATCH_SIZE = int(os.environ.get("MI_BATCH_SIZE", "25"))
TIMEOUT = int(os.environ.get("MI_TIMEOUT", "120"))

FIELDS = [
    "url", "title", "source", "source_kind", "published_raw", "language",
    "first_seen", "relevance", "relevant", "score", "priority", "category",
    "categories", "companies", "competitors", "projects", "regions",
    "market_signal", "information_status", "recommended_action",
    "summary", "reason", "nkt_implication",
]

EXTRA_COLUMNS = [
    ("published_at", "TEXT"),
    ("publish_error", "TEXT"),
]


def connect():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def migrate(conn):
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(articles)")}
    added = []
    for name, coltype in EXTRA_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE articles ADD COLUMN {name} {coltype}")
            added.append(name)
    if added:
        conn.commit()
        print(f"schema: added {', '.join(added)}", file=sys.stderr)


INT_FIELDS = {"score", "relevant"}


def clean(value):
    """Power Automate handles missing properties badly. Never emit null."""
    if value is None:
        return ""
    return value


def clean_field(name, value):
    """Keep types stable.

    Power Automate's Parse JSON validates against the schema you generated from
    a sample, so a field that is 92 in one payload and "92" in another will fail
    the flow. SQLite is loosely typed, so pin the numeric fields explicitly.
    """
    if name in INT_FIELDS:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return clean(value)


def fetch_pending(conn, limit=None, resend=False, min_score=None,
                  relevant_only=False):
    have = {r["name"] for r in conn.execute("PRAGMA table_info(articles)")}
    missing = [f for f in FIELDS if f not in have]
    if "scored_at" not in have:
        sys.exit("This database has not been scored yet.\n"
                 "  Run: python3 score.py --limit 5")
    if missing:
        print(f"note: columns not present, will be sent empty: "
              f"{', '.join(missing)}", file=sys.stderr)

    where = ["scored_at IS NOT NULL", "scored_at != ''"]
    if "score_error" in have:
        where.append("(score_error IS NULL OR score_error = '')")
    params = []

    if not resend:
        where.append("(published_at IS NULL OR published_at = '')")
    if min_score is not None and "score" in have:
        where.append("score >= ?")
        params.append(min_score)
    if relevant_only and "relevant" in have:
        where.append("relevant = 1")

    present = [f for f in FIELDS if f in have]
    order = "score DESC, first_seen DESC" if "score" in have else "first_seen DESC"
    sql = (f"SELECT {', '.join(present)} FROM articles"
           f" WHERE {' AND '.join(where)} ORDER BY {order}")
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall(), missing


def build_payload(rows, run_id, missing=()):
    articles = []
    for r in rows:
        keys = r.keys()
        articles.append({
            f: (clean_field(f, None) if (f in missing or f not in keys)
                else clean_field(f, r[f]))
            for f in FIELDS})
    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "article_count": len(articles),
        "articles": articles,
    }


def post(payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK,
        data=body,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")[:400]


def main():
    ap = argparse.ArgumentParser(description="MI pipeline - publish to Power Automate")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the payload without sending")
    ap.add_argument("--save", action="store_true",
                    help="with --dry-run, also write sample_payload.json")
    ap.add_argument("--limit", type=int, help="send at most N articles")
    ap.add_argument("--min-score", type=int, help="only articles scoring >= N")
    ap.add_argument("--relevant-only", action="store_true",
                    help="skip articles judged Not relevant")
    ap.add_argument("--resend", action="store_true",
                    help="include articles already published")
    args = ap.parse_args()

    if not args.dry_run and not WEBHOOK:
        sys.exit("MI_WEBHOOK_URL is not set.\n"
                 '  export MI_WEBHOOK_URL="https://...your flow trigger URL..."')

    conn = connect()
    migrate(conn)

    rows, missing = fetch_pending(conn, limit=args.limit, resend=args.resend,
                                  min_score=args.min_score,
                                  relevant_only=args.relevant_only)
    if not rows:
        print("nothing to publish", file=sys.stderr)
        return

    run_id = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"{len(rows)} article(s) to publish", file=sys.stderr)

    if args.dry_run:
        payload = build_payload(rows, run_id, missing)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.save:
            out = os.path.join(HERE, "sample_payload.json")
            open(out, "w", encoding="utf-8").write(text)
            print(f"written to {out}", file=sys.stderr)
        print(text)
        print("\n(dry run - nothing sent, nothing marked published)",
              file=sys.stderr)
        return

    sent = failed = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        payload = build_payload(batch, run_id, missing)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        try:
            status, body = post(payload)
            for r in batch:
                conn.execute(
                    "UPDATE articles SET published_at = ?, publish_error = ''"
                    " WHERE url = ?", (now, r["url"]))
            conn.commit()
            sent += len(batch)
            print(f"  batch {start // BATCH_SIZE + 1}: {len(batch)} sent"
                  f" (HTTP {status})", file=sys.stderr)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            err = f"HTTP {e.code}: {detail}"
            for r in batch:
                conn.execute("UPDATE articles SET publish_error = ? WHERE url = ?",
                             (err, r["url"]))
            conn.commit()
            failed += len(batch)
            print(f"! batch {start // BATCH_SIZE + 1} failed: {err}",
                  file=sys.stderr)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            for r in batch:
                conn.execute("UPDATE articles SET publish_error = ? WHERE url = ?",
                             (err, r["url"]))
            conn.commit()
            failed += len(batch)
            print(f"! batch {start // BATCH_SIZE + 1} failed: {err}",
                  file=sys.stderr)

    print(f"\nsent: {sent} | failed: {failed}", file=sys.stderr)
    conn.close()


if __name__ == "__main__":
    main()

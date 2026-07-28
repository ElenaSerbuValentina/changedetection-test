#!/usr/bin/env python3
"""
Market Intelligence pipeline - extraction layer.

Takes articles that ingest.py recorded with status='new', fetches each one,
strips navigation/ads/boilerplate, and stores the clean article text.

    python3 extract.py --limit 5      try five articles first
    python3 extract.py                process everything pending
    python3 extract.py --retry        re-attempt previously failed articles
    python3 extract.py --show <url>   print what was extracted for one article

Status flow:
    new  ->  extracted     text retrieved successfully
         ->  empty         page fetched but no article text found
         ->  failed        fetch error, or a format we can't read (PDF etc.)

Dependencies:
    pip3 install --user trafilatura
"""

import argparse
import gzip
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "articles.db")

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Be polite: minimum seconds between requests to the same host.
HOST_DELAY = 2.0
MIN_TEXT_CHARS = 200

try:
    import trafilatura
except ImportError as e:
    sys.exit(
        f"Could not import trafilatura: {e}\n\n"
        f"This interpreter is: {sys.executable}\n"
        f"Install into THIS interpreter (not whatever 'pip3' points at):\n"
        f"    {sys.executable} -m pip install --user trafilatura\n\n"
        f"If the error above names a different module, trafilatura is installed\n"
        f"but one of its dependencies is missing or broken - install that instead."
    )


# ---------------------------------------------------------------------------
# Schema migration - adds the extraction columns to an existing database
# ---------------------------------------------------------------------------

EXTRA_COLUMNS = [
    ("text", "TEXT"),
    ("text_chars", "INTEGER"),
    ("language", "TEXT"),
    ("meta_date", "TEXT"),
    ("meta_title", "TEXT"),
    ("fetched_at", "TEXT"),
    ("extract_note", "TEXT"),
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


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def http_get(url, timeout=30):
    """Fetch with a browser User-Agent.

    trafilatura's own fetch_url() sends a 'trafilatura/x.y' User-Agent, which
    many corporate sites reject outright. This is the same approach ingest.py
    uses, which is already proven against these hosts.

    Returns (content_type, text) or raises.
    """
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip",
        "Accept-Language": "en,*;q=0.5",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        ctype = (resp.headers.get("Content-Type") or "").lower()
        charset = resp.headers.get_content_charset() or "utf-8"
        return ctype, raw.decode(charset, errors="replace")


def toggle_www(url):
    """https://x.com/a  <->  https://www.x.com/a"""
    p = urllib.parse.urlsplit(url)
    if p.netloc.startswith("www."):
        netloc = p.netloc[4:]
    else:
        netloc = "www." + p.netloc
    return urllib.parse.urlunsplit((p.scheme, netloc, p.path, p.query, p.fragment))


def fetch_and_extract(url):
    """Return (status, payload_dict). Never raises."""
    attempts = [url]
    try:
        ctype, downloaded = http_get(url)
    except urllib.error.HTTPError as e:
        # A canonicalised URL may have had 'www.' stripped for dedup purposes;
        # some hosts 404 without it. Try the other form before giving up.
        if e.code in (404, 403):
            alt = toggle_www(url)
            attempts.append(alt)
            try:
                ctype, downloaded = http_get(alt)
            except Exception:
                return "failed", {"extract_note": f"HTTP {e.code} (tried both www forms)"}
        else:
            return "failed", {"extract_note": f"HTTP {e.code}"}
    except Exception as e:
        return "failed", {"extract_note": f"fetch error: {type(e).__name__}: {e}"}

    if not downloaded:
        return "failed", {"extract_note": "empty response body"}

    # trafilatura handles HTML only. Flag anything else rather than feeding it junk.
    if "pdf" in ctype or downloaded.lstrip()[:8].lower().startswith("%pdf"):
        return "failed", {"extract_note": "PDF - needs a separate reader"}
    if ctype and "html" not in ctype and "xml" not in ctype and "text" not in ctype:
        return "failed", {"extract_note": f"unsupported content-type: {ctype[:60]}"}

    try:
        result = trafilatura.extract(
            downloaded,
            url=url,
            output_format="json",
            with_metadata=True,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
    except Exception as e:
        return "failed", {"extract_note": f"extract error: {type(e).__name__}: {e}"}

    if not result:
        return "empty", {"extract_note": "no article text found in page"}

    import json as _json
    try:
        data = _json.loads(result)
    except Exception:
        return "empty", {"extract_note": "extractor returned unparseable output"}

    text = (data.get("text") or "").strip()
    if len(text) < MIN_TEXT_CHARS:
        return "empty", {
            "text": text,
            "text_chars": len(text),
            "extract_note": f"only {len(text)} chars - likely a stub or consent wall",
        }

    return "extracted", {
        "text": text,
        "text_chars": len(text),
        "language": data.get("language") or "",
        "meta_date": data.get("date") or "",
        "meta_title": (data.get("title") or "").strip()[:500],
        "extract_note": "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(limit=None, retry=False, quiet=False):
    conn = connect()
    migrate(conn)

    wanted = ("new", "failed", "empty") if retry else ("new",)
    placeholders = ",".join("?" * len(wanted))
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(articles)")}
    fetch_col = "fetch_url" if "fetch_url" in cols else "url"
    sql = (f"SELECT url, COALESCE(NULLIF({fetch_col},''), url) AS fetch_target,"
           f" title, source FROM articles WHERE status IN ({placeholders})"
           " ORDER BY first_seen DESC")
    params = list(wanted)
    if limit:
        sql += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("nothing pending", file=sys.stderr)
        return

    print(f"{len(rows)} article(s) to process\n", file=sys.stderr)

    last_hit = defaultdict(float)
    tally = defaultdict(int)

    for i, row in enumerate(rows, 1):
        url = row["url"]
        target = row["fetch_target"]
        host = urllib.parse.urlsplit(target).netloc

        wait = HOST_DELAY - (time.monotonic() - last_hit[host])
        if wait > 0:
            time.sleep(wait)
        last_hit[host] = time.monotonic()

        status, payload = fetch_and_extract(target)
        tally[status] += 1

        payload["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fields = ", ".join(f"{k} = ?" for k in payload)
        conn.execute(
            f"UPDATE articles SET status = ?, {fields} WHERE url = ?",
            [status] + list(payload.values()) + [url],
        )
        conn.commit()

        if not quiet or status != "extracted":
            mark = {"extracted": " ", "empty": "?", "failed": "!"}[status]
            chars = payload.get("text_chars", 0)
            note = payload.get("extract_note", "")
            label = (row["title"] or url)[:58]
            print(f"{mark} [{i}/{len(rows)}] {label:<60} {chars:>6} chars"
                  f" {note}", file=sys.stderr)

    print("\n" + " | ".join(f"{k}: {v}" for k, v in sorted(tally.items())),
          file=sys.stderr)

    remaining = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE status = 'extracted'").fetchone()[0]
    print(f"total extracted in database: {remaining}", file=sys.stderr)
    conn.close()


def show(url):
    conn = connect()
    row = conn.execute(
        "SELECT * FROM articles WHERE url = ? OR url LIKE ?",
        (url, f"%{url}%")).fetchone()
    if not row:
        sys.exit(f"no article matching {url!r}")

    for key in ("source", "title", "meta_title", "url", "language",
                "published_raw", "meta_date", "status", "text_chars",
                "extract_note"):
        if key in row.keys():
            print(f"{key:>14}: {row[key]}")
    print("\n--- text ---\n")
    print((row["text"] or "")[:3000])
    conn.close()


def main():
    ap = argparse.ArgumentParser(description="MI pipeline - article extraction")
    ap.add_argument("--limit", type=int, help="process at most N articles")
    ap.add_argument("--retry", action="store_true",
                    help="also re-attempt previously failed/empty articles")
    ap.add_argument("--show", metavar="URL", help="print one article's extraction")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="only log problems")
    args = ap.parse_args()

    if args.show:
        show(args.show)
        return
    process(limit=args.limit, retry=args.retry, quiet=args.quiet)


if __name__ == "__main__":
    main()

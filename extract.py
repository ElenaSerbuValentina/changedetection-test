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
import re
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

class _Redirect308(urllib.request.HTTPRedirectHandler):
    """urllib's default handler covers 301/302/303/307 but not 308."""

    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_301(req, fp, 301, msg, headers)

    https_error_308 = http_error_308


_OPENER = urllib.request.build_opener(_Redirect308)


def http_get(url, timeout=30):
    """Fetch with a browser User-Agent.

    trafilatura's own fetch_url() sends a 'trafilatura/x.y' User-Agent, which
    many corporate sites reject outright. This is the same approach ingest.py
    uses, which is already proven against these hosts.

    Returns (content_type, raw_bytes, charset) or raises.
    Raw bytes rather than text, because PDFs must not be decoded.
    """
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
        "Accept-Encoding": "gzip",
        "Accept-Language": "en,*;q=0.5",
    })
    with _OPENER.open(req, timeout=timeout) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        ctype = (resp.headers.get("Content-Type") or "").lower()
        charset = resp.headers.get_content_charset() or "utf-8"
        return ctype, raw, charset


def is_pdf(ctype, raw):
    return "pdf" in ctype or raw[:5].lstrip()[:4] == b"%PDF"


def pdf_to_text(raw):
    """Extract text from PDF bytes. Returns (text, note)."""
    try:
        from pdfminer.high_level import extract_text as _pdf_text
        from pdfminer.layout import LAParams
    except ImportError:
        return "", "PDF - pdfminer.six not installed"

    try:
        import io
        import logging
        # pdfminer is extremely chatty about malformed PDFs
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        text = _pdf_text(io.BytesIO(raw), laparams=LAParams())
    except Exception as e:
        return "", f"PDF unreadable: {type(e).__name__}"

    # Collapse the ragged line breaks pdfminer produces from column layout,
    # while keeping paragraph boundaries.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(lines).strip(), ""


def detect_language(text):
    """Best-effort language tag. Optional dependency; empty string if absent."""
    if not text:
        return ""
    try:
        import py3langid
    except ImportError:
        return ""
    try:
        code, _ = py3langid.classify(text[:2000])
        return code or ""
    except Exception:
        return ""


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
    try:
        ctype, raw, charset = http_get(url)
    except urllib.error.HTTPError as e:
        # A canonicalised URL may have had 'www.' stripped for dedup purposes;
        # some hosts 404 without it. Try the other form before giving up.
        if e.code in (404, 403):
            try:
                ctype, raw, charset = http_get(toggle_www(url))
            except Exception:
                return "failed", {"extract_note": f"HTTP {e.code} (tried both www forms)"}
        else:
            return "failed", {"extract_note": f"HTTP {e.code}"}
    except Exception as e:
        return "failed", {"extract_note": f"fetch error: {type(e).__name__}: {e}"}

    if not raw:
        return "failed", {"extract_note": "empty response body"}

    # --- PDF path -----------------------------------------------------------
    # Sumitomo and Furukawa publish many press releases as PDFs rather than
    # HTML pages. Without this they are simply invisible to the pipeline.
    if is_pdf(ctype, raw):
        text, note = pdf_to_text(raw)
        if not text:
            return "failed", {"extract_note": note or "PDF produced no text"}
        if len(text) < MIN_TEXT_CHARS:
            return "empty", {
                "text": text,
                "text_chars": len(text),
                "language": detect_language(text),
                "extract_note": f"PDF, only {len(text)} chars - likely a scan",
            }
        return "extracted", {
            "text": text,
            "text_chars": len(text),
            "language": detect_language(text),
            "meta_date": "",
            "meta_title": "",
            "extract_note": "from PDF",
        }

    # --- HTML path ----------------------------------------------------------
    downloaded = raw.decode(charset, errors="replace")

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
        # trafilatura only runs its classifier when target_language is set, and
        # that parameter also FILTERS - which would discard most of this corpus.
        # So detect separately instead.
        "language": detect_language(text),
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

#!/usr/bin/env python3
"""
Market Intelligence pipeline - ingestion layer.

Reads sources.json, fetches each listing page (or RSS feed), extracts articles,
and records anything not seen before in a SQLite database. Only genuinely new
articles are emitted.

    python3 ingest.py --init                  create the database
    python3 ingest.py --inspect "Prysmian - press releases"
                                              help work out selectors for a site
    python3 ingest.py --dry-run               fetch and report, store nothing
    python3 ingest.py                         normal run
    python3 ingest.py --backfill              record everything as seen without
                                              emitting it (run once per new source
                                              so you don't get 50 "new" articles)

Dependencies:
    pip3 install --user selectolax
If pip is unavailable:
    sudo dnf install python3-lxml     (the script falls back to lxml)
"""

import argparse
import gzip
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "sources.json")
DB = os.path.join(HERE, "articles.db")

TIMEOUT = 25
RETRIES = 2
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Query params that identify a page vs. ones that are just tracking noise.
TRACKING_PARAMS = re.compile(
    r"^(utm_|fbclid|gclid|mc_cid|mc_eid|_ga|ref|source|campaign)", re.I
)

# ----------------------------------------------------------------------------
# HTML parsing - selectolax preferred, lxml as fallback
# ----------------------------------------------------------------------------

_PARSER = None

def _load_parser():
    global _PARSER
    if _PARSER is not None:
        return _PARSER
    try:
        from selectolax.parser import HTMLParser

        class Sel:
            name = "selectolax"

            @staticmethod
            def parse(html):
                return HTMLParser(html)

            @staticmethod
            def select(node, css):
                return node.css(css) if css else []

            @staticmethod
            def select_one(node, css):
                return node.css_first(css) if css else None

            @staticmethod
            def text(node):
                if node is None:
                    return ""
                # separator matters: <br> and nested tags would otherwise
                # concatenate adjacent words into one string
                return " ".join(node.text(separator=" ", strip=True).split())

            @staticmethod
            def attr(node, name):
                return (node.attributes.get(name) or "") if node is not None else ""

        _PARSER = Sel
        return _PARSER
    except ImportError:
        pass

    try:
        from lxml import html as lhtml
        from lxml.cssselect import CSSSelector

        _cache = {}

        def _sel(css):
            if css not in _cache:
                _cache[css] = CSSSelector(css)
            return _cache[css]

        class Lx:
            name = "lxml"

            @staticmethod
            def parse(html):
                return lhtml.fromstring(html)

            @staticmethod
            def select(node, css):
                return _sel(css)(node) if css else []

            @staticmethod
            def select_one(node, css):
                if not css:
                    return None
                found = _sel(css)(node)
                return found[0] if found else None

            @staticmethod
            def text(node):
                if node is None:
                    return ""
                return " ".join(node.text_content().split())

            @staticmethod
            def attr(node, name):
                return (node.get(name) or "") if node is not None else ""

        _PARSER = Lx
        return _PARSER
    except ImportError:
        pass

    sys.exit(
        "No HTML parser available.\n"
        "  pip3 install --user selectolax\n"
        "or\n"
        "  sudo dnf install python3-lxml python3-cssselect"
    )

# ----------------------------------------------------------------------------
# Fetching
# ----------------------------------------------------------------------------

def fetch(url, timeout=TIMEOUT):
    last = None
    for attempt in range(RETRIES + 1):
        try:
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
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.geturl(), raw.decode(charset, errors="replace")
        except Exception as e:
            last = e
            if attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
    raise last

# ----------------------------------------------------------------------------
# URL canonicalisation - this is what makes dedup work
# ----------------------------------------------------------------------------

def unwrap_google(url):
    """Google Alerts wrap targets in google.com/url?...&url=REAL. Unwrap them."""
    p = urllib.parse.urlparse(url)
    if "google." not in p.netloc or not p.path.startswith("/url"):
        return url
    q = urllib.parse.parse_qs(p.query)
    for key in ("url", "q"):
        if key in q and q[key]:
            return q[key][0]
    return url


def canonical(url, base=None):
    """Absolute, tracking-free, fragment-free, trailing-slash-normalised."""
    if base:
        url = urllib.parse.urljoin(base, url)
    url = unwrap_google(url)
    p = urllib.parse.urlsplit(url)

    # Scheme is normalised for the dedup KEY only. The real URL is preserved
    # separately in fetch_url, so http-only hosts still get fetched correctly.
    scheme = p.scheme.lower() or "https"
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if (scheme == "https" and netloc.endswith(":443")) or \
       (scheme == "http" and netloc.endswith(":80")):
        netloc = netloc.rsplit(":", 1)[0]
    scheme = "https"

    keep = [(k, v) for k, v in urllib.parse.parse_qsl(p.query)
            if not TRACKING_PARAMS.match(k)]
    query = urllib.parse.urlencode(sorted(keep))

    path = p.path
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))

# ----------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------

def extract_scrape(source, base_url, html):
    P = _load_parser()
    tree = P.parse(html)
    out = []

    items = P.select(tree, source.get("item", ""))
    if not items:
        return out

    for it in items:
        link_node = P.select_one(it, source.get("link") or "a")
        href = P.attr(link_node, "href")
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue

        title_node = P.select_one(it, source.get("title", "")) or link_node
        title = P.text(title_node)
        date = P.text(P.select_one(it, source.get("date", "")))

        out.append({
            "url": canonical(href, base_url),
            "fetch_url": urllib.parse.urljoin(base_url, href),
            "title": " ".join(title.split())[:500],
            "published_raw": date[:100],
        })
    return out


def extract_rss(source, base_url, xml_text):
    out = []
    try:
        root = ET.fromstring(xml_text.encode("utf-8", errors="replace"))
    except ET.ParseError as e:
        raise ValueError(f"not valid XML: {e}")

    ns = {"a": "http://www.w3.org/2005/Atom"}

    for item in root.iter():
        tag = item.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue

        link = ""
        title = ""
        pub = ""
        for child in item:
            ctag = child.tag.split("}")[-1]
            if ctag == "link":
                link = (child.get("href") or child.text or "").strip()
            elif ctag == "title":
                title = "".join(child.itertext()).strip()
            elif ctag in ("pubDate", "published", "updated"):
                pub = (child.text or "").strip()

        if not link:
            continue

        title = re.sub(r"<[^>]+>", "", title)
        out.append({
            "url": canonical(link, base_url),
            "fetch_url": unwrap_google(urllib.parse.urljoin(base_url, link)),
            "title": " ".join(title.split())[:500],
            "published_raw": pub[:100],
        })
    return out

# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    url           TEXT PRIMARY KEY,
    fetch_url     TEXT,
    title         TEXT,
    source        TEXT NOT NULL,
    source_kind   TEXT,
    published_raw TEXT,
    first_seen    TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'new'
);
CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);

CREATE TABLE IF NOT EXISTS runs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started   TEXT NOT NULL,
    source    TEXT NOT NULL,
    found     INTEGER NOT NULL DEFAULT 0,
    new_items INTEGER NOT NULL DEFAULT 0,
    error     TEXT
);
"""


def connect():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = connect()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print(f"database ready: {DB}")

# ----------------------------------------------------------------------------
# Inspect mode - helps you write selectors for a new site
# ----------------------------------------------------------------------------

def inspect(name, config):
    source = next((s for s in config["sources"] if s["name"] == name), None)
    if not source:
        sys.exit(f"no source named {name!r}")

    final_url, html = fetch(source["url"])
    P = _load_parser()
    tree = P.parse(html)
    print(f"parser: {P.name}   fetched: {final_url}   {len(html)} bytes\n")

    if source.get("item"):
        items = P.select(tree, source["item"])
        print(f"selector {source['item']!r} matched {len(items)} items")
        for it in items[:3]:
            link = P.select_one(it, source.get("link") or "a")
            print("   link :", P.attr(link, "href"))
            print("   title:", P.text(P.select_one(it, source.get("title", "")) or link)[:110])
            print("   date :", P.text(P.select_one(it, source.get("date", ""))))
            print()
        return

    # No selector yet: suggest repeating structures by grouping link paths.
    print("no 'item' selector set. Most common link path shapes on this page:\n")
    counts = {}
    for a in P.select(tree, "a"):
        href = P.attr(a, "href")
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        path = urllib.parse.urlsplit(urllib.parse.urljoin(final_url, href)).path
        shape = re.sub(r"/[^/]*\d[^/]*", "/<slug>", path)
        shape = "/".join(shape.split("/")[:4])
        counts.setdefault(shape, []).append(href)

    for shape, hrefs in sorted(counts.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"  {len(hrefs):4d}  {shape}")
        print(f"        e.g. {hrefs[0][:100]}")
    print("\nPick the shape that looks like articles, then find the repeating"
          "\nparent element in the page source and set 'item'.")

# ----------------------------------------------------------------------------
# Main run
# ----------------------------------------------------------------------------

def run(config, dry_run=False, backfill=False, only=None, quiet=False):
    conn = connect()
    conn.executescript(SCHEMA)
    # migrate older databases that predate fetch_url
    cols = {r[1] for r in conn.execute("PRAGMA table_info(articles)")}
    if "fetch_url" not in cols:
        conn.execute("ALTER TABLE articles ADD COLUMN fetch_url TEXT")
        conn.commit()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    emitted = []
    totals = {"sources": 0, "found": 0, "new": 0, "errors": 0}

    for source in config["sources"]:
        if not source.get("enabled"):
            continue
        if only and source["name"] != only:
            continue
        if "PASTE_" in source.get("url", ""):
            print(f"[skip] {source['name']}: url placeholder not filled in",
                  file=sys.stderr)
            continue

        found = new_count = 0
        error = None

        try:
            final_url, body = fetch(source["url"])
            if source.get("kind") == "rss":
                items = extract_rss(source, final_url, body)
            else:
                items = extract_scrape(source, final_url, body)

            found = len(items)
            if found == 0:
                error = "selector matched nothing"

            for item in items:
                if not item["url"]:
                    continue
                cur = conn.execute(
                    "INSERT OR IGNORE INTO articles"
                    " (url, fetch_url, title, source, source_kind, published_raw,"
                    "  first_seen, status)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (item["url"], item.get("fetch_url") or item["url"],
                     item["title"], source["name"],
                     source.get("kind", "scrape"), item["published_raw"], now,
                     "seen" if backfill else "new"),
                )
                if cur.rowcount:
                    new_count += 1
                    if not backfill:
                        emitted.append({
                            "url": item["url"],
                            "title": item["title"],
                            "source": source["name"],
                            "published_raw": item["published_raw"],
                            "first_seen": now,
                        })
                else:
                    # Already known. Fill in fetch_url if this row was created
                    # before that column existed. Not a new article.
                    conn.execute(
                        "UPDATE articles SET fetch_url = ?"
                        " WHERE url = ? AND (fetch_url IS NULL OR fetch_url = '')",
                        (item.get("fetch_url") or item["url"], item["url"]),
                    )

        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        conn.execute(
            "INSERT INTO runs (started, source, found, new_items, error)"
            " VALUES (?,?,?,?,?)",
            (now, source["name"], found, new_count, error),
        )

        totals["sources"] += 1
        totals["found"] += found
        totals["new"] += new_count
        if error:
            totals["errors"] += 1

        flag = "!" if error else " "
        if not quiet or error or new_count:
            print(f"{flag} {source['name']:<38} found={found:<4} new={new_count:<4}"
                  f" {error or ''}", file=sys.stderr)

    if dry_run:
        conn.rollback()
        print("\n(dry run - nothing written)", file=sys.stderr)
    else:
        conn.commit()
    conn.close()

    if backfill:
        print(f"\nbackfill: {totals['sources']} sources, {totals['found']} articles"
              f" marked as seen, {totals['errors']} errors", file=sys.stderr)
    else:
        print(f"\n{totals['sources']} sources | {totals['found']} articles listed"
              f" | {totals['new']} new | {totals['errors']} errors", file=sys.stderr)
        # Only dump JSON when it's going somewhere useful - a pipe or a file.
        # Printing 150 articles into a terminal helps nobody.
        if not quiet and not sys.stdout.isatty():
            json.dump(emitted, sys.stdout, ensure_ascii=False, indent=2)
            print()
        elif emitted and sys.stdout.isatty():
            print(f"({len(emitted)} new articles held back - redirect to a file"
                  f" to capture them, e.g. > corpus.json)", file=sys.stderr)
    return emitted


def main():
    ap = argparse.ArgumentParser(description="MI pipeline ingestion")
    ap.add_argument("--init", action="store_true", help="create the database")
    ap.add_argument("--inspect", metavar="SOURCE_NAME",
                    help="help work out selectors for one source")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and report without writing")
    ap.add_argument("--backfill", action="store_true",
                    help="mark everything currently listed as already seen")
    ap.add_argument("--only", metavar="SOURCE_NAME", help="run a single source")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="only log sources with errors or new articles")
    ap.add_argument("--config", default=CONFIG)
    args = ap.parse_args()

    if args.init:
        init_db()
        return

    with open(args.config, encoding="utf-8") as fh:
        config = json.load(fh)

    if args.inspect:
        inspect(args.inspect, config)
        return

    run(config, dry_run=args.dry_run, backfill=args.backfill, only=args.only,
        quiet=args.quiet)


if __name__ == "__main__":
    main()

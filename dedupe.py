#!/usr/bin/env python3
"""
Market Intelligence pipeline - near-duplicate detection.

The same story arrives many times: a Prysmian press release is picked up by the
"Prysmian" Google Alert via three trade outlets, so the pipeline sees four
articles with different URLs and different headlines saying the same thing.
URL dedup in ingest.py cannot catch this.

This stage clusters articles that report the same event, keeps one primary, and
marks the rest status='duplicate' so they never reach scoring, publishing or the
newsletter. Nothing is deleted - duplicates keep a dup_of pointer to their
primary, so you can always see what was collapsed and why.

Runs BETWEEN extract.py and score.py. Deliberately: collapsing four articles
into one before scoring saves three LLM calls.

    python3 dedupe.py --dry-run       show clusters, change nothing
    python3 dedupe.py                 mark duplicates
    python3 dedupe.py --window 14     widen the time window (default 7 days)
    python3 dedupe.py --show-clusters list what has already been grouped
    python3 dedupe.py --undo          reset all duplicate marks

Primary selection, in order:
  1. Already published (never demote something the analysts have seen)
  2. A company's own press release beats third-party coverage
  3. Longer article text
  4. Seen first
"""

import argparse
import os
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "articles.db")

WINDOW_DAYS = 7
TITLE_THRESHOLD = 0.60
TEXT_THRESHOLD = 0.50
TEXT_SAMPLE_CHARS = 1500

EXTRA_COLUMNS = [
    ("dup_of", "TEXT"),        # url of the primary article
    ("dup_score", "REAL"),     # similarity that triggered the match
    ("dup_checked", "TEXT"),   # timestamp, so we don't re-compare endlessly
]

# Words that carry no signal for matching headlines.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "has",
    "have", "had", "will", "would", "its", "it", "that", "this", "new", "says",
    "said", "after", "over", "into", "out", "up", "down", "first",
}

CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def normalise_text(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s):
    """Word tokens for space-delimited languages, character bigrams for CJK.

    Korean, Japanese and Chinese do not separate words with spaces, so naive
    whitespace tokenisation gives one giant token and every comparison fails.
    """
    s = normalise_text(s)
    if not s:
        return set()
    if CJK.search(s):
        compact = re.sub(r"\s+", "", s)
        return {compact[i:i + 2] for i in range(len(compact) - 1)} or {compact}
    return {w for w in s.split() if w not in STOPWORDS and len(w) > 2}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def containment(a, b):
    """How much of the smaller set sits inside the larger.

    Catches the case where a trade outlet reprints a press release verbatim but
    adds three paragraphs of context - Jaccard punishes the length difference,
    containment does not.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def is_cjk(*strings):
    return any(CJK.search(s or "") for s in strings)


def similarity(a, b):
    """Return (score, basis) for two article rows.

    Thresholds differ by script. CJK text is compared using character bigrams
    rather than words, and bigram overlap runs systematically lower than word
    overlap for the same degree of similarity - so applying the English
    thresholds to Korean or Chinese misses genuine duplicates.
    """
    cjk = is_cjk(a["title"], b["title"])
    title_min = 0.40 if cjk else TITLE_THRESHOLD
    text_min = 0.35 if cjk else TEXT_THRESHOLD
    contain_min = 0.60 if cjk else 0.80

    ta, tb = tokens(a["title"]), tokens(b["title"])
    title_j = jaccard(ta, tb)
    title_c = containment(ta, tb)

    xa = tokens((a["text"] or "")[:TEXT_SAMPLE_CHARS])
    xb = tokens((b["text"] or "")[:TEXT_SAMPLE_CHARS])
    text_j = jaccard(xa, xb)
    text_c = containment(xa, xb)

    if title_j >= title_min:
        return title_j, "title"
    if title_c >= contain_min and text_j >= (0.20 if cjk else 0.30):
        return title_c, "title-contained"
    if text_j >= text_min:
        return text_j, "body"
    if cjk and text_c >= 0.55 and title_c >= 0.45:
        return text_c, "body-contained"
    return 0.0, ""


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

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


def parse_when(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def rank(row):
    """Lower is better. Decides which article of a cluster survives."""
    published = 0 if (row["published_at"] or "") else 1
    own_site = 0 if (row["source_kind"] or "") == "scrape" else 1
    length = -(row["text_chars"] or 0)
    seen = row["first_seen"] or ""
    return (published, own_site, length, seen)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(window_days=WINDOW_DAYS, dry_run=False, quiet=False):
    conn = connect()
    migrate(conn)

    rows = conn.execute(
        "SELECT url, title, source, source_kind, first_seen, text, text_chars,"
        "       status, published_at, dup_of, scored_at"
        "  FROM articles"
        " WHERE status IN ('extracted', 'duplicate')"
        "   AND text IS NOT NULL AND text != ''"
        " ORDER BY first_seen"
    ).fetchall()

    if not rows:
        print("nothing to compare", file=sys.stderr)
        return

    print(f"comparing {len(rows)} article(s), +/- {window_days} day window\n",
          file=sys.stderr)

    # Bucket by day so we only compare articles that are near each other in
    # time. Comparing everything against everything is O(n^2) and pointless -
    # two articles a month apart are not the same story.
    by_day = defaultdict(list)
    for r in rows:
        when = parse_when(r["first_seen"])
        key = when.date() if when else None
        by_day[key].append(r)

    # union-find over article urls
    parent = {r["url"]: r["url"] for r in rows}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pair_basis = {}
    comparisons = 0

    for r in rows:
        when = parse_when(r["first_seen"])
        if when is None:
            continue
        neighbours = []
        for offset in range(-window_days, window_days + 1):
            neighbours.extend(by_day.get((when + timedelta(days=offset)).date(), []))

        for other in neighbours:
            if other["url"] <= r["url"]:      # each pair once
                continue
            comparisons += 1
            score, basis = similarity(r, other)
            if score:
                union(r["url"], other["url"])
                pair_basis[(r["url"], other["url"])] = (score, basis)

    clusters = defaultdict(list)
    for r in rows:
        clusters[find(r["url"])].append(r)

    multi = {k: v for k, v in clusters.items() if len(v) > 1}
    print(f"{comparisons} comparisons -> {len(multi)} cluster(s) "
          f"covering {sum(len(v) for v in multi.values())} articles\n",
          file=sys.stderr)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    marked = 0

    for members in multi.values():
        members = sorted(members, key=rank)
        primary, dupes = members[0], members[1:]

        if not quiet:
            print(f"PRIMARY  [{primary['source'][:28]}] {(primary['title'] or '')[:70]}",
                  file=sys.stderr)
            for d in dupes:
                pair = pair_basis.get((primary["url"], d["url"])) or \
                       pair_basis.get((d["url"], primary["url"])) or (0.0, "chain")
                print(f"   dup   [{d['source'][:28]}] {(d['title'] or '')[:64]}"
                      f"   ({pair[1]} {pair[0]:.2f})", file=sys.stderr)
            print("", file=sys.stderr)

        if dry_run:
            marked += len(dupes)
            continue

        for d in dupes:
            if d["published_at"]:
                continue          # already in SharePoint, leave it alone
            conn.execute(
                "UPDATE articles SET status = 'duplicate', dup_of = ?,"
                " dup_score = ?, dup_checked = ? WHERE url = ?",
                (primary["url"],
                 (pair_basis.get((primary["url"], d["url"])) or (0.0,))[0],
                 now, d["url"]))
            marked += 1

        conn.execute(
            "UPDATE articles SET status = 'extracted', dup_of = NULL,"
            " dup_checked = ? WHERE url = ? AND status = 'duplicate'",
            (now, primary["url"]))

    if not dry_run:
        conn.execute(
            "UPDATE articles SET dup_checked = ?"
            " WHERE status = 'extracted' AND (dup_checked IS NULL OR dup_checked = '')",
            (now,))
        conn.commit()

    print(f"{marked} article(s) marked as duplicates"
          f"{' (dry run - nothing changed)' if dry_run else ''}", file=sys.stderr)
    conn.close()


def show_clusters():
    conn = connect()
    migrate(conn)
    rows = conn.execute(
        "SELECT d.url, d.title, d.source, d.dup_score,"
        "       p.title AS primary_title, p.source AS primary_source"
        "  FROM articles d JOIN articles p ON d.dup_of = p.url"
        " WHERE d.dup_of IS NOT NULL AND d.dup_of != ''"
        " ORDER BY p.url, d.dup_score DESC").fetchall()
    if not rows:
        print("no duplicates recorded")
        return
    last = None
    for r in rows:
        if r["primary_title"] != last:
            print(f"\nPRIMARY [{r['primary_source'][:30]}] {r['primary_title'][:76]}")
            last = r["primary_title"]
        print(f"   dup  [{r['source'][:30]}] {(r['title'] or '')[:70]}"
              f"  ({r['dup_score'] or 0:.2f})")
    conn.close()


def undo():
    conn = connect()
    migrate(conn)
    n = conn.execute("SELECT COUNT(*) FROM articles WHERE status='duplicate'").fetchone()[0]
    conn.execute("UPDATE articles SET status='extracted', dup_of=NULL,"
                 " dup_score=NULL, dup_checked=NULL WHERE status='duplicate'")
    conn.execute("UPDATE articles SET dup_checked=NULL")
    conn.commit()
    print(f"reset {n} duplicate(s) back to 'extracted'", file=sys.stderr)
    conn.close()


def main():
    ap = argparse.ArgumentParser(description="MI pipeline - near-duplicate detection")
    ap.add_argument("--window", type=int, default=WINDOW_DAYS,
                    help=f"days either side to compare (default {WINDOW_DAYS})")
    ap.add_argument("--dry-run", action="store_true",
                    help="show clusters without changing anything")
    ap.add_argument("--show-clusters", action="store_true",
                    help="list duplicates already recorded")
    ap.add_argument("--undo", action="store_true",
                    help="clear all duplicate marks")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    if args.show_clusters:
        show_clusters()
    elif args.undo:
        undo()
    else:
        run(window_days=args.window, dry_run=args.dry_run, quiet=args.quiet)


if __name__ == "__main__":
    main()

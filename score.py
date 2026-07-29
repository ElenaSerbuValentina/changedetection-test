#!/usr/bin/env python3
"""
Market Intelligence pipeline - relevance scoring, categorisation, summary.

Reads articles with status='extracted', sends each to an LLM, and stores a
relevance verdict, category, summary and companies list.

Works with ANY OpenAI-compatible endpoint. Configure with environment variables:

  Docker Model Runner (local, free):
      export MI_API_BASE="http://localhost:12434/engines/v1"
      export MI_MODEL="ai/qwen2.5:3B-F16"
      export MI_API_KEY="not-needed"

  Azure OpenAI:
      export MI_API_BASE="https://<resource>.openai.azure.com/openai/v1"
      export MI_MODEL="<deployment-name>"
      export MI_API_KEY="<key>"

Usage:
    python3 score.py --limit 5          try five first, print results
    python3 score.py --dry-run --limit 3    call the model, store nothing
    python3 score.py                    score everything pending
    python3 score.py --rescore          re-score everything (after prompt edits)
    python3 score.py --show <url>       print one article's verdict
    python3 score.py --report           summary of what has been scored

The prompt lives in relevance_prompt.txt so it can be tuned without touching
this file. Every verdict records which prompt version produced it, so you can
tell old results from new ones.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "articles.db")
PROMPT_FILE = os.path.join(HERE, "relevance_prompt.txt")

API_BASE = os.environ.get("MI_API_BASE", "http://localhost:12434/engines/v1")
API_KEY = os.environ.get("MI_API_KEY", "not-needed")
MODEL = os.environ.get("MI_MODEL", "ai/qwen2.5:3B-F16")

# Local models have modest context windows. Truncating the article body keeps
# requests inside them; the opening of a press release carries the substance.
MAX_ARTICLE_CHARS = int(os.environ.get("MI_MAX_CHARS", "6000"))
REQUEST_TIMEOUT = int(os.environ.get("MI_TIMEOUT", "180"))
RETRIES = 2

CATEGORIES = {
    "Project Awards", "New Factory / Factory Expansions", "M&A / JV / Divestment",
    "R&D / Product Management", "CLV / Installation News", "Legal / Security",
    "Sustainability / HSE / ESG", "Investor Relations / Finance",
    "TSO / Developers", "Policy / Govt", "Market Reports", "Customer news",
    "Sustainability", "Vessels", "Products",
}

RELEVANCE_LEVELS = {"Relevant", "Potentially relevant", "Not relevant"}
PRIORITIES = {"High", "Medium", "Low", "None"}
SIGNALS = {"Positive", "Negative", "Mixed", "Neutral", "Unknown"}
INFO_STATUS = {"Confirmed", "Preliminary", "Speculative", "Rumour", "Unknown"}
ACTIONS = {"Escalate", "Include in daily digest", "Include in weekly digest",
           "Analyst review", "Archive", "Discard"}

EXTRA_COLUMNS = [
    ("relevance", "TEXT"),            # Relevant / Potentially relevant / Not relevant
    ("relevant", "INTEGER"),          # derived 0/1, for easy filtering
    ("score", "INTEGER"),
    ("priority", "TEXT"),
    ("category", "TEXT"),             # primary category
    ("categories", "TEXT"),           # up to three, comma separated
    ("companies", "TEXT"),
    ("competitors", "TEXT"),
    ("projects", "TEXT"),
    ("regions", "TEXT"),
    ("market_signal", "TEXT"),
    ("information_status", "TEXT"),
    ("recommended_action", "TEXT"),
    ("summary", "TEXT"),
    ("reason", "TEXT"),               # decision_reason
    ("nkt_implication", "TEXT"),
    ("scored_at", "TEXT"),
    ("scorer", "TEXT"),
    ("prompt_version", "TEXT"),
    ("score_error", "TEXT"),
]


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


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def load_prompt():
    if not os.path.exists(PROMPT_FILE):
        sys.exit(f"missing {PROMPT_FILE}")
    text = open(PROMPT_FILE, encoding="utf-8").read()
    version = hashlib.sha256(text.encode()).hexdigest()[:8]
    return text, version


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def call_model(system_prompt, user_content):
    """POST to an OpenAI-compatible /chat/completions. Returns text or raises."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": 700,
    }
    req = urllib.request.Request(
        f"{API_BASE.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "api-key": API_KEY,          # Azure OpenAI uses this header name
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_verdict(raw):
    """Pull a JSON object out of the model's reply. Small models add chatter."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def _pick(value, allowed, default):
    v = (value or "").strip()
    return v if v in allowed else default


def _joinlist(value, limit=500):
    if value is None:
        return ""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    else:
        parts = [str(p).strip() for p in value]
    return ", ".join(p for p in parts if p)[:limit]


def normalise(v):
    """Coerce whatever the model returned into the shape the database expects.

    Small models drift: they invent categories, return "yes" instead of true,
    and sometimes wrap numbers in strings. Anything unrecognised falls back to
    a safe default rather than failing the whole article.
    """
    score = v.get("relevance_score", v.get("score", 0))
    try:
        score = max(0, min(100, int(float(score))))
    except (TypeError, ValueError):
        score = 0

    relevance = _pick(v.get("relevance"), RELEVANCE_LEVELS, "")
    if not relevance:
        relevance = ("Relevant" if score >= 60
                     else "Potentially relevant" if score >= 40
                     else "Not relevant")

    not_relevant = relevance == "Not relevant"

    primary = (v.get("primary_category") or "").strip()
    if primary not in CATEGORIES or not_relevant:
        primary = "" if not_relevant else "Other"
        if primary == "Other":
            # 'Other' is not in the approved taxonomy - flag rather than invent
            primary = ""

    cats = v.get("categories") or []
    if isinstance(cats, str):
        cats = [c.strip() for c in cats.split(",")]
    cats = [c for c in (str(x).strip() for x in cats) if c in CATEGORIES][:3]
    if not_relevant:
        cats = []
    if primary and primary not in cats:
        cats.insert(0, primary)
        cats = cats[:3]

    return {
        "relevance": relevance,
        "relevant": 0 if not_relevant else 1,
        "score": score,
        "priority": _pick(v.get("priority"), PRIORITIES,
                          "None" if not_relevant else "Low"),
        "category": primary,
        "categories": ", ".join(cats),
        "companies": _joinlist(v.get("companies")),
        "competitors": _joinlist(v.get("competitors")),
        "projects": _joinlist(v.get("projects")),
        "regions": _joinlist(v.get("countries_or_regions")),
        "market_signal": _pick(v.get("market_signal"), SIGNALS, "Unknown"),
        "information_status": _pick(v.get("information_status"), INFO_STATUS,
                                    "Unknown"),
        "recommended_action": _pick(v.get("recommended_action"), ACTIONS,
                                    "Discard" if not_relevant else "Analyst review"),
        "summary": (v.get("summary") or "").strip()[:2000],
        "reason": (v.get("decision_reason") or v.get("reason") or "").strip()[:500],
        "nkt_implication": (v.get("nkt_implication") or "").strip()[:1000],
    }


def score_article(row, system_prompt):
    body = (row["text"] or "")[:MAX_ARTICLE_CHARS]
    user = (
        f"SOURCE: {row['source']}\n"
        f"HEADLINE: {row['title']}\n"
        f"PUBLISHED: {row['published_raw']}\n"
        f"LANGUAGE: {row['language'] or 'unknown'}\n\n"
        f"ARTICLE:\n{body}"
    )

    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            raw = call_model(system_prompt, user)
            return normalise(parse_verdict(raw)), None
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            last_err = f"HTTP {e.code}: {detail}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < RETRIES:
            time.sleep(2 * (attempt + 1))
    return None, last_err


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def run(limit=None, rescore=False, dry_run=False, quiet=False):
    system_prompt, version = load_prompt()
    conn = connect()
    migrate(conn)

    where = "status = 'extracted'"
    if not rescore:
        where += " AND (scored_at IS NULL OR scored_at = '')"
    sql = (f"SELECT url, title, source, published_raw, language, text"
           f" FROM articles WHERE {where} AND text IS NOT NULL AND text != ''"
           f" ORDER BY first_seen DESC")
    params = []
    if limit:
        sql += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("nothing to score", file=sys.stderr)
        return

    print(f"model    : {MODEL}\nendpoint : {API_BASE}\n"
          f"prompt   : {version}\narticles : {len(rows)}\n", file=sys.stderr)

    tally = Counter()
    started = time.monotonic()

    for i, row in enumerate(rows, 1):
        verdict, err = score_article(row, system_prompt)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if err:
            tally["error"] += 1
            if not dry_run:
                conn.execute(
                    "UPDATE articles SET score_error = ?, scored_at = ?,"
                    " scorer = ?, prompt_version = ? WHERE url = ?",
                    (err, now, MODEL, version, row["url"]))
                conn.commit()
            print(f"! [{i}/{len(rows)}] {(row['title'] or '')[:50]:<52} {err[:80]}",
                  file=sys.stderr)
            continue

        tally[verdict["relevance"]] += 1
        if verdict["category"]:
            tally[verdict["category"]] += 1

        if not dry_run:
            verdict.update({"scored_at": now, "scorer": MODEL,
                            "prompt_version": version, "score_error": ""})
            sets = ", ".join(f"{k} = ?" for k in verdict)
            conn.execute(f"UPDATE articles SET {sets} WHERE url = ?",
                         list(verdict.values()) + [row["url"]])
            conn.commit()

        if not quiet:
            mark = {"Relevant": "*", "Potentially relevant": "~",
                    "Not relevant": " "}[verdict["relevance"]]
            print(f"{mark} [{i}/{len(rows)}] {verdict['score']:>3}"
                  f" {verdict['priority']:<7}"
                  f" {(verdict['category'] or '-')[:26]:<28}"
                  f" {(row['title'] or '')[:40]}", file=sys.stderr)
            if dry_run:
                print(f"         reason: {verdict['reason'][:170]}", file=sys.stderr)
                if verdict["summary"]:
                    print(f"         {verdict['summary'][:200]}", file=sys.stderr)
                if verdict["competitors"]:
                    print(f"         competitors: {verdict['competitors'][:100]}",
                          file=sys.stderr)
                print("", file=sys.stderr)

    elapsed = time.monotonic() - started
    per = elapsed / max(len(rows), 1)
    print(f"\n{len(rows)} scored in {elapsed/60:.1f} min ({per:.1f}s each)",
          file=sys.stderr)
    print(f"Relevant: {tally['Relevant']} | Potentially: "
          f"{tally['Potentially relevant']} | Not relevant: "
          f"{tally['Not relevant']} | errors: {tally['error']}", file=sys.stderr)
    if dry_run:
        print("(dry run - nothing stored)", file=sys.stderr)
    conn.close()


def report():
    conn = connect()
    migrate(conn)

    print("=== relevance ===")
    for r in conn.execute(
            "SELECT relevance, COUNT(*) n, ROUND(AVG(score),1) avg_score"
            " FROM articles WHERE scored_at IS NOT NULL AND scored_at != ''"
            " GROUP BY relevance ORDER BY avg_score DESC"):
        print(f"  {(r['relevance'] or '?'):<22} {r['n']:>5}   avg score {r['avg_score']}")

    print("\n=== priority ===")
    for r in conn.execute(
            "SELECT priority, COUNT(*) n FROM articles"
            " WHERE relevant = 1 GROUP BY priority ORDER BY n DESC"):
        print(f"  {(r['priority'] or '?'):<22} {r['n']:>5}")

    print("\n=== categories (relevant only) ===")
    for r in conn.execute(
            "SELECT category, COUNT(*) n FROM articles"
            " WHERE relevant = 1 GROUP BY category ORDER BY n DESC"):
        print(f"  {r['category']:<34} {r['n']:>5}")

    print("\n=== by source (relevant / total scored) ===")
    for r in conn.execute(
            "SELECT source, SUM(COALESCE(relevant,0)) rel, COUNT(*) n"
            " FROM articles WHERE scored_at IS NOT NULL AND scored_at != ''"
            " GROUP BY source ORDER BY rel DESC LIMIT 20"):
        print(f"  {r['source'][:40]:<42} {r['rel']:>4} / {r['n']}")

    errs = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE score_error IS NOT NULL"
        " AND score_error != ''").fetchone()[0]
    if errs:
        print(f"\n{errs} article(s) failed scoring - see score_error column")
    conn.close()


def show(needle):
    conn = connect()
    row = conn.execute(
        "SELECT * FROM articles WHERE url = ? OR url LIKE ? OR title LIKE ?",
        (needle, f"%{needle}%", f"%{needle}%")).fetchone()
    if not row:
        sys.exit(f"no article matching {needle!r}")
    for k in ("source", "title", "url", "language", "published_raw", "status",
              "relevance", "score", "priority", "category", "categories",
              "companies", "competitors", "projects", "regions",
              "market_signal", "information_status", "recommended_action",
              "scorer", "prompt_version", "score_error"):
        if k in row.keys():
            print(f"{k:>18}: {row[k]}")
    print(f"\n{'summary':>18}: {row['summary'] or '(none)'}")
    print(f"{'reason':>18}: {row['reason'] or '(none)'}")
    print(f"{'nkt_implication':>18}: {row['nkt_implication'] or '(none)'}")
    conn.close()


def main():
    ap = argparse.ArgumentParser(description="MI pipeline - relevance scoring")
    ap.add_argument("--limit", type=int, help="score at most N articles")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score articles that already have a verdict")
    ap.add_argument("--dry-run", action="store_true",
                    help="call the model and print, but store nothing")
    ap.add_argument("--show", metavar="URL_OR_TITLE", help="print one verdict")
    ap.add_argument("--report", action="store_true", help="summarise results")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    if args.show:
        show(args.show)
    elif args.report:
        report()
    else:
        run(limit=args.limit, rescore=args.rescore, dry_run=args.dry_run,
            quiet=args.quiet)


if __name__ == "__main__":
    main()

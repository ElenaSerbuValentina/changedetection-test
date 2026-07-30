#!/usr/bin/env python3
"""
Market Intelligence pipeline - orchestrator.

Runs the five pipeline stages in order. Each stage remains a standalone script
you can still run by hand; this just sequences them, checks the environment
before starting, and reports what happened.

    STAGES
      1. ingest    find new articles on listing pages and RSS feeds
      2. extract   fetch each new article and pull out clean text
      3. dedupe    collapse near-duplicate stories to one primary
      4. score     relevance, category and summary via an LLM
      5. publish   POST results to Power Automate -> SharePoint

    TYPICAL USE

      # everything, but only score 30 articles this run
      python3 run.py --score-limit 30

      # just one stage
      python3 run.py --only score --score-limit 10

      # everything except publishing (safe while testing)
      python3 run.py --skip publish

      # a range of stages
      python3 run.py --from extract --to score

      # see what would run, without running it
      python3 run.py --plan

      # start over: wipe the database and rebuild the baseline
      python3 run.py --reset --yes

    ENVIRONMENT

    Read from the shell, or from a .env file in this folder if present.

      MI_API_BASE     LLM endpoint      e.g. http://localhost:12434/engines/v1
      MI_MODEL        model id          e.g. ai/gemma3:4B-Q4_K_M
      MI_API_KEY      api key           'not-needed' for a local model
      MI_WEBHOOK_URL  Power Automate HTTP trigger URL   (publish stage only)

    Only the stages you actually run need their variables, so you can develop
    the ingest and extract stages without an LLM configured at all.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable          # use the same interpreter that started us,
                                 # so the venv is inherited automatically
DB = os.path.join(HERE, "articles.db")
ENV_FILE = os.path.join(HERE, ".env")


# ---------------------------------------------------------------------------
# Stage definitions
#
# Each stage declares:
#   script    the file to run
#   needs     environment variables that must be set before it can run
#   describe  one line shown in --plan and in the run log
# Arguments are built per-stage in build_args() below, because they depend on
# the command-line options.
# ---------------------------------------------------------------------------

STAGES = [
    {
        "name": "ingest",
        "script": "ingest.py",
        "needs": [],
        "describe": "find new articles on listing pages and RSS feeds",
    },
    {
        "name": "extract",
        "script": "extract.py",
        "needs": [],
        "describe": "fetch each new article and extract clean text",
    },
    {
        "name": "dedupe",
        "script": "dedupe.py",
        "needs": [],
        "describe": "collapse near-duplicate stories to one primary",
    },
    {
        "name": "score",
        "script": "score.py",
        "needs": ["MI_API_BASE", "MI_MODEL"],
        "describe": "relevance, category and summary via the LLM",
    },
    {
        "name": "publish",
        "script": "publish.py",
        "needs": ["MI_WEBHOOK_URL"],
        "describe": "POST results to Power Automate",
    },
]

STAGE_NAMES = [s["name"] for s in STAGES]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env_file(path=ENV_FILE):
    """Load KEY=value lines from .env, without overriding the real environment.

    Shell variables win, so you can override a single setting for one run:
        MI_MODEL=something-else python3 run.py --only score
    """
    if not os.path.exists(path):
        return []
    loaded = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def check_env(stages):
    """Fail before doing any work, not twenty minutes in."""
    missing = {}
    for stage in stages:
        for var in stage["needs"]:
            if not os.environ.get(var):
                missing.setdefault(var, []).append(stage["name"])
    if missing:
        print("Missing environment variables:\n", file=sys.stderr)
        for var, used_by in missing.items():
            print(f"  {var:<16} needed by: {', '.join(used_by)}", file=sys.stderr)
        print("\nSet them in the shell, or put them in .env in this folder.",
              file=sys.stderr)
        print("Or skip the stages that need them, e.g. --skip publish",
              file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument construction per stage
# ---------------------------------------------------------------------------

def build_args(stage_name, opts):
    """Translate orchestrator options into that stage's own CLI flags."""
    args = []

    if stage_name == "ingest":
        if opts.backfill:
            args.append("--backfill")
        if opts.dry_run:
            args.append("--dry-run")
        args.append("--quiet")

    elif stage_name == "extract":
        if opts.extract_limit:
            args += ["--limit", str(opts.extract_limit)]
        if opts.retry_failed:
            args.append("--retry")
        args.append("--quiet")

    elif stage_name == "dedupe":
        if opts.dry_run:
            args.append("--dry-run")
        args.append("--quiet")

    elif stage_name == "score":
        if opts.score_limit:
            args += ["--limit", str(opts.score_limit)]
        if opts.score_source:
            args += ["--source", opts.score_source]
        if opts.rescore:
            args.append("--rescore")
        if opts.dry_run:
            args.append("--dry-run")
        args.append("--quiet")

    elif stage_name == "publish":
        if opts.publish_limit:
            args += ["--limit", str(opts.publish_limit)]
        if opts.min_score is not None:
            args += ["--min-score", str(opts.min_score)]
        if opts.relevant_only:
            args.append("--relevant-only")
        if opts.dry_run:
            args.append("--dry-run")

    return args


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def select_stages(opts):
    """Work out which stages to run, from --only / --from / --to / --skip."""
    if opts.only:
        chosen = [s for s in STAGES if s["name"] in opts.only]
    else:
        start = STAGE_NAMES.index(opts.from_stage) if opts.from_stage else 0
        end = STAGE_NAMES.index(opts.to_stage) if opts.to_stage else len(STAGES) - 1
        if start > end:
            sys.exit(f"--from {opts.from_stage} comes after --to {opts.to_stage}")
        chosen = STAGES[start:end + 1]

    if opts.skip:
        chosen = [s for s in chosen if s["name"] not in opts.skip]
    if not chosen:
        sys.exit("no stages selected")
    return chosen


def run_stage(stage, opts):
    """Run one stage as a subprocess. Returns (exit_code, seconds)."""
    script = os.path.join(HERE, stage["script"])
    if not os.path.exists(script):
        print(f"  missing script: {script}", file=sys.stderr)
        return 127, 0.0

    cmd = [PYTHON, script] + build_args(stage["name"], opts)
    print(f"\n{'=' * 72}\n{stage['name'].upper()}  -  {stage['describe']}\n"
          f"$ {' '.join(cmd[1:])}\n{'=' * 72}", flush=True)

    started = time.monotonic()
    # stdout of a stage is its data (JSON); stderr is its progress log.
    # Let both flow through to the caller so redirecting run.py works naturally.
    result = subprocess.run(cmd, cwd=HERE)
    return result.returncode, time.monotonic() - started


def reset_database(confirmed):
    """Delete the database so the next run starts from nothing."""
    files = [DB, DB + "-wal", DB + "-shm"]
    present = [f for f in files if os.path.exists(f)]
    if not present:
        print("no database to remove")
        return
    if not confirmed:
        print("This will delete:", file=sys.stderr)
        for f in present:
            print(f"  {f}", file=sys.stderr)
        print("\nEverything ingested, extracted, scored and published is lost.\n"
              "Re-run with --yes to confirm.", file=sys.stderr)
        sys.exit(1)
    for f in present:
        os.remove(f)
        print(f"removed {f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Market Intelligence pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages, in order: " + " -> ".join(STAGE_NAMES))

    # which stages
    ap.add_argument("--only", nargs="+", choices=STAGE_NAMES, metavar="STAGE",
                    help="run only these stages")
    ap.add_argument("--from", dest="from_stage", choices=STAGE_NAMES,
                    metavar="STAGE", help="start at this stage")
    ap.add_argument("--to", dest="to_stage", choices=STAGE_NAMES,
                    metavar="STAGE", help="stop after this stage")
    ap.add_argument("--skip", nargs="+", choices=STAGE_NAMES, metavar="STAGE",
                    help="skip these stages")

    # per-stage limits
    ap.add_argument("--extract-limit", type=int, metavar="N",
                    help="extract at most N articles")
    ap.add_argument("--score-limit", type=int, metavar="N",
                    help="score at most N articles")
    ap.add_argument("--publish-limit", type=int, metavar="N",
                    help="publish at most N articles")

    # stage behaviour
    ap.add_argument("--backfill", action="store_true",
                    help="ingest: mark everything currently listed as already "
                         "seen (use once when adding a source)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="extract: re-attempt articles that failed before")
    ap.add_argument("--rescore", action="store_true",
                    help="score: re-score articles that already have a verdict "
                         "(use after editing relevance_prompt.txt)")
    ap.add_argument("--score-source", metavar="SUBSTRING",
                    help="score: only sources matching this, e.g. 'Alert'")
    ap.add_argument("--min-score", type=int, metavar="N",
                    help="publish: only articles scoring N or above")
    ap.add_argument("--relevant-only", action="store_true",
                    help="publish: skip articles judged Not relevant")

    # global
    ap.add_argument("--dry-run", action="store_true",
                    help="pass --dry-run to every stage that supports it: "
                         "nothing is written, nothing is sent")
    ap.add_argument("--plan", action="store_true",
                    help="show what would run, then exit")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="carry on to the next stage even if one fails")
    ap.add_argument("--reset", action="store_true",
                    help="delete the database before running")
    ap.add_argument("--yes", action="store_true",
                    help="confirm --reset without prompting")

    opts = ap.parse_args()

    loaded = load_env_file()
    if loaded:
        print(f"loaded from .env: {', '.join(loaded)}", file=sys.stderr)

    if opts.reset:
        reset_database(opts.yes)

    stages = select_stages(opts)

    # --plan shows the sequence without running anything, so you can check the
    # flags do what you expect before a long run.
    if opts.plan:
        print(f"\nwould run {len(stages)} stage(s):\n")
        for i, s in enumerate(stages, 1):
            args = " ".join(build_args(s["name"], opts))
            print(f"  {i}. {s['name']:<9} {s['script']} {args}")
            print(f"     {s['describe']}")
        missing = [v for s in stages for v in s["needs"] if not os.environ.get(v)]
        if missing:
            print(f"\n  NOT SET: {', '.join(sorted(set(missing)))}")
        print()
        return

    check_env(stages)

    started_at = datetime.now(timezone.utc)
    print(f"pipeline start {started_at.isoformat(timespec='seconds')}"
          f"   stages: {' -> '.join(s['name'] for s in stages)}"
          f"{'   [DRY RUN]' if opts.dry_run else ''}", file=sys.stderr)

    results = []
    for stage in stages:
        code, seconds = run_stage(stage, opts)
        results.append((stage["name"], code, seconds))
        if code != 0 and not opts.continue_on_error:
            print(f"\n{stage['name']} failed (exit {code}) - stopping.\n"
                  f"Use --continue-on-error to carry on regardless.",
                  file=sys.stderr)
            break

    # summary
    total = sum(r[2] for r in results)
    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}", file=sys.stderr)
    for name, code, seconds in results:
        status = "ok" if code == 0 else f"FAILED ({code})"
        print(f"  {name:<10} {status:<14} {seconds/60:6.1f} min", file=sys.stderr)
    print(f"  {'total':<10} {'':<14} {total/60:6.1f} min", file=sys.stderr)

    sys.exit(1 if any(c != 0 for _, c, _ in results) else 0)


if __name__ == "__main__":
    main()

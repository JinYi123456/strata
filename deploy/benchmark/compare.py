"""
Compare two benchmark runs and emit a markdown table.

    python compare.py baseline.json graviton.json

The baseline is the left-hand side; every ratio is target / baseline, so a
speed-up is > 1.0 and a regression is < 1.0. Both directions are labelled
explicitly rather than implying that bigger is always better.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(a: float | None, b: float | None) -> float | None:
    if not a or not b:
        return None
    return b / a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("target", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    base, targ = load(args.baseline), load(args.target)
    bh, th = base["host"], targ["host"]

    lines: list[str] = []
    add = lines.append

    add("| | baseline | target |")
    add("|---|---|---|")
    add(f"| architecture | {bh['machine']} | {th['machine']} |")
    add(f"| processor | {bh['processor'] or 'unknown'} | {th['processor'] or 'unknown'} |")
    add(f"| cpus | {bh['cpus']} | {th['cpus']} |")
    add(f"| OpenCV | {bh['opencv']} ({bh['opencv_build']}) | {th['opencv']} ({th['opencv_build']}) |")
    add("")

    add("### Operations (median ms, lower is better)")
    add("")
    add("| operation | baseline | target | target/baseline |")
    add("|---|---:|---:|---:|")
    for name, b in base["microbenchmarks"].items():
        t = targ["microbenchmarks"].get(name)
        if not t:
            continue
        r = ratio(b["median_ms"], t["median_ms"])
        verdict = f"{r:.2f}x" if r else "-"
        add(f"| {name} | {b['median_ms']:.2f} | {t['median_ms']:.2f} | {verdict} |")

    add("")
    add("### Full scan")
    add("")
    bs, ts = base["full_scan"], targ["full_scan"]
    add("| metric | baseline | target | target/baseline |")
    add("|---|---:|---:|---:|")
    r = ratio(bs["median_wall_ms"], ts["median_wall_ms"])
    add(f"| median wall (ms) | {bs['median_wall_ms']:.1f} | {ts['median_wall_ms']:.1f} | {(f'{r:.2f}x' if r else '-')} |")
    r = ratio(bs["p95_wall_ms"], ts["p95_wall_ms"])
    add(f"| p95 wall (ms) | {bs['p95_wall_ms']:.1f} | {ts['p95_wall_ms']:.1f} | {(f'{r:.2f}x' if r else '-')} |")
    r = ratio(bs["scans_per_hour"], ts["scans_per_hour"])
    add(f"| scans/hour | {bs['scans_per_hour']:.1f} | {ts['scans_per_hour']:.1f} | {(f'{r:.2f}x' if r else '-')} |")
    add(f"| backend | {bs['backend_resolved']} | {ts['backend_resolved']} | |")

    add("")
    add("### Stage breakdown (median ms)")
    add("")
    add("| stage | baseline | target | target/baseline |")
    add("|---|---:|---:|---:|")
    for stage, b in bs["stage_median_ms"].items():
        t = ts["stage_median_ms"].get(stage)
        if t is None:
            continue
        r = ratio(b, t)
        add(f"| {stage} | {b:.1f} | {t:.1f} | {(f'{r:.2f}x' if r else '-')} |")

    bc, tc = base.get("cost") or {}, targ.get("cost") or {}
    if bc.get("usd_per_1000_scans") and tc.get("usd_per_1000_scans"):
        add("")
        add("### Cost")
        add("")
        add(f"- baseline ({bc['instance']}): ${bc['usd_per_1000_scans']:.3f} per 1000 scans")
        add(f"- target ({tc['instance']}): ${tc['usd_per_1000_scans']:.3f} per 1000 scans")

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

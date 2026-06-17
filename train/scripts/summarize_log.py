#!/usr/bin/env python3
import json
import re
import statistics
import sys
from pathlib import Path


STEP_RE = re.compile(r"\bstep\s+(?P<step>\d+)\b", re.IGNORECASE)
FIELD_RES = {
    "loss": re.compile(r"\bloss\s+([-+0-9.eE]+)", re.IGNORECASE),
    "lr": re.compile(r"\blr\s+([-+0-9.eE]+)", re.IGNORECASE),
    "mem_gib": re.compile(r"\bmem\s+([-+0-9.eE]+)\s*GiB", re.IGNORECASE),
    "tps_total": re.compile(r"\btps\s+([-+0-9.eE]+)\s*\([-+0-9.eE]+/gpu\)", re.IGNORECASE),
    "tps_per_gpu": re.compile(r"\btps\s+[-+0-9.eE]+\s*\(([-+0-9.eE]+)/gpu\)|\btps/gpu\s+([-+0-9.eE]+)", re.IGNORECASE),
}


def f(value):
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: summarize_log.py <train.log> <summary.json> [warmup_steps]", file=sys.stderr)
        return 2
    log_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    warmup_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    records = []
    seen = set()
    for line in log_path.read_text(errors="replace").splitlines():
        if "[val]" in line:
            continue
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group("step"))
        key = (step, line)
        if key in seen:
            continue
        seen.add(key)
        fields = {}
        for name, regex in FIELD_RES.items():
            field_match = regex.search(line)
            fields[name] = f(next((g for g in field_match.groups() if g is not None), None)) if field_match else None
        rec = {
            "step": step,
            "loss": fields["loss"],
            "lr": fields["lr"],
            "mem_gib": fields["mem_gib"],
            "tps_total": fields["tps_total"],
            "tps_per_gpu": fields["tps_per_gpu"],
            "line": line[-1000:],
        }
        records.append(rec)

    steady = [r for r in records if r["step"] >= warmup_steps and r["tps_per_gpu"] is not None]
    tps = [r["tps_per_gpu"] for r in steady]
    tps_total = [r["tps_total"] for r in steady if r["tps_total"] is not None]
    mem = [r["mem_gib"] for r in records if r["mem_gib"] is not None]
    losses = [r["loss"] for r in records if r["loss"] is not None]

    summary = {
        "log": str(log_path),
        "parsed_step_records": len(records),
        "warmup_steps_discarded": warmup_steps,
        "first_step": records[0]["step"] if records else None,
        "last_step": records[-1]["step"] if records else None,
        "steady_tps_per_gpu_count": len(tps),
        "steady_tps_per_gpu_mean": statistics.fmean(tps) if tps else None,
        "steady_tps_per_gpu_median": statistics.median(tps) if tps else None,
        "steady_tps_per_gpu_min": min(tps) if tps else None,
        "steady_tps_per_gpu_max": max(tps) if tps else None,
        "steady_tps_total_mean": statistics.fmean(tps_total) if tps_total else None,
        "steady_tps_total_median": statistics.median(tps_total) if tps_total else None,
        "max_mem_gib_seen": max(mem) if mem else None,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "records": records,
    }
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

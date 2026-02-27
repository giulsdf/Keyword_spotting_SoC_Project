#!/usr/bin/env python3
"""
utils/merge_logs.py
===================
Unisce il log delle keyword con il log dei sensori
(del progetto soc_arch_21) basandosi sul timestamp.

Uso:
    python utils/merge_logs.py \
        --keywords logs/keyword_log.txt \
        --sensors  /path/to/sensor_log.csv \
        --out      logs/merged.csv \
        --window   1.0          # secondi di tolleranza per il match
"""

import argparse
import csv
import datetime
from bisect import bisect_left


def parse_keyword_log(path: str):
    """Legge il log keyword e restituisce lista di dict."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = dict(p.split("=", 1) for p in line.split(" | ") if "=" in p)
            # es: unix=1706262121.423
            try:
                ts      = float(parts["unix"])
                keyword = parts["keyword"]
                conf    = float(parts["conf"])
                events.append({"unix": ts, "keyword": keyword, "conf": conf})
            except (KeyError, ValueError):
                continue
    return sorted(events, key=lambda e: e["unix"])


def parse_sensor_log(path: str):
    """
    Legge il CSV del progetto soc_arch_21.
    Assume che la prima colonna si chiami 'timestamp' (unix float o ISO).
    Adatta se necessario.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts_raw = row.get("timestamp") or row.get("time") or row.get("ts")
                if ts_raw is None:
                    continue
                # prova prima unix, poi ISO
                try:
                    ts = float(ts_raw)
                except ValueError:
                    ts = datetime.datetime.fromisoformat(ts_raw).timestamp()
                row["_unix"] = ts
                rows.append(row)
            except Exception:
                continue
    return sorted(rows, key=lambda r: r["_unix"])


def merge(keywords, sensors, window_sec=1.0):
    """Associa ogni evento keyword alla riga sensore più vicina nel tempo."""
    sensor_times = [r["_unix"] for r in sensors]
    merged = []

    for ev in keywords:
        # trova la riga sensore più vicina
        pos = bisect_left(sensor_times, ev["unix"])
        candidates = []
        if pos < len(sensors):
            candidates.append(sensors[pos])
        if pos > 0:
            candidates.append(sensors[pos - 1])

        best = min(candidates, key=lambda r: abs(r["_unix"] - ev["unix"]), default=None)
        if best is None or abs(best["_unix"] - ev["unix"]) > window_sec:
            best = {}

        row = {
            "keyword_unix": ev["unix"],
            "keyword":      ev["keyword"],
            "confidence":   ev["conf"],
        }
        row.update({k: v for k, v in best.items() if not k.startswith("_")})
        merged.append(row)

    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords", required=True)
    ap.add_argument("--sensors",  required=True)
    ap.add_argument("--out",      default="logs/merged.csv")
    ap.add_argument("--window",   type=float, default=1.0)
    args = ap.parse_args()

    print(f"📂  Lettura keyword log: {args.keywords}")
    keywords = parse_keyword_log(args.keywords)
    print(f"    {len(keywords)} eventi keyword trovati")

    print(f"📂  Lettura sensor log: {args.sensors}")
    sensors = parse_sensor_log(args.sensors)
    print(f"    {len(sensors)} righe sensore trovate")

    merged = merge(keywords, sensors, args.window)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if merged:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=merged[0].keys())
            writer.writeheader()
            writer.writerows(merged)
        print(f"✅  File unito salvato: {args.out} ({len(merged)} righe)")
    else:
        print("⚠️   Nessun dato da unire.")


if __name__ == "__main__":
    main()

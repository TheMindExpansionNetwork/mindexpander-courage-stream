#!/usr/bin/env python3
"""
SageAttention vs Stock SDPA benchmark for VoxCPM2.

Hits both the original Modal endpoint and the SageAttention-optimized one
with identical text, measures generation time, RTF, and audio duration.
Produces a comparison table.

Usage:
  # Compare both endpoints side-by-side
  python scripts/benchmark_sage.py

  # Override endpoint URLs
  python scripts/benchmark_sage.py \\
    --original https://your-original.modal.run \\
    --sage https://your-sage.modal.run

  # Custom test texts & parameters
  python scripts/benchmark_sage.py \\
    --text "A longer test sentence for benchmarking purposes." \\
    --steps 20 --cfg 2.0 --runs 3

  # Run only one side (e.g., just check Sage)
  python scripts/benchmark_sage.py --sage-only

  # Save audio outputs for A/B listening
  python scripts/benchmark_sage.py --save-audio /tmp/bench_audio
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# ── Default endpoint URLs (modal.run deployed endpoints) ──
DEFAULT_ORIGINAL = os.getenv(
    "BENCH_ORIGINAL_ENDPOINT",
    "https://m1ndb0t-2045--mindexpander-voxcpm-api.modal.run",
)
DEFAULT_SAGE = os.getenv(
    "BENCH_SAGE_ENDPOINT",
    "https://m1ndb0t-2045--mindexpander-voxcpm-sage-api.modal.run",
)

# API key
API_KEY = os.getenv("MINDEXPANDER_API_KEY", "mindexpander-dev-token")

# ── Default test prompts (short / medium / long) ──
TEST_PROMPTS = {
    "short": "Hello world.",
    "medium": "The quick brown fox jumps over the lazy dog near the riverbank.",
    "long": (
        "In the year 2147, humanity had spread across the solar system. "
        "Mars was a bustling hub of commerce, the asteroid belt a frontier "
        "of mining colonies, and Europa's subsurface ocean harbored the "
        "first confirmed extraterrestrial life. But it was on Titan, "
        "Saturn's largest moon, that the most startling discovery was made."
    ),
}


def make_request(
    endpoint_url: str,
    text: str,
    steps: int = 10,
    cfg: float = 2.0,
    voice: str = "default",
    format: str = "wav",
    timeout: int = 180,
) -> dict:
    """
    Send a TTS generation request and return timing info.
    Returns dict with keys: status, duration_s, elapsed_s, rtf, audio_bytes, sage_active, error.
    """
    url = f"{endpoint_url.rstrip('/')}/v1/audio/speech"
    payload = {
        "model": "mindexpander-voxcpm2",
        "input": text,
        "voice": voice,
        "response_format": format,
        "cfg_value": cfg,
        "inference_timesteps": steps,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        return {"status": "timeout", "error": f"Request timed out after {timeout}s"}
    except requests.exceptions.ConnectionError as e:
        return {"status": "connection_error", "error": str(e)}
    elapsed = time.perf_counter() - start

    if resp.status_code != 200:
        return {
            "status": f"http_{resp.status_code}",
            "error": resp.text[:300],
            "elapsed_s": round(elapsed, 2),
        }

    duration = float(resp.headers.get("X-Duration-Seconds", 0))
    rtf = elapsed / duration if duration > 0 else float("inf")
    sage_active = resp.headers.get("X-Sage-Patched", "false").lower() == "true"

    return {
        "status": "ok",
        "duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(rtf, 4),
        "audio_bytes": resp.content,
        "sage_active": sage_active,
        "error": None,
    }


def run_benchmark_on_endpoint(
    endpoint: str,
    label: str,
    texts: list[str],
    steps: int,
    cfg: float,
    runs: int,
    warmup: bool = True,
) -> list[dict]:
    """
    Run benchmark on a single endpoint across multiple texts/runs.
    Returns list of per-run result dicts.
    """
    results = []

    # Warmup with first text (fire-and-forget)
    if warmup and texts:
        print(f"  [{label}] Warming up with '{texts[0][:40]}...'")
        _ = make_request(endpoint, texts[0], steps=steps, cfg=cfg, timeout=300)

    for text in texts:
        short_preview = text[:50].replace("\n", " ")
        for run in range(runs):
            print(f"  [{label}] Run {run+1}/{runs} — '{short_preview}...'", end=" ")
            result = make_request(endpoint, text, steps=steps, cfg=cfg)
            result["label"] = label
            result["run"] = run + 1
            result["text_preview"] = short_preview
            result["steps"] = steps
            result["cfg"] = cfg

            if result["status"] == "ok":
                print(f"✅ {result['elapsed_s']}s / {result['duration_s']}s audio → RTF {result['rtf']}")
            else:
                print(f"❌ {result['status']} — {result.get('error', '')[:80]}")

            results.append(result)
            time.sleep(0.5)  # polite spacing

    return results


def print_summary(results: list[dict]):
    """Print a formatted comparison table."""
    # Group by label
    by_label = {}
    for r in results:
        lbl = r.get("label", "unknown")
        by_label.setdefault(lbl, []).append(r)

    print("\n" + "=" * 80)
    print("  SageAttention vs Stock SDPA — VoxCPM2 Benchmark Summary")
    print("=" * 80)

    for label, group in by_label.items():
        ok_runs = [r for r in group if r["status"] == "ok"]
        if not ok_runs:
            print(f"\n  {label}: No successful runs — {group[0].get('error', 'unknown error')}")
            continue

        elapsed_vals = [r["elapsed_s"] for r in ok_runs]
        rtf_vals = [r["rtf"] for r in ok_runs]
        dur_vals = [r["duration_s"] for r in ok_runs]

        avg_elapsed = sum(elapsed_vals) / len(elapsed_vals)
        avg_rtf = sum(rtf_vals) / len(rtf_vals)
        avg_dur = sum(dur_vals) / len(dur_vals)
        min_rtf = min(rtf_vals)
        max_rtf = max(rtf_vals)

        sage_status = "SageActive" if ok_runs[0].get("sage_active") else "Stock SDPA"

        print(f"\n  ── {label} ({sage_status}) ──")
        print(f"  Runs:           {len(ok_runs)}/{len(group)} successful")
        print(f"  Avg generation: {avg_elapsed:.2f}s")
        print(f"  Avg audio:      {avg_dur:.2f}s")
        print(f"  Avg RTF:        {avg_rtf:.4f}  (min {min_rtf:.4f}, max {max_rtf:.4f})")

    # Side-by-side comparison if both labels present
    labels = list(by_label.keys())
    if len(labels) >= 2:
        l0, l1 = labels[0], labels[1]
        ok0 = [r for r in by_label[l0] if r["status"] == "ok"]
        ok1 = [r for r in by_label[l1] if r["status"] == "ok"]
        if ok0 and ok1:
            avg_rtf0 = sum(r["rtf"] for r in ok0) / len(ok0)
            avg_rtf1 = sum(r["rtf"] for r in ok1) / len(ok1)
            avg_el0 = sum(r["elapsed_s"] for r in ok0) / len(ok0)
            avg_el1 = sum(r["elapsed_s"] for r in ok1) / len(ok1)

            speedup_rtf = avg_rtf0 / avg_rtf1 if avg_rtf1 > 0 else 0
            speedup_time = avg_el0 / avg_el1 if avg_el1 > 0 else 0

            # Determine which is faster
            faster = l1 if avg_rtf1 < avg_rtf0 else l0

            print(f"\n  ── Comparison ──")
            print(f"  {l0}:  Avg RTF {avg_rtf0:.4f}, Avg time {avg_el0:.2f}s")
            print(f"  {l1}:  Avg RTF {avg_rtf1:.4f}, Avg time {avg_el1:.2f}s")
            print(f"  Speedup (RTF): {speedup_rtf:.2f}x  ({faster} is faster)")
            print(f"  Speedup (time): {speedup_time:.2f}x")

            # Check for realtime
            if avg_rtf1 < 1.0:
                print(f"  ✅ {l1} achieves realtime (RTF < 1.0)")
            if avg_rtf0 < 1.0:
                print(f"  ✅ {l0} achieves realtime (RTF < 1.0)")

    print("\n" + "=" * 80)


def save_audio(results: list[dict], output_dir: str):
    """Save generated audio files for A/B listening tests."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(results):
        if r["status"] != "ok" or r.get("audio_bytes") is None:
            continue
        label = r.get("label", "unknown").replace(" ", "_").replace("-", "_")
        fname = f"{label}_run{r.get('run', 0):02d}_{i:03d}.wav"
        fpath = out / fname
        fpath.write_bytes(r["audio_bytes"])
        print(f"  💾 Saved: {fpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SageAttention vs stock SDPA for VoxCPM2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--original", default=DEFAULT_ORIGINAL,
        help="URL of the original (non-Sage) Modal endpoint",
    )
    parser.add_argument(
        "--sage", default=DEFAULT_SAGE,
        help="URL of the SageAttention-optimized Modal endpoint",
    )
    parser.add_argument(
        "--sage-only", action="store_true",
        help="Only test the Sage endpoint (skip original)",
    )
    parser.add_argument(
        "--original-only", action="store_true",
        help="Only test the original endpoint (skip Sage)",
    )
    parser.add_argument(
        "--prompt", default="all",
        choices=["short", "medium", "long", "all"],
        help="Which test prompt(s) to use (default: all)",
    )
    parser.add_argument(
        "--text", default=None,
        help="Custom test text (overrides --prompt)",
    )
    parser.add_argument(
        "--steps", type=int, default=10,
        help="Inference timesteps (default: 10)",
    )
    parser.add_argument(
        "--cfg", type=float, default=2.0,
        help="CFG value (default: 2.0)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of runs per text per endpoint (default: 1)",
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip warmup run",
    )
    parser.add_argument(
        "--save-audio", default=None,
        help="Directory to save generated audio for A/B comparison",
    )
    parser.add_argument(
        "--json-output", default=None,
        help="Path to save full results as JSON",
    )
    args = parser.parse_args()

    # ── Determine texts ──
    if args.text:
        texts = [args.text]
    elif args.prompt == "all":
        texts = [TEST_PROMPTS["short"], TEST_PROMPTS["medium"], TEST_PROMPTS["long"]]
    else:
        texts = [TEST_PROMPTS[args.prompt]]

    # ── Determine endpoints ──
    endpoints = []
    if not args.sage_only:
        endpoints.append((args.original, "Stock-SDPA"))
    if not args.original_only:
        endpoints.append((args.sage, "SageAttention"))

    if not endpoints:
        print("❌ No endpoints selected (both --sage-only and --original-only?)")
        sys.exit(1)

    print(f"\n🔬 VoxCPM2 SageAttention Benchmark")
    print(f"   Texts: {len(texts)} prompt(s), {args.runs} run(s) each")
    print(f"   Steps: {args.steps}, CFG: {args.cfg}")
    print(f"   Endpoints: {[e[1] for e in endpoints]}")
    print()

    all_results = []

    for url, label in endpoints:
        print(f"── Benchmarking {label} ({url}) ──")
        results = run_benchmark_on_endpoint(
            endpoint=url,
            label=label,
            texts=texts,
            steps=args.steps,
            cfg=args.cfg,
            runs=args.runs,
            warmup=not args.no_warmup,
        )
        all_results.extend(results)

    # ── Summary ──
    print_summary(all_results)

    # ── Save audio ──
    if args.save_audio:
        print(f"\n💾 Saving audio to {args.save_audio}/ ...")
        save_audio(all_results, args.save_audio)

    # ── Save JSON ──
    if args.json_output:
        serializable = []
        for r in all_results:
            sr = {k: v for k, v in r.items() if k != "audio_bytes"}
            serializable.append(sr)
        with open(args.json_output, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"📊 Full results saved to {args.json_output}")


if __name__ == "__main__":
    main()

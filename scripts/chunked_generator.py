#!/usr/bin/env python3
"""
Chunked Audio Generator for VoxCPM2

Fixes voice degradation in longer generations by:
  1. Splitting text into short chunks (~30-45s each)
  2. Generating each chunk via a separate Modal API call
  3. Peak-normalizing audio levels across all chunks
  4. Inserting 0.3s silence between chunks for natural pacing
  5. Concatenating into one clean final WAV file

Usage:
  python scripts/chunked_generator.py --script scripts/sunshine_short_script.json
  python scripts/chunked_generator.py --script scripts/sunshine_short_script.json --output final.wav
  python scripts/chunked_generator.py --script scripts/sunshine_short_script.json --save-chunks
"""

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path

import requests
import numpy as np
import soundfile as sf

# ── defaults ────────────────────────────────────────────────────────────────
ENDPOINT_URL = os.getenv(
    "COURAGE_TTS_ENDPOINT",
    "https://m1ndb0t-2045--mindexpander-voxcpm-api.modal.run",
)
API_KEY = os.getenv("MINDEXPANDER_API_KEY", "mindexpander-dev-token")
DEFAULT_SILENCE_S = 0.3          # silence between chunks
DEFAULT_SAMPLE_RATE = 24000      # VoxCPM2 output sample rate
DEFAULT_CFG = 2.0
DEFAULT_STEPS = 10
MAX_RETRIES = 2                  # retry on failure


# ── helpers ─────────────────────────────────────────────────────────────────

def _cache_key(text: str, cfg: float, steps: int) -> str:
    """Deterministic key so repeated runs don't re-generate identical chunks."""
    payload = f"{text}|cfg={cfg}|steps={steps}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def generate_tts(
    text: str,
    cfg_value: float = DEFAULT_CFG,
    inference_timesteps: int = DEFAULT_STEPS,
    cache_dir: str | None = None,
) -> tuple[bytes | None, float, float]:
    """Generate TTS audio from the Modal endpoint.

    Returns (audio_bytes, audio_duration_seconds, generation_wall_time).
    Returns (None, 0, gen_time) on failure.
    """
    url = f"{ENDPOINT_URL}/v1/audio/speech"
    payload = {
        "model": "mindexpander-voxcpm2",
        "input": text.strip(),
        "voice": "default",
        "response_format": "wav",
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # check cache
    if cache_dir:
        key = _cache_key(text, cfg_value, inference_timesteps)
        cache_path = os.path.join(cache_dir, f"{key}.wav")
        if os.path.exists(cache_path):
            info = sf.info(cache_path)
            with open(cache_path, "rb") as f:
                return f.read(), info.duration, 0.0

    print(f"  🎤 Generating ({len(text.strip())} chars)...", flush=True)
    start = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=180)
    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"  ❌ Error {resp.status_code}: {resp.text[:200]}", flush=True)
        return None, 0.0, elapsed

    duration = float(resp.headers.get("X-Duration-Seconds", 0))
    rtf = elapsed / duration if duration > 0 else float("inf")
    print(f"  ✅ {duration:.1f}s audio in {elapsed:.1f}s (RTF: {rtf:.2f})", flush=True)

    # cache if requested
    if cache_dir:
        key = _cache_key(text, cfg_value, inference_timesteps)
        cache_path = os.path.join(cache_dir, f"{key}.wav")
        with open(cache_path, "wb") as f:
            f.write(resp.content)

    return resp.content, duration, elapsed


def read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Read WAV bytes into (samples, sample_rate)."""
    from io import BytesIO
    bio = BytesIO(data)
    samples, sr = sf.read(bio)
    return samples, sr


def peak_normalize(samples: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize audio samples so the max absolute value equals target_peak."""
    peak = np.max(np.abs(samples))
    if peak == 0 or peak <= target_peak * 1.01:
        # already quiet enough, no amplification needed
        return samples
    return samples * (target_peak / peak)


def make_silence(duration_s: float, sample_rate: int) -> np.ndarray:
    """Return a 1-D array of zeros for the given duration."""
    n = int(duration_s * sample_rate)
    return np.zeros(n, dtype=np.float32)


def concatenate_wavs(
    chunks: list[np.ndarray],
    sample_rate: int,
    silence_s: float = DEFAULT_SILENCE_S,
) -> np.ndarray:
    """Concatenate audio chunks with silence between them.

    Each chunk is peak-normalized independently before joining.
    """
    silence = make_silence(silence_s, sample_rate)
    parts: list[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        parts.append(peak_normalize(chunk))
        if i < len(chunks) - 1:
            parts.append(silence)
    return np.concatenate(parts)


# ── main pipeline ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chunked VoxCPM2 Generator — fix voice degradation",
    )
    parser.add_argument(
        "--script", required=True,
        help="Path to JSON script file (list of objects or strings)",
    )
    parser.add_argument(
        "--output", default="chunked_output.wav",
        help="Path for the final concatenated WAV file (default: chunked_output.wav)",
    )
    parser.add_argument(
        "--silence", type=float, default=DEFAULT_SILENCE_S,
        help=f"Silence between chunks in seconds (default: {DEFAULT_SILENCE_S})",
    )
    parser.add_argument(
        "--cfg", type=float, default=DEFAULT_CFG,
        help=f"Classifier-free guidance (default: {DEFAULT_CFG})",
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS,
        help=f"Inference timesteps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--endpoint", help="Override TTS endpoint URL",
    )
    parser.add_argument(
        "--save-chunks", action="store_true",
        help="Save individual chunk WAV files alongside the final output",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Cache directory for generated chunks (avoids re-generation)",
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Skip peak normalization (not recommended)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print script without generating",
    )
    args = parser.parse_args()

    global ENDPOINT_URL
    if args.endpoint:
        ENDPOINT_URL = args.endpoint

    # load script
    with open(args.script) as f:
        script = json.load(f)

    # Normalise to list of {"text": ..., ...}
    items: list[dict] = []
    for entry in script:
        if isinstance(entry, str):
            items.append({"text": entry})
        elif isinstance(entry, dict):
            items.append(entry)
        else:
            print(f"⚠️  Skipping unrecognised entry: {entry}")
            continue

    if not items:
        print("❌ No valid entries in script.")
        sys.exit(1)

    if args.dry_run:
        for i, item in enumerate(items):
            print(f"[{i+1}/{len(items)}] {item['text'][:100]}...")
        return

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    # output directory
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ensure we save chunks if output dir is writable and chunks requested
    chunks_dir = None
    if args.save_chunks:
        chunks_dir = os.path.splitext(args.output)[0] + "_chunks"
        os.makedirs(chunks_dir, exist_ok=True)

    print(f"\n🧩 Chunked Generator — {len(items)} chunks")
    print(f"   Endpoint  : {ENDPOINT_URL}")
    print(f"   Settings  : steps={args.steps}, cfg={args.cfg}, silence={args.silence}s")
    print(f"   Output    : {os.path.abspath(args.output)}\n")

    # ── phase 1: generate all chunks ────────────────────────────────────
    chunk_data: list[tuple[np.ndarray, int, float, float]] = []
    # (samples, sample_rate, audio_duration, gen_wall_time)
    total_gen_time = 0.0
    total_audio_dur = 0.0
    success_count = 0

    for i, item in enumerate(items):
        text = item["text"].strip()
        label = item.get("label", f"chunk_{i+1:02d}")
        print(f"[{i+1}/{len(items)}] {label}")
        print(f"  📝 {text[:100]}{'...' if len(text) > 100 else ''}")

        audio_bytes = None
        for attempt in range(1 + MAX_RETRIES):
            audio_bytes, dur, gen_time = generate_tts(
                text,
                cfg_value=args.cfg,
                inference_timesteps=args.steps,
                cache_dir=args.cache_dir,
            )
            if audio_bytes is not None:
                break
            if attempt < MAX_RETRIES:
                wait = 2 * (attempt + 1)
                print(f"  🔄 Retry {attempt+1}/{MAX_RETRIES} in {wait}s...")
                time.sleep(wait)

        if audio_bytes is None:
            print(f"  ❌ Failed after {MAX_RETRIES+1} attempts, skipping chunk.")
            continue

        # decode WAV bytes
        with open(f"/tmp/_chunk_{i}.wav", "wb") as f:
            f.write(audio_bytes)
        samples, sr = sf.read(f"/tmp/_chunk_{i}.wav")
        os.unlink(f"/tmp/_chunk_{i}.wav")

        chunk_data.append((samples, sr, dur, gen_time))
        total_gen_time += gen_time
        total_audio_dur += dur
        success_count += 1

        # save individual chunk if requested
        if chunks_dir:
            chunk_path = os.path.join(chunks_dir, f"{label}.wav")
            sf.write(chunk_path, samples, sr)
            print(f"  💾 {chunk_path}")

    # ── phase 2: stats ──────────────────────────────────────────────────
    if success_count == 0:
        print("\n❌ No chunks generated successfully. Aborting.")
        sys.exit(1)

    print(f"\n📊 Chunk Stats")
    print(f"   Generated   : {success_count}/{len(items)}")
    print(f"   Audio dur   : {total_audio_dur:.1f}s ({total_audio_dur/60:.1f} min)")
    print(f"   Gen time    : {total_gen_time:.1f}s")
    rtf_total = total_gen_time / total_audio_dur if total_audio_dur > 0 else 0
    print(f"   Overall RTF : {rtf_total:.3f}")

    # per-chunk detail
    print(f"\n   {'Chunk':<12} {'Audio':>8} {'Gen':>8} {'RTF':>8}")
    print(f"   {'─'*12} {'─'*8} {'─'*8} {'─'*8}")
    for i, (_, _, d, g) in enumerate(chunk_data):
        rtf = g / d if d > 0 else float("inf")
        print(f"   chunk_{i+1:02d}.wav  {d:>7.1f}s {g:>7.1f}s {rtf:>7.3f}")

    # ── phase 3: normalize & concatenate ────────────────────────────────
    sample_rate = chunk_data[0][1] if chunk_data else DEFAULT_SAMPLE_RATE
    all_samples = [c[0] for c in chunk_data]

    # find global peak across all chunks for consistent normalization
    if not args.no_normalize:
        global_peak = max(np.max(np.abs(s)) for s in all_samples)
        print(f"\n🔊 Global peak: {global_peak:.4f}  — normalizing to 0.95")
        all_samples = [s * (0.95 / global_peak) if global_peak > 0 else s
                       for s in all_samples]

    final = concatenate_wavs(all_samples, sample_rate, silence_s=args.silence)
    # re-normalize the full concatenated file to ensure no clipping from silence joins
    final = peak_normalize(final, target_peak=0.95)

    sf.write(args.output, final, sample_rate)
    final_dur = len(final) / sample_rate
    out_size_mb = os.path.getsize(args.output) / (1024 * 1024)

    print(f"\n✅ Final output written to: {os.path.abspath(args.output)}")
    print(f"   Duration    : {final_dur:.1f}s ({final_dur/60:.1f} min)")
    print(f"   File size   : {out_size_mb:.1f} MB")
    print(f"   Silences    : {len(chunk_data)-1} × {args.silence}s = "
          f"{(len(chunk_data)-1)*args.silence:.1f}s total")
    print(f"   Sample rate : {sample_rate} Hz")
    print(f"\n🏁 Done! 🎉")


if __name__ == "__main__":
    main()

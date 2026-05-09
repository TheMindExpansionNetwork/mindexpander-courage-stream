#!/usr/bin/env python3
"""
MindExpander Courage TV Stream Script

Generates TTS audio from a script file using the Modal VoxCPM2 endpoint,
plays it back, and controls the TV overlay via browser source.

Usage:
  python scripts/stream_courage.py --script scripts/sample_script.json

The script JSON format:
[
  {"text": "Line one", "channel": "CH 03", "pause_after": 1.0},
  {"text": "Line two", "channel": "CH 07", "pause_after": 0.5}
]
"""

import argparse
import json
import os
import sys
import time
import subprocess
import requests
import tempfile
from pathlib import Path


ENDPOINT_URL = os.getenv(
    "COURAGE_TTS_ENDPOINT",
    "https://m1ndb0t-2045--mindexpander-voxcpm-api.modal.run"
)


def generate_tts(text: str, cfg_value: float = 2.0, inference_timesteps: int = 10) -> bytes:
    """Generate TTS audio from the Modal endpoint."""
    url = f"{ENDPOINT_URL}/v1/audio/speech"
    payload = {
        "model": "mindexpander-voxcpm2",
        "input": text,
        "voice": "default",
        "response_format": "wav",
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
    }

    api_key = os.getenv("MINDEXPANDER_API_KEY", "mindexpander-dev-token")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"  🎤 Generating: {text[:60]}...")
    start = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"  ❌ Error {resp.status_code}: {resp.text[:200]}")
        return None

    duration = float(resp.headers.get("X-Duration-Seconds", 0))
    rtf = elapsed / duration if duration > 0 else float("inf")
    print(f"  ✅ {duration:.1f}s audio in {elapsed:.1f}s (RTF: {rtf:.2f})")

    return resp.content


def play_audio(audio_bytes: bytes, format: str = "wav"):
    """Play audio using ffplay (or save to file)."""
    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
            timeout=120,
        )
    except FileNotFoundError:
        print("  ⚠️  ffplay not found, saving to /tmp/courage_last.wav instead")
        out = "/tmp/courage_last.wav"
        Path(tmp_path).rename(out)
        print(f"  💾 Saved: {out}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def run_stream(script_path: str, cfg: float, steps: int, save_dir: str = None):
    """Run through a script file, generating and playing each line."""
    with open(script_path) as f:
        lines = json.load(f)

    print(f"\n📺 Courage TV Stream — {len(lines)} lines")
    print(f"   Endpoint: {ENDPOINT_URL}")
    print(f"   Settings: steps={steps}, cfg={cfg}\n")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    total_audio = 0
    total_time = 0

    for i, line in enumerate(lines):
        text = line["text"]
        channel = line.get("channel", "CH 03")
        pause = line.get("pause_after", 0.5)

        print(f"[{i+1}/{len(lines)}] {channel} — {text[:80]}")

        start = time.time()
        audio = generate_tts(text, cfg_value=cfg, inference_timesteps=steps)
        gen_time = time.time() - start

        if audio is None:
            print("  ⏭️  Skipping...")
            continue

        if save_dir:
            out_path = os.path.join(save_dir, f"line_{i+1:03d}.wav")
            with open(out_path, "wb") as f:
                f.write(audio)
            print(f"  💾 Saved: {out_path}")

        # Play the audio
        play_audio(audio)

        # Track stats
        total_audio += float(0)  # duration from headers
        total_time += gen_time

        # Pause between lines
        if pause > 0:
            time.sleep(pause)

    print(f"\n🏁 Stream complete!")
    print(f"   Total generation time: {total_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Courage TV Stream Runner")
    parser.add_argument("--script", required=True, help="Path to script JSON file")
    parser.add_argument("--cfg", type=float, default=2.0, help="CFG value")
    parser.add_argument("--steps", type=int, default=10, help="Inference timesteps")
    parser.add_argument("--save-dir", help="Directory to save generated audio files")
    parser.add_argument("--endpoint", help="Override TTS endpoint URL")
    args = parser.parse_args()

    if args.endpoint:
        global ENDPOINT_URL
        ENDPOINT_URL = args.endpoint

    run_stream(args.script, args.cfg, args.steps, args.save_dir)


if __name__ == "__main__":
    main()

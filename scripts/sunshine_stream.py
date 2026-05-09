#!/usr/bin/env python3
"""
The Sunshine Makers Stream Script

Generates AI commentary for the 1935 cartoon "The Sunshine Makers"
using the VoxCPM2 Modal endpoint. One continuous ramble per chunk.

Usage:
  python scripts/sunshine_stream.py --endpoint https://YOUR_URL
"""

import argparse
import json
import os
import sys
import time
import tempfile
import subprocess
import requests

ENDPOINT_URL = os.getenv(
    "COURAGE_TTS_ENDPOINT",
    "https://m1ndb0t-2045--mindexpander-voxcpm-api.modal.run"
)

# Timestamp-based script: (timestamp_seconds, text)
# These are timed to match key moments in the 7-minute cartoon
SCRIPT = [
    (0, """
(slow, relaxed, like you are watching something weird at 3 AM)
Okay chat... so I found this cartoon from 1935 called The Sunshine Makers
and... [sigh] I need to talk about it.
"""),

    (45, """
So basically there are these happy orange gnomes and their whole deal is
they bottle sunshine. Like that is their job. They wake up, they go to work,
they bottle sunshine into little milk bottles.
"""),

    (90, """
And then there are these sad blue goblins who live in a cave and they are
just... miserable. Like existentially miserable. And the gnomes are like,
you know what will fix this? MILK. Specifically sunshine-milk.
"""),

    (135, """
So they literally load up catapults and start BOMBING the goblins with
bottles of liquid sunshine and the goblins drink it and turn orange and
start dancing. I am not making any of this up. This is a real cartoon
that someone made in 1935.
"""),

    (180, """
And the thing is... it works. Like the message is literally, the solution
to all sadness is to force happiness on people through projectile dairy
products. And honestly? [Uhm] ...I kind of respect it.
"""),

    (225, """
Like someone in a board room at Borden Milk said, what if we made a cartoon
where our product defeats depression through violence? And everyone was like
yeah, green light, ship it.
"""),

    (270, """
The animation is actually gorgeous though. The color coding is brilliant,
everything happy is orange and warm and everything sad is blue and cold.
It is like a screensaver from 1935 that someone accidentally put a story into.
"""),

    (315, """
[sigh] Anyway, we are watching this on a cyberpunk TV and I cannot recommend
it enough. Seven minutes of your life. You will never look at milk the
same way again.
"""),

    (360, """
[Uhm] so chat... what did we just watch? I genuinely do not know. But I
think I need to go drink some milk and contemplate my existence. Stay
sunshine out there.
"""),
]


def generate_tts(text, cfg=2.0, steps=10):
    """Generate TTS audio from the Modal endpoint."""
    url = f"{ENDPOINT_URL}/v1/audio/speech"
    payload = {
        "model": "mindexpander-voxcpm2",
        "input": text.strip(),
        "voice": "default",
        "response_format": "wav",
        "cfg_value": cfg,
        "inference_timesteps": steps,
    }
    api_key = os.getenv("MINDEXPANDER_API_KEY", "mindexpander-dev-token")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"  🎤 Generating ({len(text.strip())} chars)...", flush=True)
    start = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=180)
    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"  ❌ Error {resp.status_code}: {resp.text[:200]}")
        return None, 0, 0

    duration = float(resp.headers.get("X-Duration-Seconds", 0))
    rtf = elapsed / duration if duration > 0 else float("inf")
    print(f"  ✅ {duration:.1f}s audio in {elapsed:.1f}s (RTF: {rtf:.2f})")
    return resp.content, duration, elapsed


def play_audio(audio_bytes):
    """Play audio using ffplay."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp], timeout=120)
    except FileNotFoundError:
        out = "/tmp/sunshine_last.wav"
        os.rename(tmp, out)
        print(f"  💾 Saved: {out}")
    finally:
        try: os.unlink(tmp)
        except: pass


def main():
    parser = argparse.ArgumentParser(description="Sunshine Makers Stream")
    parser.add_argument("--endpoint", help="Override TTS endpoint URL")
    parser.add_argument("--save-dir", help="Save generated audio files")
    parser.add_argument("--dry-run", action="store_true", help="Print script without generating")
    parser.add_argument("--generate-all", action="store_true", help="Generate all chunks first, then play")
    args = parser.parse_args()

    if args.endpoint:
        global ENDPOINT_URL
        ENDPOINT_URL = args.endpoint

    if args.dry_run:
        for ts, text in SCRIPT:
            print(f"[{ts//60}:{ts%60:02d}] {text.strip()[:80]}...")
        return

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n☀️ The Sunshine Makers Stream — {len(SCRIPT)} chunks")
    print(f"   Endpoint: {ENDPOINT_URL}")
    print(f"   Total runtime: ~{SCRIPT[-1][0]//60} minutes\n")

    total_gen_time = 0
    total_audio_dur = 0

    for i, (ts, text) in enumerate(SCRIPT):
        print(f"[{i+1}/{len(SCRIPT)}] @ {ts//60}:{ts%60:02d}")
        audio, dur, gen_time = generate_tts(text)

        if audio is None:
            continue

        if args.save_dir:
            path = os.path.join(args.save_dir, f"chunk_{i+1:02d}.wav")
            with open(path, "wb") as f:
                f.write(audio)
            print(f"  💾 {path}")

        if not args.generate_all:
            play_audio(audio)

        total_gen_time += gen_time
        total_audio_dur += dur

    rtf = total_gen_time / total_audio_dur if total_audio_dur > 0 else 0
    print(f"\n🏁 Done! Total: {total_audio_dur:.1f}s audio in {total_gen_time:.1f}s (RTF: {rtf:.2f})")


if __name__ == "__main__":
    main()

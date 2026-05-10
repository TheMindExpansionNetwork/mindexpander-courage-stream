#!/usr/bin/env python3
"""
Tech Support Conversation Generator
Two-voice back-and-forth: MindExpander (VoxCPM) vs Agent (Pocket TTS)
"""
import json, os, sys, time, hashlib, argparse
from io import BytesIO
from pathlib import Path

import requests
import numpy as np
import soundfile as sf

# ── endpoints ────────────────────────────────────────────────────────────────
VOXCPM_URL = os.getenv("COURAGE_TTS_ENDPOINT", "https://m1ndb0t-2045--mindexpander-voxcpm-api.modal.run")
VOXCPM_KEY = os.getenv("MINDEXPANDER_API_KEY", "mindexpander-dev-token")
POCKET_TTS_URL = "http://localhost:49112"
AGENT_VOICE = "javert"  # corporate male agent

# ── generation settings ─────────────────────────────────────────────────────
VOXCPM_STEPS = 8       # fewer steps = faster but slightly lower quality
VOXCPM_CFG = 2.5        # good for clear speech
VOXCPM_SR = 24000       # VoxCPM2 native sample rate
POCKET_SR = 24000       # Pocket TTS sample rate (will resample to match)
FINAL_SR = 24000
LOUDNORM_TARGET = -16   # LUFS target for consistent volume

# ── conversation script ─────────────────────────────────────────────────────
# ── conversation script (🌌 QA LOG #TC-0002 — Billing Dispute) ─────────────
CONVERSATION = [
    # (speaker, text)
    ("agent", "Thank you for calling Mind Expansion Billing Department. This is Jordan. How can I assist you today?"),
    ("caller", "Yeah, I just checked my credit statement and there's a charge for forty-nine ninety-nine for something called Intergalactic Premium Consciousness Streaming. I did not sign up for this."),
    ("agent", "Let me pull up your account. Can I get your expansion license number or the last four digits of your cosmic signature?"),
    ("caller", "I don't know my cosmic signature. Nobody told me I'd need a cosmic signature. I just want this charge reversed. It's been three months. That's a hundred and fifty dollars."),
    ("agent", "I understand. Let me look at your subscription history. I'm showing here that you accepted the premium trial during your last firmware update. It converts to paid after thirty days."),
    ("caller", "Wait. What? You mean that checkbox I clicked to update my consciousness module automatically signed me up for a subscription? That's buried on page forty-seven of the terms nobody reads."),
    ("agent", "The terms are available for review at any time, sir. Would you like me to read them to you now?"),
    ("caller", "No, I don't want you to read them. I want you to cancel it and refund me. I haven't even used premium streaming. My consciousness has been streaming at the same resolution it always has. Standard definition thoughts. Basic cable consciousness."),
    ("agent", "I can help you with the cancellation, but unfortunately our refund policy only covers the most recent billing cycle. That would be forty-nine ninety-nine."),
    ("caller", "So you're telling me I'm out a hundred bucks because of a checkbox? That's insane. This is exactly the kind of thing the Intergalactic Consumer Protection Agency should know about."),
    ("agent", "You're certainly welcome to file a complaint, sir. I can provide you with the case reference number for the ICP-A."),
    ("caller", "Oh, I bet you can. What's the number? Write this down. I want everything documented. Every single charge. Every single call."),
    ("agent", "Let me pull up that reference number for you now. Please hold."),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def generate_voxcpm(text: str, retries: int = 2) -> bytes | None:
    """Generate MindExpander voice via VoxCPM Modal endpoint."""
    url = f"{VOXCPM_URL}/v1/audio/speech"
    payload = {
        "model": "mindexpander-voxcpm2",
        "input": text.strip(),
        "voice": "default",
        "response_format": "wav",
        "cfg_value": VOXCPM_CFG,
        "inference_timesteps": VOXCPM_STEPS,
    }
    headers = {"Authorization": f"Bearer {VOXCPM_KEY}", "Content-Type": "application/json"}

    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=180)
            if resp.status_code == 200:
                return resp.content
            print(f"  VoxCPM {resp.status_code}: {resp.text[:150]}", flush=True)
        except Exception as e:
            print(f"  VoxCPM error: {e}", flush=True)
        if attempt < retries:
            time.sleep(3)
    return None


def generate_pocket(text: str) -> bytes | None:
    """Generate agent voice via Pocket TTS."""
    url = f"{POCKET_TTS_URL}/v1/audio/speech"
    payload = {
        "model": "tts-1",
        "input": text.strip(),
        "voice": AGENT_VOICE,
        "response_format": "wav",
        "speed": 0.85,  # slower, more natural
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.content
        print(f"  PocketTTS {resp.status_code}: {resp.text[:150]}", flush=True)
    except Exception as e:
        print(f"  PocketTTS error: {e}", flush=True)
    return None


def loudnorm(samples: np.ndarray, sr: int, target_lufs: float = -16) -> np.ndarray:
    """Normalize to target LUFS using ffmpeg loudnorm-style RMS approximation."""
    peak = np.max(np.abs(samples))
    if peak < 1e-6:
        return samples
    # Simple peak-headroom normalization (close enough for conversation)
    target_peak = 0.85  # leave headroom for mix
    return samples * (target_peak / peak)


def add_silence(duration_s: float, sr: int) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/tech_support_call_v4.wav")
    parser.add_argument("--output-mp3", default="output/tech_support_call_v4.mp3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hold-music", default=None, help="Path to hold music MP3 (optional)")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    if args.dry_run:
        for speaker, text in CONVERSATION:
            tag = "🧠 CALLER" if speaker == "caller" else "📞 AGENT"
            print(f"{tag}: {text[:100]}...")
        return

    print(f"🎙️  Tech Support Conversation Generator")
    print(f"   Turns: {len(CONVERSATION)}")
    print(f"   Caller voice: VoxCPM (steps={VOXCPM_STEPS}, cfg={VOXCPM_CFG})")
    print(f"   Agent voice: Pocket TTS ({AGENT_VOICE}, speed=0.85)")
    print()

    # Warmup VoxCPM first
    print("🔥 Warming up VoxCPM endpoint...")
    _ = generate_voxcpm("test warmup")
    print()

    # Generate all turns
    clips = []  # list of (samples, sr, speaker)
    
    for i, (speaker, text) in enumerate(CONVERSATION):
        tag = "🧠" if speaker == "caller" else "📞"
        print(f"[{i+1}/{len(CONVERSATION)}] {tag} {text[:80]}...")

        if speaker == "caller":
            audio = generate_voxcpm(text)
            sr = VOXCPM_SR
        else:
            audio = generate_pocket(text)
            sr = POCKET_SR

        if audio is None:
            print(f"  ❌ Failed — skipping turn")
            continue

        # Decode and resample if needed
        bio = BytesIO(audio)
        samples, actual_sr = sf.read(bio)
        
        # Ensure mono
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        
        # Resample to final rate if different
        if actual_sr != FINAL_SR:
            from scipy import signal as sp_sig
            ratio = FINAL_SR / actual_sr
            n_out = int(len(samples) * ratio)
            samples = sp_sig.resample(samples, n_out)
        
        # Normalize
        samples = loudnorm(samples, FINAL_SR)
        clips.append((samples, speaker))
        
        dur = len(samples) / FINAL_SR
        print(f"  ✅ {dur:.1f}s")

    if not clips:
        print("❌ No clips generated.")
        sys.exit(1)

    # ── assemble conversation with natural pacing ──────────────────────────
    # Timing: faster response gaps between angry caller and agent, 
    # slightly longer pause after agent speaks (they're slow/corporate)
    
    parts = []
    for i, (samples, speaker) in enumerate(clips):
        parts.append(samples)
        
        if i < len(clips) - 1:
            next_speaker = clips[i + 1][1]
            if speaker == "caller" and next_speaker == "agent":
                # Agent pauses slightly before responding
                gap = 0.5
            elif speaker == "agent" and next_speaker == "caller":
                # Caller jumps in fast (frustrated)
                gap = 0.25
            else:
                gap = 0.35
            parts.append(add_silence(gap, FINAL_SR))

    # Hold music at the end
    if args.hold_music and os.path.exists(args.hold_music):
        print(f"\n🎵 Adding hold music: {args.hold_music}")
        import subprocess, tempfile
        # Convert hold music to WAV at correct rate
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", args.hold_music,
            "-ar", str(FINAL_SR), "-ac", "1", "-t", "12",
            "-af", "afade=t=in:d=0.5,afade=t=out:st=10:d=2,volume=0.4",
            tmp
        ], capture_output=True)
        if os.path.exists(tmp):
            hold_samples, _ = sf.read(tmp)
            if hold_samples.ndim > 1:
                hold_samples = hold_samples.mean(axis=1)
            # Agent announces hold, then music
            parts.append(add_silence(0.4, FINAL_SR))
            parts.append(hold_samples)
            os.unlink(tmp)
            print(f"   Hold music: {len(hold_samples)/FINAL_SR:.1f}s")

    final = np.concatenate(parts)
    # Final peak normalize
    peak = np.max(np.abs(final))
    if peak > 0:
        final = final * (0.95 / peak)

    # Write WAV
    sf.write(args.output, final, FINAL_SR)
    dur = len(final) / FINAL_SR
    print(f"\n✅ WAV: {args.output} ({dur:.1f}s)")

    # Convert to MP3 for Telegram
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", args.output,
        "-codec:a", "libmp3lame", "-b:a", "128k",
        args.output_mp3
    ], capture_output=True)
    
    if os.path.exists(args.output_mp3):
        size_kb = os.path.getsize(args.output_mp3) / 1024
        print(f"✅ MP3: {args.output_mp3} ({size_kb:.0f} KB)")
    else:
        print(f"⚠️  MP3 conversion failed")

    print(f"\n🏁 Done!")


if __name__ == "__main__":
    main()

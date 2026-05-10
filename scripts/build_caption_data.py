#!/usr/bin/env python3
"""
Generate HyperFrames composition HTML for tech support call video.
MindExpander (top-left) vs Agent (bottom-right) with synced captions.
"""
import json, os, math
from pathlib import Path

# Load transcription
with open("/opt/data/workspace/projects/mindexpander-courage-stream/output/tech_support_transcript.json") as f:
    transcript = json.load(f)

words = transcript["words"]
total_dur = transcript["duration_s"]

# Speaker assignment: alternate between agent and caller based on segments
# Agent = Pocket TTS (javert), Caller = MindExpander (VoxCPM)
# The conversation alternates: agent, caller, agent, caller...
segments = transcript["segments"]

# Assign speakers — they alternate
speaker_map = []
current = "agent"  # starts with agent
for seg in segments:
    speaker_map.append({
        "start": seg["start"],
        "end": seg["end"],
        "speaker": current,
    })
    current = "caller" if current == "agent" else "agent"

# Build word-level caption data
captions = []
current_seg = 0
for w in words:
    # Find which segment this word belongs to
    while current_seg < len(speaker_map) and w["start"] > speaker_map[current_seg]["end"]:
        current_seg += 1
    speaker = speaker_map[min(current_seg, len(speaker_map)-1)]["speaker"] if speaker_map else "agent"
    
    captions.append({
        "word": w["word"],
        "start": w["start"],
        "end": w["end"],
        "speaker": speaker,
    })

# Write speaker mapping for the HTML
speaker_json = json.dumps(speaker_map)
captions_json = json.dumps(captions[:500])  # sample for preview

print(f"Total words: {len(captions)}")
print(f"Total duration: {total_dur:.1f}s")
print(f"Speaker segments: {len(speaker_map)}")
for s in speaker_map:
    tag = "🧠" if s["speaker"] == "caller" else "📞"
    print(f"  [{s['start']:.1f}-{s['end']:.1f}s] {tag} {s['speaker']}")

# Save captions with speaker info
captions_out = {
    "words": captions,
    "segments": speaker_map,
    "total_duration": total_dur,
}
out_path = "/opt/data/workspace/projects/mindexpander-courage-stream/output/captions_with_speakers.json"
with open(out_path, "w") as f:
    json.dump(captions_out, f)
print(f"\n✅ Saved to {out_path}")

# Also generate the full convenience JS file for HyperFrames
js_lines = ["// Auto-generated caption data for HyperFrames"]
js_lines.append(f"const CAPTION_WORDS = {json.dumps(captions)};")
js_lines.append(f"const SPEAKER_SEGMENTS = {json.dumps(speaker_map)};")
js_lines.append(f"const TOTAL_DURATION = {total_dur};")

js_path = "/opt/data/workspace/projects/mindexpander-courage-stream/output/caption_data.js"
with open(js_path, "w") as f:
    f.write("\n".join(js_lines))
print(f"✅ JS data saved to {js_path}")

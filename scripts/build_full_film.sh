#!/bin/bash
# Full 8-min composite pipeline
# Usage: bash scripts/build_full_film.sh
set -e

PROJECT="/opt/data/workspace/projects/mindexpander-courage-stream"
OUT="$PROJECT/output/sunshine_makers"
CARTOON="$PROJECT/assets/cartoons/sunshine_makers.mp4"
TV_FRAME="$PROJECT/cyberpunk-tv/renders/cyberpunk-tv_2026-05-09_22-55-07.mp4"
COMMENTARY="$OUT/full_commentary.wav"
FILM_AUDIO="$OUT/film_audio_24k.wav"

DURATION=488  # 8:08

echo "🎬 Building full Sunshine Makers composite..."
echo "   Cartoon: $CARTOON"
echo "   TV Frame: $TV_FRAME"
echo "   Commentary: $COMMENTARY"
echo "   Film audio: $FILM_AUDIO"

# Step 1: Loop TV frame to full duration
echo ""
echo "📺 Step 1: Looping TV frame to ${DURATION}s..."
ffmpeg -y -stream_loop -1 -i "$TV_FRAME" -t "$DURATION" \
  -c:v libx264 -crf 18 -preset fast -an \
  "$OUT/tv_frame_looped.mp4" 2>&1 | tail -2

# Step 2: Mix audio - film lowered + commentary
echo ""
echo "🔊 Step 2: Mixing audio..."
ffmpeg -y \
  -i "$FILM_AUDIO" \
  -i "$COMMENTARY" \
  -filter_complex "[0:a]volume=0.15[bg];[bg][1:a]amix=inputs=2:duration=first:dropout_transition=2,volume=1.2[amix]" \
  -map "[amix]" -acodec pcm_s16le -ar 24000 \
  "$OUT/mixed_audio.wav" 2>&1 | tail -2

# Step 3: Composite cartoon into TV frame
echo ""
echo "🎨 Step 3: Compositing video..."
ffmpeg -y \
  -i "$CARTOON" \
  -i "$OUT/tv_frame_looped.mp4" \
  -filter_complex "
    [0:v]scale=472:332:force_original_aspect_ratio=decrease,pad=472:332:(ow-iw)/2:(oh-ih)/2,setsar=1,setpts=PTS-STARTPTS[cartoon];
    [1:v]colorkey=0x00FF00:similarity=0.15:blend=0.2[fg];
    [cartoon][fg]overlay=722:299,scale=1920:1080,setsar=1,format=yuv420p[v]
  " \
  -map "[v]" \
  -c:v libx264 -crf 22 -preset fast \
  -an \
  -shortest -movflags +faststart \
  "$OUT/full_video_nosound.mp4" 2>&1 | tail -3

# Step 4: Mux with mixed audio
echo ""
echo "🔗 Step 4: Muxing final video..."
ffmpeg -y \
  -i "$OUT/full_video_nosound.mp4" \
  -i "$OUT/mixed_audio.wav" \
  -c:v copy -c:a aac -b:a 192k \
  -shortest -movflags +faststart \
  "$OUT/sunshine_makers_full_final.mp4" 2>&1 | tail -3

SIZE=$(du -h "$OUT/sunshine_makers_full_final.mp4" | cut -f1)
echo ""
echo "✅ Done! $OUT/sunshine_makers_full_final.mp4 ($SIZE)"

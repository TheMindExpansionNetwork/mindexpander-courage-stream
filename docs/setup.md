# Detailed Setup Guide

## Prerequisites

- Python 3.11+
- Modal CLI (`pip install modal`)
- FFmpeg/FFplay for audio playback
- OBS Studio for streaming
- HuggingFace account (for model weights)

## Step 1: Clone the Project

```bash
git clone https://github.com/TheMindExpansionNetwork/mindexpander-courage-stream.git
cd mindexpander-courage-stream
pip install requests
```

## Step 2: Set Up Modal

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal setup

# Set your HuggingFace token as a Modal secret
modal secret create huggingface-token HF_TOKEN=hf_your_token_here
```

## Step 3: Upload LoRA Weights

```bash
# If you have local LoRA weights
modal volume put voxcpm-lora ./lora_weights.safetensors /latest/lora_weights.safetensors
modal volume put voxcpm-lora ./lora_config.json /latest/lora_config.json

# Or they may already be on the volume from training
modal volume ls voxcpm-lora /latest/
```

## Step 4: Deploy the Endpoint

```bash
modal deploy modal/mindexpander_voxcpm_modal_app.py

# The output will show your endpoint URL, e.g.:
# https://your-user--mindexpander-voxcpm-api.modal.run
```

## Step 5: Test the Endpoint

```bash
# Health check
curl https://your-endpoint.modal.run/health

# Generate a test audio file
curl -X POST https://your-endpoint.modal.run/v1/audio/speech \
  -H "Authorization: Bearer mindexpander-dev-token" \
  -H "Content-Type: application/json" \
  -d '{"model":"mindexpander-voxcpm2","input":"Hello from the clone!","voice":"default"}' \
  --output test.wav

# Play it
ffplay test.wav
```

## Step 6: Set Up OBS

### Browser Source (TV Overlay)
1. In OBS, click **+** in Sources → **Browser**
2. Name it "Courage TV"
3. Set **Local File** → browse to `overlay/courage-tv.html`
4. Set width/height to match your canvas (e.g., 1920×1080)
5. Check **"Shutdown source when not visible"**

### Video Source
1. Add your video/media source **below** the browser source
2. Position it to fill the screen area of the TV
3. Right-click the browser source → **Blending Mode** → **Normal**

### Audio
1. The TTS audio plays through your system audio
2. In OBS, add an **Audio Output Capture** source to capture it
3. Or use the saved WAV files as media sources

## Step 7: Run a Stream

```bash
# Set your endpoint URL
export COURAGE_TTS_ENDPOINT="https://your-endpoint.modal.run"

# Run the sample script
python scripts/stream_courage.py \
  --script scripts/sample_script.json \
  --save-dir output/

# Or create your own script (see README for format)
```

## Step 8: Cost Management

The Modal endpoint auto-scales to zero when not in use. To stop it manually:

```bash
# Check running tasks
modal app list

# Stop the app
modal app stop mindexpander-voxcpm

# Redeploy when needed
modal deploy modal/mindexpander_voxcpm_modal_app.py
```

## Troubleshooting

### "Model not found" error
- Check that the HF cache volume has the model files
- Run: `modal volume ls huggingface-cache`

### Slow first request (cold start)
- First request takes ~25s for model loading
- Subsequent requests are fast (~1.5-2s)
- Set `min_containers=1` in the Modal app to keep a warm container (costs more)

### Audio quality issues
- Increase `inference_timesteps` (try 12 or 15)
- Adjust `cfg_value` (lower = more natural, higher = more expressive)
- Check that LoRA weights are loaded (check Modal logs)

### OBS overlay not showing
- Make sure the browser source is above your video source in the source list
- Try refreshing the browser source (right-click → Refresh)
- Check that the HTML file path is correct

# 📺 MindExpander Courage TV

**AI Voice Clone + Retro TV Overlay for Live Streaming**

A full-stack project combining real-time AI voice cloning (VoxCPM2 + LoRA) with a
Courage the Cowardly Dog-inspired animated TV overlay for OBS live streams.

## What Is This?

Write a script → AI generates your cloned voice in real-time → Play it over a
creepy old TV overlay while public domain cartoons play on the screen.

Like MST3K, but the narrator is an AI clone of your voice.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  OBS Scene                                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Courage TV Overlay (HTML Browser Source)         │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Video Source (public domain cartoon)       │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │  Scan lines • VHS glitch • Dust • Phosphor glow   │  │
│  └───────────────────────────────────────────────────┘  │
│  Audio: AI Voice Clone (VoxCPM2 + LoRA)                │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Modal Endpoint (A100 80GB)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  MiniCPM4    │→ │  DiT (10     │→ │  Audio VAE   │  │
│  │  (LLM)       │  │  steps, CFG) │  │  (decoder)   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  OpenAI-compatible API: POST /v1/audio/speech           │
│  RTF: 0.641 (1.56x faster than realtime)               │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Deploy the TTS Endpoint

```bash
# Upload LoRA weights to Modal volume
modal volume put voxcpm-lora /path/to/lora_weights.safetensors /latest/lora_weights.safetensors

# Deploy the endpoint
modal deploy modal/mindexpander_voxcpm_modal_app.py
```

### 2. Use the TV Overlay in OBS

1. Add a **Browser Source** in OBS
2. Set URL to the local file: `overlay/courage-tv.html` (or host it)
3. Set dimensions: **1920x1080** (or your canvas size)
4. Add your video source **below** the browser source in the scene

### 3. Run the Stream Script

```bash
# Install dependencies
pip install requests

# Run with the sample script
python scripts/stream_courage.py --script scripts/sample_script.json --save-dir output/

# Custom settings
python scripts/stream_courage.py \
  --script scripts/my_script.json \
  --cfg 2.0 \
  --steps 10 \
  --save-dir output/
```

## TV Overlay Controls

The overlay exposes a JavaScript API via `window.CourageTV`:

```javascript
// Change channel display
CourageTV.setChannel("CH 07");

// Toggle REC indicator
CourageTV.setRecording(true);

// Load video into screen
CourageTV.setVideo("https://example.com/video.mp4");

// Load iframe (YouTube embed)
CourageTV.setIframe("https://www.youtube.com/embed/...");

// Trigger manual glitch effect
CourageTV.glitch();

// Trigger static flash
CourageTV.flash();
```

## Script Format

Scripts are JSON arrays of line objects:

```json
[
  {
    "text": "What the AI voice says",
    "channel": "CH 03",
    "pause_after": 1.0
  }
]
```

Fields:
- `text` (required): The text to synthesize
- `channel`: Channel display on the TV (default: "CH 03")
- `pause_after`: Seconds to pause after this line (default: 0.5)

## Voice Design Tips

VoxCPM2 supports emotion and non-verbal cues in the text:

- **Emotion**: Start text with `(whisper)`, `(excited)`, `(scared)`, etc.
- **Non-verbal**: Use `[laughing]`, `[sigh]`, `[gasp]`, `[Uhm]` inline
- **Parameters**: `cfg_value` (1.0-3.0, default 2.0), `inference_timesteps` (4-30, default 10)

## GPU Performance

| GPU | RTF | Realtime? | Cost/hr of audio |
|---|---|---|---|
| A10G | 1.650 | ❌ | $1.10 |
| A100 40GB | 0.920 | ✅ | $2.21 |
| RTX PRO 6000 | 0.774 | ✅✅ | $2.72 |
| **A100 80GB** | **0.641** | **✅✅✅** | **$2.26** |

**Recommended**: A100 80GB with steps=10, cfg=2.0

## Project Structure

```
mindexpander-courage-stream/
├── README.md
├── LICENSE
├── .gitignore
├── modal/
│   └── mindexpander_voxcpm_modal_app.py   # Modal TTS endpoint
├── overlay/
│   └── courage-tv.html                     # Animated TV overlay
├── scripts/
│   ├── stream_courage.py                   # Stream runner
│   └── sample_script.json                  # Example script
├── docs/
│   └── setup.md                            # Detailed setup guide
└── assets/
    └── images/                             # Generated TV frames
```

## Related Repos

- [TheMindExpansionNetwork/mindexpander_vox_CPM](https://github.com/TheMindExpansionNetwork/mindexpander_vox_CPM) — VoxCPM2 fork with LoRA
- [TheMindExpansionNetwork/M1ND3XPAND3RS-VOICE-VoxCPM2-train-clean](https://huggingface.co/TheMindExpansionNetwork/M1ND3XPAND3RS-VOICE-VoxCPM2-train-clean) — Training dataset
- [TheMindExpansionNetwork/M1ND3XPAND3RS-VOICE-VoxCPM-ready](https://huggingface.co/TheMindExpansionNetwork/M1ND3XPAND3RS-VOICE-VoxCPM-ready) — Voice samples

## License

MIT — Use it, fork it, make weird streams with it.

---

*"Return the slab!" — Courage the Cowardly Dog*

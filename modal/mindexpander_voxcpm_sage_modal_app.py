"""
VoxCPM2 MindExpander — SageAttention-accelerated endpoint on Modal.

Replaces torch.nn.functional.scaled_dot_product_attention with SageAttention
to speed up the MiniCPM-4 LLM backbone inside VoxCPM2. SageAttention 2.x
supports A10G/A100/H100-class GPUs; SageAttention 1.x is for consumer
cards (RTX4090/RTX3090).

Usage:
  # Deploy (creates persistent endpoint):
  modal deploy modal/mindexpander_voxcpm_sage_modal_app.py

  # Or run locally (ephemeral):
  modal run modal/mindexpander_voxcpm_sage_modal_app.py

  # Benchmark vs original:
  python scripts/benchmark_sage.py

Falls back gracefully to stock SDPA if sageattention installation fails.
"""

import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "mindexpander-voxcpm-sage"
HF_MODEL_ID = "openbmb/VoxCPM2"

# Modal volumes — share with original app
LORA_VOLUME = "voxcpm-lora"
HF_CACHE_VOLUME = "huggingface-cache"
OUTPUTS_VOLUME = "outputs"


# ---------------------------------------------------------------------------
# SageAttention monkey-patch utility
# ---------------------------------------------------------------------------

SAGE_PATCH_APPLIED = False
SAGE_PATCH_ERROR = None


def apply_sage_patch():
    """
    Monkey-patch torch.nn.functional.scaled_dot_product_attention
    with sageattention. Returns True on success, False on failure.
    """
    global SAGE_PATCH_APPLIED, SAGE_PATCH_ERROR

    if SAGE_PATCH_APPLIED:
        return True

    try:
        from sageattention import sageattn
        import torch.nn.functional as F

        _original_sdp = F.scaled_dot_product_attention

        def _patched_sdp(query, key, value, attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None, enable_gqa=False):
            """
            SageAttention wrapper matching the torch SDPA signature used by
            MiniCPM-4 / VoxCPM2. SageAttention handles is_causal internally;
            unsupported kwargs (scale, enable_gqa) are ignored.
            """
            try:
                return sageattn(query, key, value, is_causal=is_causal)
            except Exception:
                # Fall back to original on shape / dtype mismatches
                return _original_sdp(
                    query, key, value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )

        # Attach the original so we can restore if needed
        _patched_sdp._original = _original_sdp

        F.scaled_dot_product_attention = _patched_sdp
        SAGE_PATCH_APPLIED = True
        print(f"[sage-patch] SUCCESS — SageAttention monkey-patch applied.", flush=True)
        return True

    except ImportError as e:
        SAGE_PATCH_ERROR = f"ImportError: {e}"
        print(f"[sage-patch] FAILED ImportError: {e}", flush=True)
        return False
    except Exception as e:
        SAGE_PATCH_ERROR = f"{type(e).__name__}: {e}"
        print(f"[sage-patch] FAILED {type(e).__name__}: {e}", flush=True)
        return False


def restore_sdp():
    """Restore the original scaled_dot_product_attention if it was patched."""
    global SAGE_PATCH_APPLIED
    if not SAGE_PATCH_APPLIED:
        return
    try:
        import torch.nn.functional as F
        patched = F.scaled_dot_product_attention
        if hasattr(patched, '_original'):
            F.scaled_dot_product_attention = patched._original
        SAGE_PATCH_APPLIED = False
        print("[sage-patch] SDPA restored to original.")
    except Exception as e:
        print(f"[sage-patch] Could not restore SDPA: {e}")


# ---------------------------------------------------------------------------
# Modal image — includes sageattention on top of the standard VoxCPM image
# ---------------------------------------------------------------------------

def _build_gpu_image():
    """Build the GPU image with SageAttention support."""
    base = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "ffmpeg", "libsndfile1")
        # Core torch + VoxCPM deps
        .pip_install(
            "torch>=2.5.0",
            "torchaudio>=2.5.0",
            "transformers>=4.36.2",
            "einops",
            "inflect",
            "addict",
            "wetext",
            "datasets>=3,<4",
            "huggingface-hub",
            "pydantic",
            "safetensors",
            "soundfile",
            "librosa",
            "numpy<2",
            "accelerate",
            extra_index_url="https://download.pytorch.org/whl/cu121",
        )
        # SageAttention (2.x for A100/H100/A10G, 1.x for consumer cards)
        # Try 2.x first since our GPU is A10G (datacenter-class Ampere)
        .pip_install(
            "triton>=2.3.0",
            "sageattention",
        )
        # VoxCPM itself
        .pip_install(
            "git+https://github.com/OpenBMB/VoxCPM.git",
        )
    )
    return base


gpu_image = _build_gpu_image()

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi", "uvicorn")
)

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# CPU API (routes to GPU)
# ---------------------------------------------------------------------------

def _read_bearer_token():
    return os.getenv("MINDEXPANDER_API_KEY", "mindexpander-dev-token")


@app.function(
    image=cpu_image,
    cpu=0.25,
    memory=512,
    timeout=300,
    scaledown_window=60,
    min_containers=0,
)
@modal.asgi_app(label=f"{APP_NAME}-api")
def api():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response, JSONResponse

    web_app = FastAPI(title="MindExpander VoxCPM2 TTS API (SageAttention)")

    @web_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": HF_MODEL_ID,
            "endpoint": APP_NAME,
            "sage_patched": SAGE_PATCH_APPLIED,
            "lora": True,
            "openai_compatible": "/v1/audio/speech",
        }

    @web_app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": f"{APP_NAME}-v2",
                    "object": "model",
                    "owned_by": "mindexpander",
                    "permission": [],
                }
            ],
        }

    @web_app.post("/v1/audio/speech")
    async def generate_speech(request: Request):
        body = await request.json()
        text = body.get("input", "")
        voice = body.get("voice", "default")
        response_format = body.get("response_format", "wav")
        speed = body.get("speed", 1.0)
        cfg_value = body.get("cfg_value", 2.0)
        inference_timesteps = body.get("inference_timesteps", 10)

        if not text:
            raise HTTPException(status_code=400, detail="input is required")

        runner = VoxCPMSageRunner()
        result = await runner.generate.remote.aio(
            text=text,
            voice=voice,
            response_format=response_format,
            speed=speed,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )

        media_type = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "flac": "audio/flac",
            "pcm": "audio/pcm",
        }.get(response_format, "audio/wav")

        return Response(
            content=result["audio_bytes"],
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{response_format}"',
                "X-Duration-Seconds": str(result.get("duration_seconds", 0)),
                "X-Sample-Rate": str(result.get("sample_rate", 48000)),
                "X-Sage-Patched": str(SAGE_PATCH_APPLIED),
                "X-Generation-Time": str(result.get("generation_time_seconds", 0)),
            },
        )

    @web_app.post("/benchmark")
    async def benchmark(request: Request):
        """
        Benchmark endpoint: generates test text and returns detailed timing.
        Accepts: {"text": "...", "steps": 10, "cfg": 2.0, "warmup_runs": 1}
        Returns: timing breakdown, RTF, sage_status.
        """
        body = await request.json()
        text = body.get("text", "Hello world, this is a benchmark test for SageAttention.")
        steps = body.get("steps", 10)
        cfg = body.get("cfg", 2.0)
        warmup_runs = body.get("warmup_runs", 1)

        runner = VoxCPMSageRunner()
        result = await runner.benchmark.remote.aio(
            text=text,
            steps=steps,
            cfg=cfg,
            warmup_runs=warmup_runs,
        )
        return JSONResponse(content=result)

    return web_app


# ---------------------------------------------------------------------------
# GPU class — SageAttention-patched VoxCPM2 runner
# ---------------------------------------------------------------------------

@app.cls(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=600,
    scaledown_window=120,
    min_containers=0,
    volumes={
        f"/cache/{LORA_VOLUME}": modal.Volume.from_name(LORA_VOLUME, create_if_missing=True),
        f"/cache/hf": modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True),
        "/outputs": modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True),
    },
    secrets=[],
)
class VoxCPMSageRunner:
    """GPU-backed VoxCPM2 with SageAttention acceleration."""

    @modal.enter()
    def load_model(self):
        """Load model once — applies Sage patch before any torch imports."""
        import os as _os
        _os.environ["HF_HOME"] = "/cache/hf"
        _os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
        _os.environ["HF_HUB_CACHE"] = "/cache/hf/hub"
        _os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        # ── Apply SageAttention monkey-patch ──
        self.sage_available = apply_sage_patch()
        self.sage_error = SAGE_PATCH_ERROR

        import torch
        from voxcpm.core import VoxCPM
        from voxcpm.model.voxcpm2 import LoRAConfig

        print(f"[VoxCPMSageRunner] Loading {HF_MODEL_ID} with LoRA...")
        print(f"[VoxCPMSageRunner] SageAttention active: {self.sage_available}")
        start = time.time()

        lora_dir = Path(f"/cache/{LORA_VOLUME}/latest")
        lora_config_path = lora_dir / "lora_config.json"
        lora_weights_path = lora_dir / "lora_weights.safetensors"

        lora_cfg = None
        lora_path = None

        if lora_weights_path.exists():
            print(f"[VoxCPMSageRunner] Found LoRA weights: {lora_weights_path}")
            if lora_config_path.exists():
                with open(lora_config_path) as f:
                    cfg_dict = json.load(f)
                lora_cfg = LoRAConfig(**cfg_dict.get("lora_config", {
                    "enable_lm": True,
                    "enable_dit": True,
                    "enable_proj": False,
                    "r": 32,
                    "alpha": 16,
                }))
            else:
                lora_cfg = LoRAConfig(
                    enable_lm=True, enable_dit=True,
                    enable_proj=False, r=32, alpha=16,
                )
            lora_path = str(lora_dir)
            print(f"[VoxCPMSageRunner] LoRA config: r={lora_cfg.r}, alpha={lora_cfg.alpha}")
        else:
            print(f"[VoxCPMSageRunner] No LoRA weights at {lora_weights_path}")
            for alt in [
                Path(f"/cache/{LORA_VOLUME}/lora_weights.safetensors"),
                Path(f"/cache/{LORA_VOLUME}/checkpoints/latest/lora_weights.safetensors"),
            ]:
                if alt.exists():
                    print(f"[VoxCPMSageRunner] Found LoRA at alt path: {alt}")
                    lora_path = str(alt.parent)
                    lora_cfg = LoRAConfig(enable_lm=True, enable_dit=True, enable_proj=False, r=32, alpha=16)
                    break

        self.model = VoxCPM.from_pretrained(
            hf_model_id=HF_MODEL_ID,
            load_denoiser=False,
            optimize=False,
            lora_config=lora_cfg,
            lora_weights_path=lora_path,
        )

        elapsed = time.time() - start
        self.sample_rate = self.model.tts_model.sample_rate
        print(f"[VoxCPMSageRunner] Model loaded in {elapsed:.1f}s, sample_rate={self.sample_rate}")
        print(f"[VoxCPMSageRunner] LoRA enabled: {self.model.lora_enabled}")

    @modal.method()
    def generate(
        self,
        text: str,
        voice: str = "default",
        response_format: str = "wav",
        speed: float = 1.0,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ) -> dict:
        """Generate speech from text."""
        import soundfile as sf

        start = time.time()

        audio_np = self.model.generate(
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=False,
            denoise=False,
        )

        duration = len(audio_np) / self.sample_rate
        elapsed = time.time() - start
        rtf = elapsed / duration if duration > 0 else float("inf")
        print(f"[VoxCPMSageRunner] {duration:.1f}s audio in {elapsed:.1f}s (RTF: {rtf:.3f}) — {text[:80]}")

        buf = io.BytesIO()
        if response_format == "wav":
            sf.write(buf, audio_np, self.sample_rate, format="WAV")
        elif response_format == "flac":
            sf.write(buf, audio_np, self.sample_rate, format="FLAC")
        elif response_format in ("mp3", "opus"):
            sf.write(buf, audio_np, self.sample_rate, format="WAV")
            buf.seek(0)
            wav_data = buf.read()
            import subprocess
            fmt = "mp3" if response_format == "mp3" else "opus"
            proc = subprocess.run(
                ["ffmpeg", "-i", "pipe:0", "-f", fmt, "pipe:1"],
                input=wav_data, capture_output=True,
            )
            if proc.returncode == 0:
                audio_bytes = proc.stdout
            else:
                audio_bytes = wav_data
                response_format = "wav"
        else:
            sf.write(buf, audio_np, self.sample_rate, format="WAV")

        audio_bytes = buf.getvalue() if response_format in ("wav", "flac") else audio_bytes

        return {
            "audio_bytes": audio_bytes,
            "duration_seconds": round(duration, 2),
            "sample_rate": self.sample_rate,
            "format": response_format,
            "generation_time_seconds": round(elapsed, 2),
            "rtf": round(rtf, 4),
            "sage_active": self.sage_available,
        }

    @modal.method()
    def benchmark(
        self,
        text: str = "Hello world, this is a benchmark test for SageAttention acceleration.",
        steps: int = 10,
        cfg: float = 2.0,
        warmup_runs: int = 1,
    ) -> dict:
        """Run a timed benchmark and return detailed metrics."""
        import time as _t

        # Warmup
        for _ in range(warmup_runs):
            _ = self.model.generate(
                text=text,
                cfg_value=cfg,
                inference_timesteps=steps,
                normalize=False,
                denoise=False,
            )

        # Timed run
        start = _t.perf_counter()
        audio_np = self.model.generate(
            text=text,
            cfg_value=cfg,
            inference_timesteps=steps,
            normalize=False,
            denoise=False,
        )
        elapsed = _t.perf_counter() - start

        duration = len(audio_np) / self.sample_rate
        rtf = elapsed / duration if duration > 0 else float("inf")

        return {
            "text": text,
            "text_length_chars": len(text),
            "audio_duration_seconds": round(duration, 2),
            "generation_time_seconds": round(elapsed, 3),
            "rtf": round(rtf, 4),
            "steps": steps,
            "cfg": cfg,
            "warmup_runs": warmup_runs,
            "sample_rate": self.sample_rate,
            "sage_active": self.sage_available,
            "sage_error": self.sage_error,
            "gpu_memory_allocated_mb": _get_gpu_memory(),
        }


# ---------------------------------------------------------------------------
# Standalone GPU function (for direct .remote() calls)
# ---------------------------------------------------------------------------

@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=600,
    scaledown_window=120,
    min_containers=0,
    volumes={
        f"/cache/{LORA_VOLUME}": modal.Volume.from_name(LORA_VOLUME, create_if_missing=True),
        f"/cache/hf": modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True),
        "/outputs": modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True),
    },
)
def generate_audio(
    text: str,
    voice: str = "default",
    response_format: str = "wav",
    speed: float = 1.0,
) -> dict:
    """Standalone GPU function for generating audio."""
    runner = VoxCPMSageRunner()
    runner.load_model()
    return runner.generate(text=text, voice=voice, response_format=response_format, speed=speed)


# ---------------------------------------------------------------------------
# Model caching
# ---------------------------------------------------------------------------

@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=600,
    volumes={
        f"/cache/{LORA_VOLUME}": modal.Volume.from_name(LORA_VOLUME, create_if_missing=True),
        f"/cache/hf": modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True),
    },
)
def cache_model():
    """Pre-download VoxCPM2 model to HF cache volume."""
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    os.environ["HF_HUB_CACHE"] = "/cache/hf/hub"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    from pathlib import Path
    from huggingface_hub import snapshot_download

    print(f"[cache_model] Downloading {HF_MODEL_ID} to /cache/hf/hub...")
    path = snapshot_download(
        repo_id=HF_MODEL_ID,
        cache_dir="/cache/hf/hub",
        ignore_patterns=["*.md", "*.txt"],
    )
    print(f"[cache_model] Downloaded to: {path}")

    p = Path(path)
    for f in ["model.safetensors", "audiovae.pth", "config.json"]:
        fp = p / f
        if fp.exists():
            print(f"  ✅ {f}: {fp.stat().st_size / 1024 / 1024:.1f}MB")
        else:
            print(f"  ❌ {f}: MISSING")

    vol = modal.Volume.from_name(HF_CACHE_VOLUME)
    vol.commit()
    print("[cache_model] Volume committed.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_gpu_memory() -> Optional[float]:
    """Return allocated GPU memory in MB, or None if torch not available."""
    try:
        import torch
        return round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Local entrypoint — quick smoke test
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    text: str = "Is this real, or did the stars just blink? Did the room start breathing when I stopped to think?",
    voice: str = "default",
    cache: bool = False,
    benchmark: bool = False,
):
    """Test the Sage-accelerated endpoint locally."""
    if cache:
        print("Pre-caching VoxCPM2 model...")
        cache_model.remote()
        print("Done!")
        return

    if benchmark:
        print("Running benchmark...")
        runner = VoxCPMSageRunner()
        runner.load_model()
        result = runner.benchmark.remote(text=text, steps=10, cfg=2.0, warmup_runs=1)
        print(json.dumps(result, indent=2))
        return

    print(f"Generating (Sage): {text}")
    result = generate_audio.remote(text=text, voice=voice)
    out_path = Path("/tmp/mindexpander_sage_test.wav")
    out_path.write_bytes(result["audio_bytes"])
    print(f"Saved: {out_path} ({result['duration_seconds']}s, {result['sample_rate']}Hz)")
    print(f"Generation time: {result['generation_time_seconds']}s")
    print(f"SageAttention active: {result.get('sage_active', False)}")

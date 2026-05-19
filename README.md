<p align="center">
  <img src="docs/assets/banner.png" alt="SpectralQuant Ollama Bridge" width="800" />
</p>

<p align="center">
  <strong>A transparent bridge bringing SpectralQuant's massive KV cache compression and 4-bit NF4 weight quantization to any Ollama-compatible frontend.</strong>
</p>

[![Version](https://img.shields.io/github/package-json/v/kruschdev/spectralquant-ollama-bridge.svg)](https://github.com/kruschdev/spectralquant-ollama-bridge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Node](https://img.shields.io/badge/Node.js-22+-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Compatible-blue.svg)

---

## The Problem

Large Language Models require massive amounts of RAM for their KV cache during long-context inference. While engines like **SpectralQuant** solve this on the backend with advanced PyTorch compression (reducing KV cache memory by up to 10x), most existing UI frontends (like OpenWebUI or AnythingLLM) are hardcoded to talk to standard **Ollama** endpoints.

**SpectralQuant ↔ Ollama Bridge fixes this.** It acts as a transparent translation layer, allowing any Ollama-compatible frontend to seamlessly leverage SpectralQuant's backend optimizations with zero code modifications.

## What It Does

A standalone Express proxy exposing standard Ollama endpoints, translating them on-the-fly to OpenAI-compatible requests:

| Capability | What It Provides |
|-----------|-----------------|
| 🔄 **Endpoint Translation** | Converts `/api/chat` and `/api/generate` Ollama JSON requests to OpenAI `fetch` requests. |
| 🌊 **Stream Conversion** | Parses true Server-Sent Events (SSE) from the backend and converts them into Ollama NDJSON streams. |
| 🎭 **Endpoint Mocking** | Mocks `/api/tags` so existing interfaces load cleanly without crashing. |
| 🪶 **Minimal Dependencies** | Runs purely on Express and CORS. No heavy SDKs or external tooling required. |

## Why You'd Want It

**🛡️ Keep your existing tools** — Continue using OpenWebUI, AnythingLLM, or any other Ollama frontend you already love.

**🧠 Compounding Memory Savings** — Get a synergistic memory discount: native 4-bit (NF4) quantization shrinks the static model weights, while SpectralQuant compresses the dynamic KV cache by up to 10x. This allows you to fit smarter models with massive context windows onto single consumer GPUs (like an RTX 3060 or 2080 Ti).

**⚡ Run side-by-side** — Exposes the proxy on port `11437` (a non-colliding port), so it runs peacefully alongside your default Ollama instance.

---

## Quick Start

The repository is a "batteries-included" package containing both the **SpectralQuant PyTorch Backend** and the **Node.js Bridge**.

**Prerequisites:** [Docker](https://www.docker.com/) & [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU acceleration)

```bash
# 1. Clone the repository
git clone https://github.com/kruschdev/spectralquant-ollama-bridge.git
cd spectralquant-ollama-bridge

# 2. Launch both the PyTorch backend and Node.js bridge
docker compose up -d --build
```

You can view the logs of the bridge to confirm it connected successfully:
```bash
docker compose logs -f spectralquant-bridge
```

You should see:
```text
======================================================
🚀 SpectralQuant ↔ Ollama Bridge
📡 Listening on Port: 11437
🔗 Target Backend: http://spectralquant-server:11436
🤖 Mock Model Name: spectralquant:latest
======================================================
```

---

## Configuration

You can customize the bridge's behavior by passing environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | The port the Node.js bridge listens on. | `11437` |
| `SPECTRALQUANT_URL` | The URL of the SpectralQuant Python proxy. | `http://127.0.0.1:11436` |
| `MOCK_MODEL_NAME` | The fake model name exposed to Ollama clients. | `spectralquant:latest` |
| `LOAD_IN_4BIT` | Enable bitsandbytes NF4 quantization for the base model weights, reducing VRAM usage while preserving KV compression. | `true` |
| `LOAD_IN_8BIT` | Enable bitsandbytes INT8 quantization. If both are set, 4-bit takes precedence. | `false` |
| `MODEL_NAME` | The HuggingFace model ID to load on the backend. | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `HF_TOKEN` | Your HuggingFace token for gated models. | *None* |

---

## Usage Examples

### Testing with cURL

Point `curl` at your new bridge just like you would with Ollama:
```bash
curl http://localhost:11437/api/generate -d '{
  "model": "spectralquant:latest",
  "prompt": "Why is the sky blue?"
}'
```

### Using OpenWebUI

Simply add `http://localhost:11437` as an additional Ollama endpoint in your OpenWebUI settings. It will natively pick up `spectralquant:latest` as an available model!

### Running 8-Bit Calibration

If you wish to use 8-bit (INT8) precision with `LOAD_IN_8BIT=true`, you should generate the 8-bit specific eigenspectral matrices so the KV cache compression aligns with the quantized weights:

```bash
# Run this inside the backend container or a local python environment with requirements installed
python backend/calibrate_8bit.py --model Qwen/Qwen2.5-Coder-7B-Instruct
```
This will output the required `.pt` files to `backend/calibration_data/`.

---

## Acknowledgments

This bridge acts as a frontend translation layer for the core **SpectralQuant** PyTorch engine. 

All credit for the underlying KV cache compression breakthroughs—specifically the discovery of universal structural properties in key vectors—belongs to **Ashwin Gopinath** and his paper: 
> *"3% Is All You Need: Breaking TurboQuant's Compression Limit via Spectral Structure"*

You can find the original core engine repository at [Dynamis-Labs/spectralquant](https://github.com/Dynamis-Labs/spectralquant).

---

## Compatibility Updates

**v1.1.2:**
- **8-Bit (INT8) Quantization:** Added support for `LOAD_IN_8BIT=true` to enable INT8 quantization via bitsandbytes, and included a dedicated `calibrate_8bit.py` script.

**v1.1.1:**
- **Infrastructure Stability:** Added `docker-compose` health checks, `always` restart policies, and enforced startup ordering (`depends_on: service_healthy`) to ensure the Node.js bridge waits for the PyTorch backend to fully initialize.
- **4-Bit (NF4) Quantization:** Native support for loading backend models in 4-bit precision via `bitsandbytes`, massively reducing baseline VRAM requirements while keeping the SpectralQuant KV cache intact. (Configurable via `LOAD_IN_4BIT=true`).

**v1.1.0:**
- **Multi-GPU Support:** The backend proxy now fully supports `device_map="auto"` via accelerate, allowing SpectralQuant states and centroids to dynamically migrate to the correct active device.
- **bfloat16 Compatibility:** Resolved NaN/Inf device-side overflow assertions by preserving native `bfloat16` precision for models like `Qwen2.5-Coder-7B-Instruct`.

## Contributing

We welcome contributions! Please ensure code remains lightweight with minimal dependencies.

## License

MIT License © 2026 [kruschdev](https://github.com/kruschdev)

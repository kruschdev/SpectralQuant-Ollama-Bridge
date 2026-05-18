<p align="center">
  <img src="docs/assets/banner.png" alt="SpectralQuant Ollama Bridge" width="800" />
</p>

<p align="center">
  <strong>A lightweight, zero-dependency Node.js bridge that brings SpectralQuant's massive KV cache compression to native Ollama workflows.</strong>
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
| 🪶 **Zero Dependencies** | Runs purely on Express and CORS. No heavy SDKs or external tooling required. |

## Why You'd Want It

**🛡️ Keep your existing tools** — Continue using OpenWebUI, AnythingLLM, or any other Ollama frontend you already love.

**🧠 Massive memory savings** — Unlock up to 10x KV cache reduction for long-context tasks by routing through SpectralQuant.

**⚡ Run side-by-side** — Exposes the proxy on port `11437` (a non-colliding port), so it runs peacefully alongside your default Ollama instance.

---

## Quick Start

**Prerequisites:** [Node.js 22+](https://nodejs.org/) · SpectralQuant PyTorch Engine running (default `http://127.0.0.1:11436`)

```bash
# 1. Clone and install
git clone https://github.com/kruschdev/spectralquant-ollama-bridge.git
cd spectralquant-ollama-bridge
npm install

# 2. Start the bridge
npm start
```

You should see:
```text
======================================================
🚀 SpectralQuant ↔ Ollama Bridge
📡 Listening on Port: 11437
🔗 Target Backend: http://127.0.0.1:11436
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

---

## Contributing

We welcome contributions! Please ensure code remains lightweight with minimal dependencies.

## License

MIT License © 2026 [kruschdev](https://github.com/kruschdev)

import express from 'express';
import cors from 'cors';
import { setGlobalDispatcher, Agent } from 'undici';
import { Readable } from 'stream';

// Configure high-performance global agent pooling
const globalAgent = new Agent({
    keepAliveTimeout: 10 * 60 * 1000, // 10 minutes keep-alive
    keepAliveMaxTimeout: 15 * 60 * 1000,
    connections: 100, // Max concurrent sockets in the pool
    pipelining: 1
});
setGlobalDispatcher(globalAgent);

const app = express();
app.use(cors());
app.use(express.json());

// Request logging middleware
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
    next();
});

const PORT = process.env.PORT || 11437;
const SPECTRALQUANT_URL = process.env.SPECTRALQUANT_URL || 'http://127.0.0.1:11436';
const MOCK_MODEL_NAME = process.env.MOCK_MODEL_NAME || 'spectralquant:latest';
const REWRITE_MODEL_TO = process.env.REWRITE_MODEL_TO || '';

// --- Lightweight Input Validation Helpers ---
function validateMessages(messages) {
    if (!messages || !Array.isArray(messages)) {
        return "messages field must be a valid array";
    }
    if (messages.length === 0) {
        return "messages array cannot be empty";
    }
    for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (!msg || typeof msg !== 'object') {
            return `message at index ${i} must be a valid object`;
        }
        if (typeof msg.role !== 'string' || !msg.role.trim()) {
            return `message at index ${i} must contain a valid string 'role'`;
        }
        if (typeof msg.content !== 'string' || !msg.content.trim()) {
            return `message at index ${i} must contain a valid string 'content'`;
        }
    }
    return null;
}

function validateGenerate(prompt) {
    if (typeof prompt !== 'string' || !prompt.trim()) {
        return "prompt field must be a non-empty string";
    }
    return null;
}

console.log(`\n======================================================`);
console.log(`🚀 SpectralQuant ↔ Ollama Bridge`);
console.log(`📡 Listening on Port: ${PORT}`);
console.log(`🔗 Target Backend: ${SPECTRALQUANT_URL}`);
console.log(`🤖 Mock Model Name: ${MOCK_MODEL_NAME}`);
console.log(`🔄 Rewrite Model To: ${REWRITE_MODEL_TO || 'None (Pass through)'}`);
console.log(`======================================================\n`);

// --- Health Check ---
app.get('/health', async (req, res) => {
    try {
        // Attempt to reach the backend to verify it is up
        const response = await fetch(`${SPECTRALQUANT_URL}/`);
        if (response.ok || response.status === 404) {
            res.json({ status: 'healthy', backend: 'reachable', code: response.status });
        } else {
            res.status(503).json({ status: 'degraded', backend: 'unreachable', details: response.statusText });
        }
    } catch (e) {
        res.status(503).json({ status: 'down', backend: 'unreachable', error: e.message });
    }
});

// --- /api/tags ---
// Mocks the Ollama model list so UIs like OpenWebUI don't crash
app.get('/api/tags', (req, res) => {
    res.json({
        models: [
            {
                name: MOCK_MODEL_NAME,
                modified_at: new Date().toISOString(),
                size: 4000000000,
                digest: 'sha256:spectralquant_bridge_mock',
                details: {
                    format: 'gguf',
                    family: 'llama',
                    families: ['llama'],
                    parameter_size: '7B',
                    quantization_level: 'Q4_0'
                }
            }
        ]
    });
});

// --- /api/chat ---
app.post('/api/chat', async (req, res) => {
    const controller = new AbortController();
    res.on('close', () => {
        console.log(`[CONN CLOSE] Client closed connection for /api/chat. Aborting backend request.`);
        controller.abort();
    });

    try {
        const { model, messages, stream = true } = req.body;
        
        const valErr = validateMessages(messages);
        if (valErr) {
            console.warn(`[VALIDATION WARN] Bad Request on /api/chat: ${valErr}`);
            res.status(400).json({ error: "Bad Request", details: valErr });
            return;
        }
        
        const targetModel = REWRITE_MODEL_TO || model || MOCK_MODEL_NAME;
        const openaiReq = {
            model: targetModel,
            messages: messages,
            stream: stream
        };

        const response = await fetch(`${SPECTRALQUANT_URL}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(openaiReq),
            signal: controller.signal
        });

        if (!response.ok) {
            const errText = await response.text();
            if (response.status === 429) {
                res.status(429).json({ error: "Backend rate limit exceeded", details: errText });
                return;
            }
            throw new Error(`Backend Error ${response.status}: ${errText}`);
        }

        if (stream) {
            res.setHeader('Content-Type', 'application/x-ndjson');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');
            res.setHeader('X-Accel-Buffering', 'no');
            
            // Node 18+ fetch returns a web ReadableStream
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop() || "";
                
                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const dataStr = line.slice(6).trim();
                        if (dataStr === "[DONE]") {
                            res.write(JSON.stringify({
                                model: model || MOCK_MODEL_NAME,
                                created_at: new Date().toISOString(),
                                message: { role: 'assistant', content: '' },
                                done: true,
                                done_reason: 'stop',
                                total_duration: 1000000000,
                                load_duration: 1000000,
                                prompt_eval_count: 0,
                                eval_count: 0,
                                eval_duration: 1000000000
                            }) + '\n');
                            res.end();
                            return;
                        }
                        
                        try {
                            const data = JSON.parse(dataStr);
                            const content = data.choices?.[0]?.delta?.content;
                            if (content) {
                                res.write(JSON.stringify({
                                    model: model || MOCK_MODEL_NAME,
                                    created_at: new Date().toISOString(),
                                    message: { role: 'assistant', content: content },
                                    done: false
                                }) + '\n');
                            }
                        } catch (e) {
                            console.error("SSE Parse error:", e.message, "Data:", dataStr);
                        }
                    }
                }
            }
            res.end();
        } else {
            const data = await response.json();
            const content = data.choices?.[0]?.message?.content || '';
            res.json({
                model: model || MOCK_MODEL_NAME,
                created_at: new Date().toISOString(),
                message: { role: 'assistant', content: content },
                done: true,
                done_reason: 'stop',
                total_duration: 1000000000,
                load_duration: 1000000,
                prompt_eval_count: data.usage?.prompt_tokens || 0,
                eval_count: data.usage?.completion_tokens || 0,
                eval_duration: 1000000000
            });
        }
    } catch (e) {
        console.error(`[CHAT ERROR]`, e.message);
        if (!res.headersSent) {
            res.status(500).json({ error: e.message });
        }
    }
});

// --- /api/generate ---
app.post('/api/generate', async (req, res) => {
    const controller = new AbortController();
    res.on('close', () => {
        console.log(`[CONN CLOSE] Client closed connection for /api/generate. Aborting backend request.`);
        controller.abort();
    });

    try {
        const { model, prompt, stream = true } = req.body;
        
        const valErr = validateGenerate(prompt);
        if (valErr) {
            console.warn(`[VALIDATION WARN] Bad Request on /api/generate: ${valErr}`);
            res.status(400).json({ error: "Bad Request", details: valErr });
            return;
        }
        
        const targetModel = REWRITE_MODEL_TO || model || MOCK_MODEL_NAME;
        const openaiReq = {
            model: targetModel,
            messages: [{ role: 'user', content: prompt }],
            stream: stream
        };

        const response = await fetch(`${SPECTRALQUANT_URL}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(openaiReq),
            signal: controller.signal
        });

        if (!response.ok) {
            const errText = await response.text();
            if (response.status === 429) {
                res.status(429).json({ error: "Backend rate limit exceeded", details: errText });
                return;
            }
            throw new Error(`Backend Error ${response.status}: ${errText}`);
        }

        if (stream) {
            res.setHeader('Content-Type', 'application/x-ndjson');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');
            res.setHeader('X-Accel-Buffering', 'no');
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop() || "";
                
                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const dataStr = line.slice(6).trim();
                        if (dataStr === "[DONE]") {
                            res.write(JSON.stringify({
                                model: model || MOCK_MODEL_NAME,
                                created_at: new Date().toISOString(),
                                response: '',
                                done: true,
                                done_reason: 'stop',
                                context: []
                            }) + '\n');
                            res.end();
                            return;
                        }
                        
                        try {
                            const data = JSON.parse(dataStr);
                            const content = data.choices?.[0]?.delta?.content;
                            if (content) {
                                res.write(JSON.stringify({
                                    model: model || MOCK_MODEL_NAME,
                                    created_at: new Date().toISOString(),
                                    response: content,
                                    done: false
                                }) + '\n');
                            }
                        } catch (e) {
                            console.error("SSE Parse error:", e.message, "Data:", dataStr);
                        }
                    }
                }
            }
            res.end();
        } else {
            const data = await response.json();
            const content = data.choices?.[0]?.message?.content || '';
            res.json({
                model: model || MOCK_MODEL_NAME,
                created_at: new Date().toISOString(),
                response: content,
                done: true,
                done_reason: 'stop',
                context: []
            });
        }
    } catch (e) {
        console.error(`[GENERATE ERROR]`, e.message);
        if (!res.headersSent) {
            res.status(500).json({ error: e.message });
        }
    }
});

const FALLBACK_OLLAMA_URL = process.env.FALLBACK_OLLAMA_URL || 'http://127.0.0.1:11434';

// --- /api/embeddings ---
app.post('/api/embeddings', async (req, res) => {
    try {
        const response = await fetch(`${FALLBACK_OLLAMA_URL}/api/embeddings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body)
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Fallback Ollama Error ${response.status}: ${errText}`);
        }

        const data = await response.json();
        res.json(data);
    } catch (e) {
        console.error(`[EMBEDDINGS ERROR]`, e.message);
        res.status(500).json({ error: e.message });
    }
});


// --- /v1/chat/completions ---
app.post('/v1/chat/completions', async (req, res) => {
    const controller = new AbortController();
    res.on('close', () => {
        console.log(`[CONN CLOSE] Client closed connection for /v1/chat/completions. Aborting backend request.`);
        controller.abort();
    });

    try {
        const body = { ...req.body };
        
        const valErr = validateMessages(body.messages);
        if (valErr) {
            console.warn(`[VALIDATION WARN] Bad Request on /v1/chat/completions: ${valErr}`);
            res.status(400).json({ error: "Bad Request", details: valErr });
            return;
        }
        
        if (REWRITE_MODEL_TO && (body.model === MOCK_MODEL_NAME || !body.model || body.model === 'spectralquant:latest')) {
            body.model = REWRITE_MODEL_TO;
        }
        console.log(`[v1/chat/completions] Forwarding to Backend: ${SPECTRALQUANT_URL}/v1/chat/completions (Model: ${body.model})`);
        const response = await fetch(`${SPECTRALQUANT_URL}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal
        });

        if (!response.ok) {
            const errText = await response.text();
            if (response.status === 429) {
                res.status(429).json({ error: "Backend rate limit exceeded", details: errText });
                return;
            }
            throw new Error(`Backend Error ${response.status}: ${errText}`);
        }

        res.setHeader('Content-Type', response.headers.get('content-type') || 'application/json');
        if (body.stream) {
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');
            res.setHeader('X-Accel-Buffering', 'no');
        }

        Readable.from(response.body).pipe(res);
    } catch (e) {
        console.error(`[v1/chat/completions ERROR]`, e.message);
        if (!res.headersSent) {
            res.status(500).json({ error: e.message });
        }
    }
});


app.listen(PORT, '0.0.0.0', () => {
    console.log(`🟢 Bridge active on 0.0.0.0:${PORT}`);
});

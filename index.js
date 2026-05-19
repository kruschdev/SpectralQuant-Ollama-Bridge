import express from 'express';
import cors from 'cors';

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

console.log(`\n======================================================`);
console.log(`🚀 SpectralQuant ↔ Ollama Bridge`);
console.log(`📡 Listening on Port: ${PORT}`);
console.log(`🔗 Target Backend: ${SPECTRALQUANT_URL}`);
console.log(`🤖 Mock Model Name: ${MOCK_MODEL_NAME}`);
console.log(`======================================================\n`);

// --- Health Check ---
app.get('/health', async (req, res) => {
    try {
        // Attempt to reach the backend to verify it is up
        const response = await fetch(`${SPECTRALQUANT_URL}/v1/models`);
        if (response.ok) {
            res.json({ status: 'healthy', backend: 'reachable' });
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
    try {
        const { model, messages, stream = true } = req.body;
        
        const openaiReq = {
            model: model || MOCK_MODEL_NAME,
            messages: messages,
            stream: stream
        };

        const response = await fetch(`${SPECTRALQUANT_URL}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(openaiReq)
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
        res.status(500).json({ error: e.message });
    }
});

// --- /api/generate ---
app.post('/api/generate', async (req, res) => {
    try {
        const { model, prompt, stream = true } = req.body;
        
        const openaiReq = {
            model: model || MOCK_MODEL_NAME,
            messages: [{ role: 'user', content: prompt }],
            stream: stream
        };

        const response = await fetch(`${SPECTRALQUANT_URL}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(openaiReq)
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
        res.status(500).json({ error: e.message });
    }
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`🟢 Bridge active on 0.0.0.0:${PORT}`);
});

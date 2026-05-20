import os
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import threading
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Environment configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")
AVG_BITS = float(os.environ.get("AVG_BITS", "6.0"))
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "false").lower() in ("true", "1", "yes")
LOAD_IN_8BIT = os.environ.get("LOAD_IN_8BIT", "false").lower() in ("true", "1", "yes")
ENABLE_COMPRESSION = os.environ.get("ENABLE_COMPRESSION", "false").lower() in ("true", "1", "yes")
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "false").lower() in ("true", "1", "yes")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

model = None
tokenizer = None
engine_manager = None

import json
from threading import Thread
from transformers import TextIteratorStreamer
from fastapi.responses import StreamingResponse

class StopOnSignal(StoppingCriteria):
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.stop_event.is_set()

# OpenAI Compatible Request Schemas
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, engine_manager
    logger.info(f"Loading model {MODEL_NAME} onto {DEVICE}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)
        load_kwargs = {
            "device_map": "auto", "local_files_only": LOCAL_FILES_ONLY,
            "attn_implementation": "sdpa",
        }
        is_pre_quantized = any(q in MODEL_NAME.lower() for q in ["awq", "gptq", "exl2"])
        if is_pre_quantized:
            logger.info(f"Pre-quantized model format detected ({MODEL_NAME}). Bypassing bitsandbytes config to use native pre-quantized kernels.")
        elif LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Loading model in 4-bit quantization (bitsandbytes NF4)")
        elif LOAD_IN_8BIT:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            logger.info("Loading model in 8-bit quantization (bitsandbytes INT8)")
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
        model.eval()
        logger.info("Model loaded successfully.")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        yield
        return
    
    # Engine initialization — separate try block so model errors don't mask engine errors
    if ENABLE_COMPRESSION:
        try:
            from pathlib import Path
            calib_dir = Path(__file__).parent / "calibration_data"
            model_base = MODEL_NAME.split("/")[-1]
            # Use 4-bit calibration data when running quantized
            if LOAD_IN_4BIT:
                base_name = f"{model_base}_4bit_wikitext103"
                # Fall back to standard calibration if 4-bit not available
                pt_file = calib_dir / (base_name + ".pt")
                if not pt_file.exists():
                    logger.warning("No 4-bit calibration found, falling back to bf16 calibration")
                    base_name = f"{model_base}_wikitext103"
            elif LOAD_IN_8BIT:
                base_name = f"{model_base}_8bit_wikitext103"
                # Fall back to standard calibration if 8-bit not available
                pt_file = calib_dir / (base_name + ".pt")
                if not pt_file.exists():
                    logger.warning("No 8-bit calibration found, falling back to bf16 calibration")
                    base_name = f"{model_base}_wikitext103"
            else:
                base_name = f"{model_base}_wikitext103"
            pt_file = calib_dir / (base_name + ".pt")
            json_file = calib_dir / (base_name + "_meta.json")
            
            logger.info(f"Looking for calibration: {pt_file}")
            
            if pt_file.exists() and json_file.exists():
                import json
                with open(json_file) as f:
                    meta = json.load(f)
                tensors = torch.load(pt_file, weights_only=True)
                logger.info(f"Loaded {len(tensors)} tensors, {len(meta)} meta entries")
                
                calibration_data = {}
                for item in meta:
                    l, h = item["layer_idx"], item["head_idx"]
                    if l not in calibration_data: calibration_data[l] = {}
                    if h not in calibration_data[l]: calibration_data[l][h] = {}
                    prefix = f"L{l}_H{h}_{item['head_type']}"
                    if item['head_type'] == "key":
                        calibration_data[l][h]["key_eigenvectors"] = tensors[f"{prefix}_eigenvectors"]
                        calibration_data[l][h]["key_eigenvalues"] = tensors[f"{prefix}_eigenvalues"]
                        calibration_data[l][h]["key_d_eff"] = item["d_eff"]
                    else:
                        calibration_data[l][h]["val_eigenvectors"] = tensors[f"{prefix}_eigenvectors"]
                        calibration_data[l][h]["val_eigenvalues"] = tensors[f"{prefix}_eigenvalues"]
                        calibration_data[l][h]["val_d_eff"] = item["d_eff"]
                
                from spectralquant_cache import SpectralQuantEngineManager
                engine_manager = SpectralQuantEngineManager(calibration_data, avg_bits=AVG_BITS, device=str(model.device))
                logger.info(f"SpectralQuant Engine Manager initialized (avg_bits={AVG_BITS}, {len(engine_manager.key_engines)} heads)")
            else:
                logger.warning(f"Calibration data not found at {pt_file}, KV Cache compression disabled.")
            
        except Exception as e:
            logger.error(f"SpectralQuant engine init failed (inference still works without compression): {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.info("KV Cache compression disabled (ENABLE_COMPRESSION=false). Running inference-only gateway.")
        
    yield
    
    # Cleanup on shutdown
    model = None
    tokenizer = None
    engine_manager = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="SpectralQuant Inference Proxy", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
        
    # Format messages using the tokenizer's chat template
    messages_dicts = [{"role": msg.role, "content": msg.content} for msg in req.messages]
    try:
        prompt_text = tokenizer.apply_chat_template(messages_dicts, add_generation_prompt=True, tokenize=False)
    except Exception as e:
        logger.warning(f"Failed to apply chat template ({e}), falling back to raw conversation formatting.")
        prompt_text = ""
        for msg in messages_dicts:
            role = msg["role"].capitalize()
            content = msg["content"]
            prompt_text += f"\n\n{role}: {content}"
        prompt_text += "\n\nAssistant: "
    
    input_device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(input_device)
    
    # Generate
    with torch.no_grad():
        from spectralquant_cache import SpectralQuantCache
        past_key_values = None
        if engine_manager is not None:
            past_key_values = SpectralQuantCache(engine_manager)
            
        stop_event = threading.Event()
        
        async def monitor_disconnect():
            while not stop_event.is_set():
                if await request.is_disconnected():
                    logger.info("Client disconnected. Aborting generation.")
                    stop_event.set()
                    break
                await asyncio.sleep(0.1)
        
        loop = asyncio.get_running_loop()
        monitor_task = loop.create_task(monitor_disconnect())
        
        if getattr(req, "stream", False):
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **inputs, 
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                do_sample=req.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                past_key_values=past_key_values,
                streamer=streamer,
                stopping_criteria=StoppingCriteriaList([StopOnSignal(stop_event)])
            )
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            async def sse_generator():
                import queue
                try:
                    while True:
                        if stop_event.is_set():
                            break
                        try:
                            text = streamer.queue.get_nowait()
                            if text is None:
                                break
                            if text:
                                chunk = {
                                    "id": "chatcmpl-spectralquant",
                                    "object": "chat.completion.chunk",
                                    "model": req.model,
                                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                        except queue.Empty:
                            await asyncio.sleep(0.02)
                    yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Error in stream: {e}")
                finally:
                    stop_event.set()
                    await monitor_task
            
            return StreamingResponse(sse_generator(), media_type="text/event-stream")
            
        else:
            outputs = []
            def run_gen():
                try:
                    out = model.generate(
                        **inputs, 
                        max_new_tokens=req.max_tokens,
                        temperature=req.temperature,
                        do_sample=req.temperature > 0,
                        pad_token_id=tokenizer.eos_token_id,
                        past_key_values=past_key_values,
                        stopping_criteria=StoppingCriteriaList([StopOnSignal(stop_event)])
                    )
                    outputs.append(out)
                except Exception as e:
                    logger.error(f"Generation thread error: {e}")
            
            thread = Thread(target=run_gen)
            thread.start()
            
            while thread.is_alive():
                await asyncio.sleep(0.05)
                
            stop_event.set()
            await monitor_task
            
            if not outputs:
                raise HTTPException(status_code=500, detail="Generation was aborted or failed.")
            
            outputs_tensor = outputs[0]
            # Extract only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs_tensor[0][input_len:]
            response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return {
                "id": "chatcmpl-spectralquant",
                "object": "chat.completion",
                "created": 1234567890,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": input_len,
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": input_len + len(generated_tokens)
                }
            }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11436)

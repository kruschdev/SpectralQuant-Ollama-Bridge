"""
SpectralQuant calibration for 8-bit quantized models.

Runs eigenspectral calibration on the 8-bit quantized version of a model
so the KV cache compression matrices match the quantized attention patterns.
"""
import os
import sys
import argparse
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
# Monkeypatch to prevent dequantization/re-initialization OOMs during model load
transformers.modeling_utils.PreTrainedModel._initialize_missing_keys = lambda *args, **kwargs: None


# Ensure spectralquant is importable
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "spectralquant-src" / "src"))

from spectralquant.calibration import EigenspectralCalibrator

def main():
    parser = argparse.ArgumentParser(description="Calibrate SpectralQuant for 8-bit models")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B",
                       help="HuggingFace model ID")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of calibration samples")
    parser.add_argument("--max-tokens", type=int, default=50_000,
                       help="Max tokens per layer for calibration")
    args = parser.parse_args()

    model_id = args.model
    print(f"Loading tokenizer {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading model {model_id} in 8-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    
    print("Loading Wikitext-103 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = [row["text"] for row in dataset if len(row["text"].strip()) > 50]
    
    print("Initializing EigenspectralCalibrator...")
    calibrator = EigenspectralCalibrator(max_tokens_per_layer=args.max_tokens)
    
    print(f"Running calibration ({args.samples} samples)...")
    calibrator.calibrate(model, tokenizer, texts, n_samples=args.samples)
    
    out_dir = PROJECT_ROOT / "calibration_data"
    out_dir.mkdir(exist_ok=True)
    model_basename = model_id.split("/")[-1]
    # Suffix with _8bit to distinguish from bf16 and 4bit calibration
    out_path = out_dir / f"{model_basename}_8bit_wikitext103"
    
    print(f"Saving calibration data to {out_path}...")
    calibrator.save(str(out_path))
    print("Calibration complete!")

if __name__ == "__main__":
    main()

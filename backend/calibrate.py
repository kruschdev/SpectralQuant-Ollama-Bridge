import os
import sys
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure spectralquant is importable
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "spectralquant-src" / "src"))

from spectralquant.calibration import EigenspectralCalibrator

def main():
    model_id = "Qwen/Qwen3.5-9B"
    print(f"Loading tokenizer {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading model {model_id}...")
    # Load model in bfloat16 to save memory during forward passes
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    print("Loading Wikitext-103 dataset...")
    # We just need a list of strings
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = [row["text"] for row in dataset if len(row["text"].strip()) > 50]
    
    print("Initializing EigenspectralCalibrator...")
    calibrator = EigenspectralCalibrator(max_tokens_per_layer=50_000)
    
    print("Running calibration (using up to 100 samples)...")
    calibrator.calibrate(model, tokenizer, texts, n_samples=100)
    
    out_dir = PROJECT_ROOT / "calibration_data"
    out_dir.mkdir(exist_ok=True)
    model_basename = model_id.split("/")[-1]
    out_path = out_dir / f"{model_basename}_wikitext103"
    
    print(f"Saving calibration data to {out_path}...")
    calibrator.save(str(out_path))
    print("Calibration complete!")

if __name__ == "__main__":
    main()

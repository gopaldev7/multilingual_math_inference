import os
import json
import logging
import time
import torch
from typing import Any, Dict, List, Tuple
from utils import *

from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from a JSON file. Ensures that the directory exists.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} records from {file_path}")
    return data


def generate_inference(model: Any, tokenizer: Any, prompts: list, device: torch.device, max_tokens: int = 256) -> list:
    """
    Generate inference using the provided model and tokenizer.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.0, do_sample=False)
    
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


def run_inference_for_model(
    local_model_dir: str,
    model_info: Dict[str, str],
    dataset: List[Dict[str, Any]],
    prompt_template: str,
    max_tokens: int,
    accelerator: Accelerator,
    use_quantization: bool = False,
    batch_size: int = 16
) -> List[Dict[str, Any]]:
    """
    Download/load the specified model, run inference on dataset records,
    and store the results under 'inference_results' in each record.
    """
    model_name = model_info["name"]
    model_path = model_info["path"]
    logger.info(f"Processing model: {model_name}")

    local_model_path, model, tokenizer = download_and_load_model(local_model_dir, model_path, use_quantization)

    batch_prompts = []
    batch_indices = []

    for idx, record in enumerate(dataset):
            question = record.get("translated_question")
            if not question:
                record.setdefault("inference_results", {})[model_name] = None
            else:
                prompt = prompt_template.format(question=question)
                batch_prompts.append(prompt)
                batch_indices.append(idx)

            if len(batch_prompts) == batch_size:
                start_time = time.time()
                try:
                    outputs = generate_inference(model, tokenizer, batch_prompts, accelerator.device, max_tokens)
                except Exception as e:
                    logger.error(f"Error during inference for model {model_name}: {e}")
                    outputs = ["Error"] * len(batch_prompts)  
                elapsed = time.time() - start_time
                logger.info(f"Processed batch of {len(batch_prompts)} prompts in {elapsed:.2f} seconds")
                for b_idx, output in zip(batch_indices, outputs):
                    dataset[b_idx].setdefault("inference_results", {})[model_name] = output
                batch_prompts, batch_indices = [], []

    if batch_prompts:
        start_time = time.time()
        outputs = generate_inference(model, tokenizer, batch_prompts, accelerator.device, max_tokens)
        elapsed = time.time() - start_time
        logger.info(f"Processed final batch of {len(batch_prompts)} prompts in {elapsed:.2f} seconds")
        for b_idx, output in zip(batch_indices, outputs):
            dataset[b_idx].setdefault("inference_results", {})[model_name] = output

    del model
    del tokenizer
    torch.cuda.empty_cache()
    logger.info(f"Finished processing model: {model_name}")
    return dataset


def save_results(dataset: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save the inference results to a JSON file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_file}")


def main() -> None:
    accelerator = Accelerator()
    logger.info(f"Using device: {accelerator.device}")

    config = load_config("config.json")
    input_file = config["input_file_path"]
    output_file = config["output_file_path"]
    prompt_template = config["prompt_template"]
    max_tokens = config["max_tokens"]
    local_model_dir = config["local_model_dir"]
    batch_size = config["batch_size"]
    use_quantization = config.get("use_quantization", False)

    dataset = load_dataset(input_file)

    for model_info in config["models"]:
        output_data = run_inference_for_model(local_model_dir, model_info, dataset, prompt_template, max_tokens, accelerator, use_quantization, batch_size)
    
    save_results(output_data, output_file)


if __name__ == "__main__":
    main()

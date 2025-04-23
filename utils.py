import os
import json
import logging
import torch
from typing import Any, Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info(f"Configuration loaded from {config_file}")
    return config


def download_and_load_model(local_model_dir: str, model_path: str, use_quantization: bool = False) -> Tuple[str, Any, Any]:
    """
    Download a model (if not already downloaded) and load the model and tokenizer.
    Returns a tuple of (local_directory, model, tokenizer).
    """
    local_dir = os.path.join(local_model_dir, model_path.replace("/", "_"))
    if not os.path.exists(local_dir):
        logger.info(f"Downloading model {model_path} to {local_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if "t5" in model_path or "flan-t5" in model_path:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                load_in_8bit=use_quantization,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=use_quantization,
                device_map="auto",
                torch_dtype=torch.float16
            )
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)
    else:
        logger.info(f"Loading model from local directory {local_dir}")
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        if "t5" in model_path or "flan-t5" in model_path:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                local_dir,
                load_in_8bit=use_quantization,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                local_dir,
                load_in_8bit=use_quantization,
                device_map="auto",
                torch_dtype=torch.float16
            )
    return local_dir, model, tokenizer


def clear_cache_if_needed(device=None, threshold_gb : int = 19):
    """
    Clears the GPU cache if the allocated memory on the device exceeds the threshold.
    """
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    memory_threshold = threshold_gb * 1024 ** 3

    current_allocated = torch.cuda.memory_allocated(device)
    print(f"Current allocated memory on {device}: {current_allocated / (1024 ** 3):.2f} GB")

    if current_allocated >= memory_threshold:
        torch.cuda.empty_cache()
        print(f"Memory exceeded {threshold_gb}GB threshold. Cleared GPU cache.")

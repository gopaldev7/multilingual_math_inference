import json
import logging
import torch
from typing import Any, Dict, List, Tuple


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


def clear_cache_if_needed(device=None, threshold_gb : int = 19):
    """
    Clears the GPU cache if the allocated memory on the device exceeds the threshold.

    Args:
        threshold_gb (int, optional): Memory threshold in gigabytes. Defaults to 19.
        device (torch.device, optional): The GPU device to check. Defaults to cuda:0 if available.

    Returns:
        bool: True if cache was cleared, False otherwise.
    """
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    memory_threshold = threshold_gb * 1024 ** 3

    current_allocated = torch.cuda.memory_allocated(device)
    print(f"Current allocated memory on {device}: {current_allocated / (1024 ** 3):.2f} GB")

    # If memory usage exceeds the threshold, clear the cache.
    if current_allocated >= memory_threshold:
        torch.cuda.empty_cache()
        print(f"Memory exceeded {threshold_gb}GB threshold. Cleared GPU cache.")

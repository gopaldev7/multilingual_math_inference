import os
import json
import logging
import time
from typing import Optional, List, Dict, Tuple
from utils import load_config

from datasets import load_dataset
from googletrans import Translator, LANGUAGES


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def back_translate_data(input_file: str, output_file: str, max_retries: int = 3, sleep_interval: int = 2) -> None:
    """
    Loads a translated data file, back translates each 'translated_question' to English,
    and saves the updated records to a separate output file.

    Args:
        input_file (str): Path to the JSON file with translated data.
        output_file (str): Path to save the back translated data.
        max_retries (int): Maximum number of retries for translation.
        sleep_interval (int): Seconds to wait between retries.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as f:
        translated_data  = json.load(f)

    translator = Translator()

    backtrans_data = {}

    for record in translated_data:
        sample_index = record.get("sample_index")
        lang = record.get("language_code")
        translated_question = record.get("translated_question")
        
        if sample_index is None:
            continue
        
        if sample_index not in backtrans_data:
            backtrans_data[sample_index] = {
                "sample_index": sample_index,
                "original_question": None,   # filled from the English record
                "back_translations": {}
            }
        
        if lang != "en":
            back_translated = safe_translate(translator, translated_question, dest_lang="en",
                                         max_retries=max_retries, sleep_interval=sleep_interval) if translated_question else None
            
            backtrans_data[sample_index]["back_translations"][lang] = back_translated
        
        if lang == "en" and not backtrans_data[sample_index]["original_question"]:
            backtrans_data[sample_index]["original_question"] = translated_question
            logger.info(f"Back translating sample {sample_index}.")

    final_output = [backtrans_data[idx] for idx in sorted(backtrans_data.keys())]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    logger.info(f"Back translated dataset saved to {output_file}")


def load_gsm8k_dataset(config: Dict) -> Tuple[List[str], List[Dict]]:
    """
    Loads the GSM8K dataset and extracts the English questions.
    Expects keys in config:
      - dataset_name: str
      - dataset_config: str
      - split: str (e.g., "train")
      - sample_limit: int
    Returns:
        samples: List of English questions.
        full_dataset: Full dataset for retrieving answers.
    """
    logger.info("Loading GSM8K dataset...")
    dataset = load_dataset(config["dataset_name"], config["dataset_config"])[config["split"]]
    questions = dataset["question"][:config["sample_limit"]]
    samples = [q["en"] for q in questions]
    logger.info(f"Loaded {len(samples)} samples from dataset.")
    return samples, dataset

def safe_translate(translator: Translator, text: str, dest_lang: str,
                   max_retries: int, sleep_interval: int) -> Optional[str]:
    """
    Attempts to translate text to dest_lang with retries.
    Returns the translated text or None if all attempts fail.
    """
    for attempt in range(max_retries):
        try:
            translated = translator.translate(text, dest=dest_lang)
            return translated.text
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for language '{dest_lang}': {e}")
            time.sleep(sleep_interval)
    return None

def translate_samples(samples: List[str], full_dataset: List[Dict],
                      translator: Translator, config: Dict) -> List[Dict]:
    """
    Translates each sample into all target languages.
    Returns a list of records with:
      - sample_index
      - language_code, language_name
      - translated_question
      - answer (from the original dataset)
    """
    translated_data = []
    language_codes = list(LANGUAGES.keys())
    
    for idx, original_question in enumerate(samples):
        for lang in language_codes:
            translated_question = safe_translate(
                translator, original_question, dest_lang=lang,
                max_retries=config["max_retries"],
                sleep_interval=config["sleep_interval"]
            )
            record = {
                "sample_index": idx,
                "language_code": lang,
                "language_name": LANGUAGES[lang],
                "translated_question": translated_question,
                "answer": full_dataset[idx]["answer"]
            }
            translated_data.append(record)
        logger.info(f"Translated sample {idx} into {len(language_codes)} languages.")
    return translated_data

def save_translated_data(data: List[Dict], output_file: str) -> None:
    """
    Saves the translated data to a JSON file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Translated dataset saved to {output_file}")

def main():
    # Load configuration from config.json
    config = load_config("config.json")
    
    # Initialize the translator.
    translator = Translator()
    
    # Load dataset and extract English questions.
    samples, full_dataset = load_gsm8k_dataset(config)
    
    # Translate all samples.
    translated_data = translate_samples(samples, full_dataset, translator, config)
    
    # Save the translated dataset.
    save_translated_data(translated_data, config["output_file"])

if __name__ == "__main__":
    main()

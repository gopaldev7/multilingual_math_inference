#get final answer from the responses
#if model interpreting right semantically, even though giving correct answer mathematically
# make the main function modular to run specific methods only for visualisation

# change nature of plot translation accuracy to not do for each language
# get new numerical answers func to run using llm
# write all variables/ file names to config json and also decide if there is a need to save final/numerical answers to a file.
# check calculate_inference_accuracy and plot_inference_accuracy functions, not seen yet
# check if clear cache runs fine

import os
import json
import logging
import re
from typing import Any, Dict, List, Tuple
from accelerate import Accelerator

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util
from data_translator import back_translate_data
from utils import load_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_backtranslated_file(config: Dict[str, Any]) -> str:
    """
    Check if the back-translated data file exists.
    If not, run the back translation function from data_translator.py.
    Returns the path to the back-translated file.
    """
    back_translation_file = config["back_translation_file"]
    if not os.path.exists(back_translation_file):
        logger.info("Back translated data file not found. Running back translation.")
        try:
            back_translate_data(config["input_file_path"], back_translation_file)
        except ImportError as e:
            logger.error("back_translate_data function not found in data_translator.py")
            raise e
    return back_translation_file

def compute_translation_accuracy(back_translation_file: str) -> Dict[str, List[float]]:
    """
    Computes cosine similarity between the original English question and the back-translated question
    for each record. Each record has 'original_question' and 'back_translated_question'
    and a 'language_code' field.
    Returns a dictionary mapping language codes to lists of cosine similarity values.
    """
    with open(back_translation_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} records for translation accuracy computation.")
    
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    similarities_by_lang = {}
    
    for record in data:
        orig = record.get("original_question")
        back_translations = record.get("back_translations", {})
        for lang, back_trans in back_translations.items():
            if orig and back_trans:
                emb_orig = embedder.encode(orig, convert_to_tensor=True)
                emb_back = embedder.encode(back_trans, convert_to_tensor=True)
                cos_sim = util.pytorch_cos_sim(emb_orig, emb_back).item()
                if lang not in similarities_by_lang:
                    similarities_by_lang[lang] = []
                similarities_by_lang[lang].append(cos_sim)
    return similarities_by_lang

def plot_translation_accuracy(similarities_by_lang: Dict[str, List[float]]) -> None:
    """
    Plots a bar chart of average cosine similarity (translation accuracy) per language.
    """
    languages = []
    avg_similarities = []
    for lang, sims in similarities_by_lang.items():
        languages.append(lang)
        avg_similarities.append(np.mean(sims))
    
    plt.figure(figsize=(12, 6))
    plt.bar(languages, avg_similarities, color="skyblue")
    plt.xlabel("Language Code")
    plt.ylabel("Average Cosine Similarity")
    plt.title("Translation Accuracy: Cosine Similarity between Original and Back Translated Questions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def extract_numerical_answer(text: str) -> str:
    """
    Extracts the final numerical answer from the inference text result.
    """
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return matches[-1] if matches else ""

def extract_final_answers(inference_file: str, output_file: str) -> None:
    """
    Loads inference results from inference_file, extracts the final numerical answer from each model's output
    for each record, and saves the updated data (with a new 'final_answers' field) to output_file.
    """
    with open(inference_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for record in data:
        inference_results = record.get("inference_results", {})
        final_answers = {}
        for model, output in inference_results.items():
            if output:
                final = extract_numerical_answer(output)
            else:
                final = ""
            final_answers[model] = final
        record["final_answers"] = final_answers
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Final numerical answers extracted and saved to {output_file}")

def calculate_inference_accuracy(data: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compares the extracted final answers (in 'final_answers') with the ground truth (in 'answer').
    Computes overall accuracy per model and per language.
    
    Returns:
        model_accuracy: Mapping of model names to overall accuracy percentage.
        language_accuracy: Mapping of language codes to accuracy per model.
    """
    model_counts = {}
    model_correct = {}
    language_counts = {}
    language_correct = {}
    
    for record in data:
        true_answer = str(record.get("answer", "")).strip()
        lang = record.get("language_code", "unknown")
        final_answers = record.get("final_answers", {})
        for model, ans in final_answers.items():
            model_counts[model] = model_counts.get(model, 0) + 1
            # Using exact match for simplicity; you might allow numeric tolerance
            if str(ans).strip() == true_answer:
                model_correct[model] = model_correct.get(model, 0) + 1
                if lang not in language_counts:
                    language_counts[lang] = {}
                    language_correct[lang] = {}
                language_counts[lang][model] = language_counts[lang].get(model, 0) + 1
                language_correct[lang][model] = language_correct[lang].get(model, 0) + 1
            else:
                if lang not in language_counts:
                    language_counts[lang] = {}
                    language_correct[lang] = {}
                language_counts[lang][model] = language_counts[lang].get(model, 0) + 1
    
    model_accuracy = {model: (model_correct.get(model, 0) / count * 100) for model, count in model_counts.items()}
    language_accuracy = {}
    for lang, counts in language_counts.items():
        language_accuracy[lang] = {}
        for model, count in counts.items():
            correct = language_correct[lang].get(model, 0)
            language_accuracy[lang][model] = correct / count * 100
    return model_accuracy, language_accuracy

def plot_inference_accuracy(model_accuracy: Dict[str, float], language_accuracy: Dict[str, Dict[str, float]]) -> None:
    """
    Visualizes inference accuracy:
      - Overall model accuracy (bar chart).
      - Accuracy per language for each model (grouped bar chart).
    """
    # Overall model accuracy
    models = list(model_accuracy.keys())
    accuracies = [model_accuracy[m] for m in models]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color="green")
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.title("Overall Inference Accuracy per Model")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()
    
    # Accuracy per language
    languages = list(language_accuracy.keys())
    all_models = models
    x = np.arange(len(languages))
    width = 0.8 / len(all_models)
    
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(all_models):
        accs = [language_accuracy[lang].get(model, 0) for lang in languages]
        plt.bar(x + i * width, accs, width, label=model)
    plt.xlabel("Language")
    plt.ylabel("Accuracy (%)")
    plt.title("Inference Accuracy per Language")
    plt.xticks(x + width * (len(all_models)-1) / 2, languages, rotation=45)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    config = load_config("config.json")
    
    # Step 1: Check translation accuracy
    back_translation_file = load_backtranslated_file(config)
    translation_similarities = compute_translation_accuracy(back_translation_file)
    plot_translation_accuracy(translation_similarities)
    
    # Step 2: Extract final numerical answers from inference outputs.
    # The inference file is assumed to be produced by your inference script.
    inference_file = config.get("inference_file")
    final_output_file = config.get("final_output_file", "data/inference_final_answers.json")
    extract_final_answers(inference_file, final_output_file)
    
    # Load final inference data.
    with open(final_output_file, "r", encoding="utf-8") as f:
        inference_data = json.load(f)
    
    # Step 3: Calculate and visualize inference accuracy.
    model_accuracy, language_accuracy = calculate_inference_accuracy(inference_data)
    plot_inference_accuracy(model_accuracy, language_accuracy)

if __name__ == "__main__":
    main()


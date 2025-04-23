#if model interpreting right semantically, even though giving correct answer mathematically

import os
import json
import logging
import torch
import math
from typing import Any, Dict, List, Tuple
from accelerate import Accelerator
from googletrans import LANGUAGES

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from data_translator import back_translate_data
from utils import load_config, download_and_load_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_backtranslated_file(config: Dict[str, Any]) -> str:
    """
    Check if the back-translated data file exists.
    If not, run the back translation function from data_translator.py.
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

def compute_translation_accuracy(back_translation_file: str, config: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Computes cosine similarity between the original English question and the back-translated question
    for each record.
    """
    with open(back_translation_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} records for translation accuracy computation.")
    
    embedder = SentenceTransformer(config["sentence_transformer"])
    similarities_by_lang = {}
    
    for record in data:
        orig = record.get("original_question")
        back_translations = record.get("back_translations", {})
        if orig and back_translations:
            langs = list(back_translations.keys())
            back_texts = list(back_translations.values())
            emb_orig = embedder.encode(orig, convert_to_tensor=True)
            emb_backs = embedder.encode(back_texts, convert_to_tensor=True)
            cos_sims = util.pytorch_cos_sim(emb_orig.unsqueeze(0), emb_backs).squeeze(0)
            for lang, cos_sim in zip(langs, cos_sims.tolist()):
                similarities_by_lang.setdefault(lang, []).append(cos_sim)
    return similarities_by_lang

def plot_translation_accuracy(similarities_by_lang: Dict[str, List[float]]) -> None:
    """
    Creates a two-panel figure to visualize translation accuracy across languages.
    
    Panel 1: A sorted bar chart for the average cosine similarity per language,
             with error bars the overall average and median across all languages.
             
    Panel 2: A histogram of the average cosine similarities across languages with
             markers for the overall average and median.
    """
    language_stats = {
        LANGUAGES[lang]: (np.mean(scores), np.std(scores))
        for lang, scores in similarities_by_lang.items()
    }
    
    sorted_languages = sorted(language_stats.keys(), key=lambda lang: language_stats[lang][0])
    means = [language_stats[lang][0] for lang in sorted_languages]
    std_devs = [language_stats[lang][1] for lang in sorted_languages]

    overall_mean = np.mean(means)
    overall_median = np.median(means)

    fig, axs = plt.subplots(2, 1, figsize=(28, 24))
    
    # --- Panel 1: Sorted Bar Chart with Error Bars ---
    axs[0].bar(sorted_languages, means, yerr=std_devs, capsize=4, color="skyblue")
    axs[0].set_xlabel("Language")
    axs[0].set_ylabel("Average Cosine Similarity")
    axs[0].set_title("Translation Accuracy per Language\n(Bar Chart Sorted by Average Similarity)")
    axs[0].tick_params(axis="x", rotation=90)
    
    # Overlay horizontal lines for the overall mean and median.
    axs[0].axhline(overall_mean, color="red", linestyle="--", label=f"Overall Mean: {overall_mean:.2f}")
    axs[0].axhline(overall_median, color="green", linestyle="--", label=f"Overall Median: {overall_median:.2f}")
    axs[0].legend()
    
    # --- Panel 2: Histogram of Average Similarities ---
    bins = 20
    axs[1].hist(means, bins=bins, color="lightgreen", edgecolor="black")
    axs[1].set_xlabel("Average Cosine Similarity")
    axs[1].set_ylabel("Number of Languages")
    axs[1].set_title("Distribution of Average Cosine Similarities Across Languages")
    axs[1].axvline(overall_mean, color="red", linestyle="--", label=f"Mean: {overall_mean:.2f}")
    axs[1].axvline(overall_median, color="green", linestyle="--", label=f"Median: {overall_median:.2f}")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig("translation_accuracy.png")

def extract_numerical_answers(prompts: List[str],
                             extraction_model: Any,
                             extraction_tokenizer: Any,
                             device: torch.device,
                             max_tokens: int = 64) -> List[str]:
    """
    Use an LLM to extract numerical answers in a batch given a list of prompts.
    """
    if extraction_tokenizer.pad_token is None:
        extraction_tokenizer.pad_token = extraction_tokenizer.eos_token

    inputs = extraction_tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = extraction_model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.0, do_sample=False)
    
    return [extraction_tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]


def extract_final_answers(inference_file: str, output_file: str, accelerator: Accelerator, config: Dict[str, Any]) -> None:
    """
    Loads inference results from inference_file, extracts the final numerical answer from each model's output
    for each record, and saves the updated data to output_file.
    """
    local_model_path, model, tokenizer = download_and_load_model(config["local_model_dir"], config["extractor_model_path"], config["use_quantization"])

    with open(inference_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    extraction_jobs = []

    for idx, record in enumerate(data):
        inference_results = record.get("inference_results", {})
        record.setdefault("final_answers", {})
        for modelname, output in inference_results.items():
            if output:
                prompt = (
                    "Extract the final numerical answer from the text below. "
                    "Return only the number with no additional text.\n\n"
                    f"{output}\n\n"
                    "Answer:"
                )
                extraction_jobs.append((idx, modelname, prompt))
            else:
                record["final_answers"][modelname] = ""

    batch_size = config.get("batch_size", 16) 
    num_batches = math.ceil(len(extraction_jobs) / batch_size)

    for i in range(num_batches):
        batch_jobs = extraction_jobs[i * batch_size : (i + 1) * batch_size]
        batch_prompts = [job[2] for job in batch_jobs]

        batch_results = extract_numerical_answers(batch_prompts, model, tokenizer, accelerator.device)

        logger.info(f"Processed {i+1} batches of {len(batch_prompts)} prompts")

        for (record_index, modelname, _), result in zip(batch_jobs, batch_results):
            data[record_index]["final_answers"][modelname] = result

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
    plt.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.savefig("Model_accuracy.png")
    
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
    plt.tick_params(axis="x", rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Language_accuracy.png")

def plot_accuracy_distribution(language_accuracy: Dict[str, Dict[str, float]]):
    # Melt your nested dict into a long form DataFrame
    records = []
    for lang, accs in language_accuracy.items():
        for model, acc in accs.items():
            records.append({"language": lang, "model": model, "accuracy": acc})
    df_long = pd.DataFrame(records)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="model", y="accuracy", data=df_long)
    plt.ylim(0, 100)
    plt.title("Distribution of Per‑Language Accuracy by Model")
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("boxplot_accuracy.png")

def plot_accuracy_heatmap(language_accuracy: Dict[str, Dict[str, float]], models: List[str]):
    # Build a DataFrame: rows=languages, cols=models
    df = pd.DataFrame.from_dict(language_accuracy, orient="index", columns=models).fillna(0)
    
    plt.figure(figsize=(8, max(6, len(df) * 0.2)))  # height grows with number of langs
    sns.heatmap(df, cmap="YlGnBu", linewidths=0.5, linecolor="gray",
                cbar_kws={"label": "Accuracy (%)"})
    plt.title("Inference Accuracy per Language (Heatmap)")
    plt.xlabel("Model")
    plt.ylabel("Language")
    plt.tight_layout()
    plt.savefig("results/heatmap_accuracy.png")

def plot_top_bottom_languages(language_accuracy: Dict[str, Dict[str, float]], top_n=10):
    df = pd.DataFrame.from_dict(language_accuracy, orient="index")
    for model in df.columns:
        # sort languages by accuracy for this model
        sorted_langs = df[model].sort_values()
        fig, ax = plt.subplots(figsize=(6, 6))
        # Bottom n
        sorted_langs.head(top_n).plot.barh(ax=ax, color="salmon")
        ax.set_title(f"{model} — Bottom {top_n} Languages")
        ax.set_xlabel("Accuracy (%)")
        plt.tight_layout()
        plt.savefig(f"{model}_bottom_{top_n}.png")
        
        # Top n
        fig, ax = plt.subplots(figsize=(6, 6))
        sorted_langs.tail(top_n).plot.barh(ax=ax, color="seagreen")
        ax.set_title(f"{model} — Top {top_n} Languages")
        ax.set_xlabel("Accuracy (%)")
        plt.tight_layout()
        plt.savefig(f"{model}_top_{top_n}.png")

def main() -> None:
    accelerator = Accelerator()

    config = load_config("config.json")
    
    if(config["check_translation_accuracy"]):
        back_translation_file = load_backtranslated_file(config)
        translation_similarities = compute_translation_accuracy(back_translation_file, config)
        plot_translation_accuracy(translation_similarities)
    
    if(config["check_inference_accuracy"]):
        inference_file = config.get("inference_file")
        final_output_file = config.get("inference_answers_file")

        if not os.path.exists(final_output_file):
            os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
            extract_final_answers(inference_file, final_output_file, accelerator, config)
        
        with open(final_output_file, "r", encoding="utf-8") as f:
            inference_data = json.load(f)
        
        model_accuracy, language_accuracy = calculate_inference_accuracy(inference_data)
        plot_accuracy_distribution(language_accuracy)
        plot_accuracy_heatmap(language_accuracy, list(model_accuracy.keys()))
        plot_top_bottom_languages(language_accuracy)
        #plot_inference_accuracy(model_accuracy, language_accuracy)

if __name__ == "__main__":
    main()


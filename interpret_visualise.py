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
from scipy.stats import pearsonr
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

def compute_translation_accuracy(back_translation_file: str, config: Dict[str, Any]) -> None:
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

    output_file = config.get("translation_similarities_file")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(similarities_by_lang, f, indent=2, ensure_ascii=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def plot_translation_accuracy(similarities_by_lang: Dict[str, List[float]]) -> None:
    """
    Creates two separate figures to visualize translation accuracy across languages.

    Figure 1: A sorted bar chart for the average cosine similarity per language,
              with error bars and overall mean/median lines.

    Figure 2: A histogram of the average cosine similarities across languages
              with markers for overall mean and median.
    """
    language_stats = {
        LANGUAGES.get(lang, lang).title(): (np.mean(scores), np.std(scores))
        for lang, scores in similarities_by_lang.items()
    }

    sorted_languages = sorted(language_stats.keys(), key=lambda lang: language_stats[lang][0])
    means = [language_stats[lang][0] for lang in sorted_languages]
    std_devs = [language_stats[lang][1] for lang in sorted_languages]

    overall_mean = np.mean(means)
    overall_median = np.median(means)

    # --- Figure 1: Bar chart ---
    fig, ax = plt.subplots(figsize=(12, 28))
    ax.barh(sorted_languages, means, xerr=std_devs, capsize=4, color="skyblue")
    ax.set_ylabel("Language", fontsize=18)
    ax.set_xlabel("Average Cosine Similarity", fontsize=18)
    ax.set_title("Translation Accuracy per Language\n(Bar Chart Sorted by Average Similarity)", fontsize=24)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.axvline(overall_mean, color="red", linestyle="--", label=f"Overall Mean: {overall_mean:.2f}")
    ax.axvline(overall_median, color="green", linestyle="--", label=f"Overall Median: {overall_median:.2f}")
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("results/translation_accuracy_bar.png")

    # --- Figure 2: Histogram ---
    fig, ax = plt.subplots(figsize=(28, 12))
    bins = 20
    ax.hist(means, bins=bins, color="lightgreen", edgecolor="black")
    ax.set_xlabel("Average Cosine Similarity", fontsize=18)
    ax.set_ylabel("Number of Languages", fontsize=18)
    ax.set_title("Distribution of Average Cosine Similarities Across Languages", fontsize=24)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.axvline(overall_mean, color="red", linestyle="--", label=f"Mean: {overall_mean:.2f}")
    ax.axvline(overall_median, color="green", linestyle="--", label=f"Median: {overall_median:.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/translation_accuracy_hist.png")
    

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
    Visualizes inference accuracy for each model and language.
    """
    models = list(model_accuracy.keys())
    # 1) Heatmap
    plot_accuracy_heatmap(language_accuracy, models)
    
    # 2) Boxplot
    plot_accuracy_distribution(language_accuracy)
    
    # 3) Top 10 languages
    plot_top_languages(language_accuracy)

    # 4) Overall accuracy
    show_model_overall_accuracy(language_accuracy)

def plot_accuracy_distribution(language_accuracy: Dict[str, Dict[str, float]]):
    records = []
    for lang, accs in language_accuracy.items():
        for model, acc in accs.items():
            records.append({"language": lang, "model": model, "accuracy": acc})
    df_long = pd.DataFrame(records)

    order = df_long.groupby("model")["accuracy"].median().sort_values().index.tolist()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="model", y="accuracy", data=df_long, order=order)
    medians = df_long.groupby('model')['accuracy'].median().reindex(order)
    for i, m in enumerate(order):
        plt.text(i, medians[m] + 2, f"{medians[m]:.1f}%", 
            ha='center', va='bottom', fontweight='bold')
    plt.ylim(0, 100)
    plt.title("Distribution of Per‑Language Accuracy by Model")
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/boxplot_accuracy.png")

def plot_accuracy_heatmap(language_accuracy: Dict[str, Dict[str, float]], models: List[str]):

    df = pd.DataFrame.from_dict(language_accuracy, orient="index", columns=models).fillna(0) 
    df.index = [LANGUAGES.get(lang, lang).title() for lang in df.index]
    
    plt.figure(figsize=(8, max(6, len(df) * 0.2)))
    sns.heatmap(df, cmap="YlGnBu", linewidths=0.5, linecolor="gray",
                cbar_kws={"label": "Accuracy (%)"})
    plt.title("Inference Accuracy per Language (Heatmap)")
    plt.xlabel("Model")
    plt.ylabel("Language")
    plt.tight_layout()
    plt.savefig("results/heatmap_accuracy.png")

def plot_top_languages(language_accuracy: Dict[str, Dict[str, float]], top_n=10):

    df = pd.DataFrame.from_dict(language_accuracy, orient="index")
    df.index = [LANGUAGES.get(lang, lang).title() for lang in df.index]

    for model in df.columns:
        top = df[model].sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(6, 6))
        bars = ax.barh(top.index, top.values, color="seagreen")

        ax.invert_yaxis()
        max_val = top.max()
        ax.set_xlim(0, max_val * 1.10)

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center", ha="left", fontsize= 9
            )

        ax.set_title(f"{model} — Top {top_n} Languages", fontsize=14)
        ax.set_xlabel("Accuracy (%)", fontsize=12)
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(f"results/{model}_top_{top_n}.png")

def show_model_overall_accuracy(language_accuracy: Dict[str, Dict[str, float]]):
    """
    1) Prints a DataFrame of mean accuracy per model across all languages.
    2) Plots that same table as a matplotlib table.
    """
    df = pd.DataFrame(language_accuracy).T
    
    summary = df.mean().sort_values(ascending=False).round(2)
    
    fig, ax = plt.subplots(figsize=(6, summary.shape[0]*0.6 + 1))
    ax.axis("off")

    tbl = ax.table(
        cellText=[[model, f"{val:.2f}%"] for model, val in summary.items()],
        colLabels=["Model", "Mean Accuracy (%)"],
        loc='center'
    )
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.5)

    for (row, _), cell in tbl.get_celld().items():
        if row == 0:
            cell.get_text().set_fontweight('bold')

    plt.title("Overall Mean Accuracy per Model", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/overall_accuracy.png")

def plot_trans_vs_inf_correlation(back_translation_file: str, final_answers_file: str, model_names: List[str], n_bins: int = 10):
    # --- 1) Load & reshape back-translation scores ---
    bt = pd.read_json(back_translation_file)
    bt = bt.reset_index().rename(columns={'index':'sample_index'})
    bt_long = bt.melt(
        id_vars=['sample_index'],
        var_name='language_code',
        value_name='cosine_similarity'
    )

    # --- 2) Load & melt inference results ---
    inf = pd.read_json(final_answers_file)
    rows = []
    for _, row in inf.iterrows():
        idx, lang = row["sample_index"], row["language_code"]
        true_ans = str(row["answer"]).strip()
        for m in model_names:
            final = str(row["final_answers"].get(m, "")).strip()
            correct = int(final == true_ans and final != "")
            rows.append({
                "sample_index": idx,
                "language_code": lang,
                "model": m,
                "correct": correct
            })
    inf_df = pd.DataFrame(rows)

    # --- 3) Merge translation & inference data ---
    merged = inf_df.merge(
        bt_long[['sample_index','language_code','cosine_similarity']],
        on=['sample_index','language_code'],
        how='left'
    )

    # --- 4) Loop over models and plot ---
    for m in model_names:
        sub = merged[merged["model"] == m].dropna(subset=["cosine_similarity"])
        if sub.empty:
            print(f"No data for {m}, skipping.")
            continue

        # 4a) Compute Pearson correlation
        r, p = pearsonr(sub["cosine_similarity"], sub["correct"])
        print(f"{m}: Pearson r = {r:.3f}, p = {p:.3g}")
        
        # 4b) Quantile-bin the similarity scores
        sub['sim_bin'] = pd.qcut(sub['cosine_similarity'], q=n_bins, duplicates='drop')

        # 4c) Compute mean accuracy & sample count per bin
        grouped = sub.groupby('sim_bin')
        bin_stats = grouped['correct'].agg(
            mean_acc='mean',
            n='size'
        ).reset_index()

        # 4d) Compute the actual mean cosine similarity per bin
        bin_stats['x'] = grouped['cosine_similarity'].mean().values

        # 5) Plot mean accuracy with 95% CI error bars
        plt.figure(figsize=(6, 4))
        x    = bin_stats['x']
        y    = bin_stats['mean_acc']
        yerr = 1.96 * np.sqrt(y * (1 - y) / bin_stats['n'])

        plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=3)
        plt.ylim(0, 1)
        plt.xlabel("Back-Translation Cosine Similarity (binned)")
        plt.ylabel("Mean Inference Accuracy")
        plt.title(f"{m}: Accuracy vs Translation Quality\nr={r:.2f}, p={p:.2g}")
        plt.tight_layout()
        plt.savefig(f"results/binned_corr_{m.replace(' ', '_')}.png")

def main() -> None:
    accelerator = Accelerator()

    config = load_config("config.json")

    translation_similarities_file = config.get("translation_similarities_file")
    with open(translation_similarities_file, "r", encoding="utf-8") as f:
            translation_similarities = json.load(f)
    
    if(config["check_translation_accuracy"]):
        if not os.path.exists(translation_similarities_file):
            os.makedirs(os.path.dirname(translation_similarities_file), exist_ok=True)
            back_translation_file = load_backtranslated_file(config)
            compute_translation_accuracy(back_translation_file, config)
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
        plot_trans_vs_inf_correlation(
            translation_similarities_file,
            final_output_file,
            list(model_accuracy.keys()),
        )
        plot_inference_accuracy(model_accuracy, language_accuracy)

if __name__ == "__main__":
    main()


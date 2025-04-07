# Multilingual Math Inference

## Overview
This repository contains scripts and data for a multilingual math inference on some popular opensource LLMs.

The dataset used is a modified version of GSM8K dataset available on HuggingFace - [Link](https://huggingface.co/datasets/TaiMingLu/Multilingual-Benchmark)

## Folder Structure
- `data_translator.py`: Script for translating and back translating dataset.
- `inference.py`: Optimized inference script.
- `interpret_visualise.py`: Script for visualizing inference results.
- `config.json`: Configuration file for dataset paths, model paths, etc.
- `requirements.txt`: List of dependencies required for the project.

## Setup
1. Clone the repo: `git clone https://github.com/gopaldev7/<repo>.git`
2. Create a virtual environment: `python -m venv llm-env`
3. Activate and install dependencies: `pip install -r requirements.txt`

## Usage
1. Translate data: `python data_translator.py`
2. Run inference: `accelerate launch inference.py`
3. Visualize results: `python interpret_visualise.py`

## License
[MIT License](LICENSE)

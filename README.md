# AI Product Price Predictor

## Overview

AI Product Price Predictor is an end-to-end machine learning and Large Language Model (LLM) portfolio project that estimates the market price of products using local AI models powered by Ollama.

The project combines:

* Product preprocessing
* Local LLM integration
* Price prediction
* Dataset preparation
* Model evaluation
* Streamlit user interface
* Logging and exception handling
* Test-driven development

The project is inspired by the Week 6 Product Pricing workflow and is designed to demonstrate practical AI engineering skills suitable for portfolio and GitHub presentation.

---

# Features

## AI Product Analysis

* Product preprocessing using Ollama
* Product summarization
* Structured product representation

## Price Prediction

* Product price estimation
* Prediction prompt generation
* Prediction result extraction

## Evaluation System

* Absolute error calculation
* Percentage error calculation
* Accuracy measurement
* Dataset evaluation

## Local AI

* Ollama integration
* Llama 3.2 support
* Custom Ollama model support

## Software Engineering

* Logging
* Exception handling
* Modular architecture
* Unit testing
* GitHub-ready structure

---

# Technology Stack

## Backend

* Python 3.11+

## AI

* Ollama
* Llama 3.2
* LoRA Fine-Tuning (future phase)

## Frontend

* Streamlit

## Testing

* Pytest

## Data

* CSV
* JSONL

---

# Project Architecture

```text
User Input
    │
    ▼
Product Item
    │
    ▼
Preprocessor
    │
    ▼
Ollama Summary
    │
    ▼
Price Predictor
    │
    ▼
Price Evaluation
    │
    ▼
Streamlit UI
```

---

# Project Structure

```text
ai-product-price-predictor/
│
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── logging_config.py
│
├── data/
│   ├── raw/
│   │   ├── human_in.csv
│   │   └── human_out.csv
│   │
│   ├── processed/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   ├── test.jsonl
│   │   └── base_llama_results.csv
│   │
│   └── samples/
│
├── pricer/
│   ├── __init__.py
│   ├── items.py
│   ├── parser.py
│   ├── preprocessor.py
│   ├── predictor.py
│   ├── evaluator.py
│   ├── dataset_builder.py
│   ├── fine_tune_data.py
│   └── ollama_client.py
│
├── training/
│   ├── prepare_finetune_dataset.py
│   ├── train_lora_unsloth.py
│   ├── export_to_ollama.py
│   └── README_TRAINING.md
│
├── models/
│   ├── adapters/
│   └── ollama/
│       └── Modelfile
│
├── scripts/
│   ├── check_preprocessor.py
│   ├── check_predictor.py
│   └── evaluate_base_model.py
│
├── logs/
│   └── app.log
│
└── tests/
```

---

# Installation

## Clone Repository

```bash
git clone <your-repository-url>
cd ai-product-price-predictor
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Ollama Setup

Install Ollama:

https://ollama.com

Pull Llama 3.2:

```bash
ollama pull llama3.2:latest
```

Verify:

```bash
ollama list
```

Expected:

```text
llama3.2:latest
```

---

# Create Custom Ollama Model

Run:

```bash
python -m training.export_to_ollama
```

Verify:

```bash
ollama list
```

Expected:

```text
product-price-predictor
llama3.2:latest
```

---

# Run Application

Start Streamlit:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# Running Tests

Run all tests:

```bash
pytest
```

Run specific test:

```bash
pytest tests/test_06_prediction_prompt.py
```

---

# Logging

Application logs are stored in:

```text
logs/app.log
```

Logging includes:

* Information logs
* Warnings
* Errors
* Exception traces

Example:

```text
2026-06-18 03:53:58 | INFO | predictor.py | Price prediction completed
```

---

# Dataset

The project uses Week 6 product pricing data.

Files:

```text
data/raw/human_in.csv
data/raw/human_out.csv
```

Processed files:

```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl
```

---

# Fine-Tuning Pipeline

## Current Status

Current model:

```text
product-price-predictor
```

is:

```text
llama3.2 + custom prompt
```

It is NOT fine-tuned yet.

---

## Future Fine-Tuning

Workflow:

```text
Week 6 Data
      │
      ▼
JSONL Dataset
      │
      ▼
LoRA Training
      │
      ▼
Adapter Export
      │
      ▼
Ollama Import
      │
      ▼
Fine-Tuned Price Predictor
```

---

# GPU Requirement

LoRA training requires a GPU.

Recommended environments:

* Google Colab
* Kaggle
* RunPod
* Vast.ai
* Local NVIDIA CUDA GPU

CPU-only systems are not recommended.

---

# Current Limitations

Because the model is not fine-tuned yet:

* Predictions may be unstable
* Predictions may vary between runs
* Some products may receive unrealistic prices

Example:

```text
Run 1 → $87
Run 2 → $4200
```

This is expected for the baseline model.

---

# Future Improvements

## Planned Features

* LoRA Fine-Tuning
* Adapter Import into Ollama
* Model Comparison Dashboard
* Prediction History
* Batch Product Pricing
* CSV Upload
* Price Visualization
* Model Performance Reports

---

# Portfolio Highlights

This project demonstrates:

* Python Development
* Object-Oriented Programming
* Streamlit Development
* LLM Integration
* Ollama Integration
* Prompt Engineering
* Dataset Processing
* AI Evaluation Metrics
* Logging
* Exception Handling
* Test-Driven Development
* Git Workflow
* Modular Architecture

---

# Author

Developed as a portfolio project demonstrating practical AI engineering and machine learning deployment skills.

## Project Name

AI Product Price Predictor

## Version

1.0.0

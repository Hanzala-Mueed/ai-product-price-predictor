# AI Product Price Predictor

## Version 2.0.0

AI Product Price Predictor is an end-to-end AI Engineering portfolio project that estimates realistic product prices using local Large Language Models (LLMs) powered by Ollama.

The project demonstrates the complete lifecycle of an AI application:

* Data preparation
* Product preprocessing
* Local LLM inference
* Price prediction
* LoRA fine-tuning
* GGUF model export
* Ollama deployment
* Model routing
* Evaluation
* Streamlit frontend
* Logging
* Exception handling
* Test-driven development

The project is inspired by the Week 6 Product Pricing workflow and extended into a production-style AI system.

---

# Features

## Product Preprocessing

* Product cleaning
* Product summarization
* Structured product representation
* AI-powered preprocessing

## Price Prediction

* Product price estimation
* Prompt generation
* Price extraction
* Prediction validation

## Local AI Inference

* Ollama integration
* Local model execution
* Fine-tuned model support
* GGUF model deployment

## Fine-Tuning Pipeline

* LoRA Fine-Tuning
* Unsloth Training
* HuggingFace Transformers
* PEFT Adapters
* GGUF Export
* Ollama Import

## Evaluation System

* Absolute Error
* Percentage Error
* Accuracy Measurement
* Model Comparison

## Software Engineering

* Modular Architecture
* Logging
* Exception Handling
* Unit Testing
* GitHub-Friendly Commit Structure

---

# Technology Stack

## Backend

* Python 3.11+

## AI / LLM

* Ollama
* Llama 3.2
* LoRA Fine-Tuning
* Unsloth
* HuggingFace Transformers
* TRL
* PEFT
* GGUF

## Frontend

* Streamlit

## Testing

* Pytest

## Data

* CSV
* JSONL

---

# System Architecture

```text
User Input
     │
     ▼
Item Builder
     │
     ▼
Preprocessor
(product-price-predictor)
     │
     ▼
Structured Summary
     │
     ▼
Price Predictor
(product-price-predictor-finetuned)
     │
     ▼
Fallback Predictor
(llama3.2:latest)
     │
     ▼
Price Extraction
     │
     ▼
Evaluation Layer
     │
     ▼
Streamlit UI
```

---

# Model Routing Architecture

The project uses different models for different tasks.

```text
Preprocessing
      │
      ▼
product-price-predictor
(Base Model)

Prediction
      │
      ▼
product-price-predictor-finetuned
(Fine-Tuned Model)

Fallback
      │
      ▼
llama3.2:latest
```

Purpose:

* Base model cleans product data
* Fine-tuned model predicts prices
* Fallback model handles prediction failures

---

# Project Structure

```text
ai-product-price-predictor/
│
├── app.py
├── README.md
├── requirements.txt
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
│   ├── import_adapter_to_ollama.py
│   └── README_TRAINING.md
│
├── models/
│   │
│   ├── adapters/
│   │   └── product_price_lora/
│   │
│   ├── gguf/
│   │   └── product_price_predictor.gguf
│   │
│   ├── hf/
│   │   └── product_price_predictor/
│   │
│   └── ollama/
│
├── scripts/
│   ├── check_preprocessor.py
│   ├── check_predictor.py
│   ├── check_finetuned_model.py
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
git clone <repository-url>
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

# modelfile 
create folder a\ models/ollama
create a file with name: Modelfile
add this content to this file:

```bash

FROM ../hf/product_price_predictor/product_price_predictor_gguf

SYSTEM """
You are an AI product price prediction assistant.

Estimate the realistic US marketplace price of the product.

Rules:
- Respond only with one price number.
- Do not include a dollar sign.
- Do not include explanation.
- Round to the nearest dollar.
"""

PARAMETER temperature 0
PARAMETER top_p 0.1
PARAMETER num_predict 20

```

---

# Base Model Setup

Create the base custom model:

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

# Fine-Tuning Workflow

## Step 1

Generate datasets:

```bash
python -m training.prepare_finetune_dataset
```

Creates:

```text
train.jsonl
validation.jsonl
test.jsonl
```

---

## Step 2

Open Google Colab notebook.

Upload:

```text
data/processed/train.jsonl
data/processed/validation.jsonl
```

Run all notebook cells.

Training uses:

```text
Unsloth
Llama 3.2 3B
LoRA
Tesla T4 GPU
```

---

## Step 3

Download:

```text
product_price_predictor_gguf.zip
```
Unzip the folder and

Place product_price_predictor_gguf folder at :

```text
models/hf/product_price_predictor
```

---

## Step 4

create the finetuned model using this command:

```bash
ollama create product-price-predictor-finetuned -f models/ollama/Modelfile
```

<!-- Import into Ollama:

```bash
python -m training.import_adapter_to_ollama
``` -->

Verify:

```bash
ollama list
```

Expected:

```text
product-price-predictor-finetuned
product-price-predictor
llama3.2:latest
```

---

# Fine-Tuning Results

Training Environment:

```text
Google Colab
Tesla T4 GPU
Unsloth
```

Training Configuration:

```text
Base Model:
Llama-3.2-3B-Instruct

LoRA Rank:
16

LoRA Alpha:
16

Epochs:
1

Training Examples:
80

Validation Examples:
10
```

Training Output:

```text
Training Loss:
3.28

Validation Loss:
2.44
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

Run individual test:

```bash
pytest tests/test_16_model_routing.py
```

---

# Logging

Logs are stored at:

```text
logs/app.log
```

Logging includes:

* INFO
* WARNING
* ERROR
* Exception Stack Traces

Example:

```text
2026-06-21 10:29:23 | INFO | predictor.py | Price prediction completed
```

---

# Dataset

Based on Week 6 pricing data.

Raw Files:

```text
data/raw/human_in.csv
data/raw/human_out.csv
```

Processed Files:

```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl
```

---

# Current Models

## Preprocessor

```text
product-price-predictor
```

Used for:

* Product cleanup
* Product summarization
* Structured product representation

---

## Predictor

```text
product-price-predictor-finetuned
```

Used for:

* Price estimation
* Price prediction

---

## Fallback

```text
llama3.2:latest
```

Used when:

* Fine-tuned model fails
* Invalid prediction returned
* Extraction errors occur

---

# Current Limitations

The fine-tuned model was trained on a relatively small dataset.

Training Dataset:

```text
80 examples
```

As a result:

* Some predictions remain inaccurate
* Some product categories are underrepresented
* Fine-tuned responses may occasionally be invalid
* Fallback model may be required

Future improvements will focus on larger datasets and stronger evaluation.

---

# Future Roadmap

## Version 2.0

* Model Comparison Dashboard
* Base vs Fine-Tuned Evaluation
* Error Analysis Reports

## Version 2.1

* Batch Product Prediction
* CSV Upload Support
* Prediction History

## Version 3.0

* Larger Fine-Tuning Dataset
* Multi-Model Ensemble
* Price Confidence Scores
* Vendor Price Comparison
* RAG-Based Product Retrieval

---

# Portfolio Highlights

This project demonstrates:

* Python Development
* Object-Oriented Programming
* Streamlit Development
* Local AI Deployment
* Ollama Integration
* LLM Engineering
* Prompt Engineering
* LoRA Fine-Tuning
* Unsloth Training
* GGUF Model Export
* Fine-Tuned Ollama Deployment
* Model Routing Architecture
* Dataset Engineering
* Evaluation Metrics
* Logging & Monitoring
* Exception Handling
* Unit Testing
* GitHub Workflow
* Modular Architecture

---

# Author

Developed as an AI Engineering portfolio project demonstrating practical machine learning deployment, local LLM integration, model fine-tuning, and production-style software engineering practices.

## Project Name

AI Product Price Predictor

## Version

1.0.0

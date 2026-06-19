from pathlib import Path

BASE_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
OUTPUT_DIR = "product_price_lora"
MAX_SEQ_LENGTH = 2048


def install_dependencies():
    return """
!pip install -U unsloth
!pip install -U datasets transformers trl peft accelerate bitsandbytes
"""


def training_code():
    return f"""
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

BASE_MODEL_NAME = "{BASE_MODEL_NAME}"
OUTPUT_DIR = "{OUTPUT_DIR}"
MAX_SEQ_LENGTH = {MAX_SEQ_LENGTH}

dataset = load_dataset(
    "json",
    data_files={{
        "train": "train.jsonl",
        "validation": "validation.jsonl",
    }},
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="messages",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        fp16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    ),
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
"""


def zip_adapter_code():
    return """
!zip -r product_price_lora.zip product_price_lora
"""


def main():
    output_path = Path("training/colab_generated_training_cells.txt")

    content = "\n\n".join(
        [
            "# Cell 1: Install dependencies",
            install_dependencies(),
            "# Cell 2: Train LoRA adapter",
            training_code(),
            "# Cell 3: Zip adapter",
            zip_adapter_code(),
        ]
    )

    output_path.write_text(content, encoding="utf-8")
    print(f"Generated Colab training cells: {output_path}")


if __name__ == "__main__":
    main()
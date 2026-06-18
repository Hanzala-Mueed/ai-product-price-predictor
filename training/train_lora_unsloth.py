from pathlib import Path
from config.logging_config import setup_logging

logger = setup_logging()

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.jsonl"
VALIDATION_FILE = BASE_DIR / "data" / "processed" / "validation.jsonl"
OUTPUT_DIR = BASE_DIR / "models" / "adapters" / "product_price_lora"

BASE_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048


def validate_training_files() -> bool:
    try:
        if not TRAIN_FILE.exists():
            raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

        if not VALIDATION_FILE.exists():
            raise FileNotFoundError(f"Validation file not found: {VALIDATION_FILE}")

        logger.info("Training files validated successfully.")
        return True

    except Exception as e:
        logger.exception("Training file validation failed.")
        raise RuntimeError(f"Training file validation failed: {e}") from e


def train_lora():
    try:
        validate_training_files()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Heavy imports stay inside this function so normal tests do not require GPU packages.
        from datasets import load_dataset
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments

        logger.info("Loading dataset from JSONL files.")

        dataset = load_dataset(
            "json",
            data_files={
                "train": str(TRAIN_FILE),
                "validation": str(VALIDATION_FILE),
            },
        )

        logger.info("Loading base model: %s", BASE_MODEL_NAME)

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
                output_dir=str(OUTPUT_DIR),
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

        logger.info("Starting LoRA fine-tuning.")
        trainer.train()

        logger.info("Saving LoRA adapter to %s", OUTPUT_DIR)
        model.save_pretrained(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))

        logger.info("LoRA fine-tuning completed successfully.")
        return OUTPUT_DIR

    except Exception as e:
        logger.exception("LoRA fine-tuning failed.")
        raise RuntimeError(f"LoRA fine-tuning failed: {e}") from e


if __name__ == "__main__":
    train_lora()
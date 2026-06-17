import csv
import json
import random
from pathlib import Path
from typing import List, Dict

from config.logging_config import setup_logging

logger = setup_logging()


class FineTuneDataBuilder:
    def __init__(
        self,
        input_csv: str | Path,
        output_dir: str | Path,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.seed = seed

    def load_rows(self) -> List[Dict[str, str]]:
        try:
            if not self.input_csv.exists():
                raise FileNotFoundError(f"Input CSV not found: {self.input_csv}")

            rows = []

            with self.input_csv.open("r", encoding="utf-8", newline="") as file:
                reader = csv.reader(file)

                for row in reader:
                    if len(row) < 2:
                        continue

                    product_text = row[0].strip()
                    price = row[1].strip()

                    if not product_text or not price:
                        continue

                    try:
                        price_value = round(float(price))
                    except ValueError:
                        continue

                    rows.append(
                        {
                            "product_text": product_text,
                            "price": str(price_value),
                        }
                    )

            if not rows:
                raise ValueError("No valid rows found in CSV.")

            logger.info("Loaded %s valid rows from %s", len(rows), self.input_csv)
            return rows

        except Exception as e:
            logger.exception("Failed to load fine-tuning CSV data.")
            raise RuntimeError(f"Failed to load fine-tuning CSV data: {e}") from e

    def format_example(self, row: Dict[str, str]) -> Dict:
        try:
            return {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI product pricing assistant. "
                            "Estimate the product price to the nearest dollar. "
                            "Respond only with the price number."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "What does this product cost to the nearest dollar?\n\n"
                            f"{row['product_text']}\n\n"
                            "Price is $"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": row["price"],
                    },
                ]
            }

        except Exception as e:
            logger.exception("Failed to format fine-tuning example.")
            raise RuntimeError(f"Failed to format fine-tuning example: {e}") from e

    def split_rows(self, rows: List[Dict[str, str]]):
        try:
            random.seed(self.seed)
            rows = rows.copy()
            random.shuffle(rows)

            total = len(rows)
            train_end = int(total * self.train_ratio)
            validation_end = train_end + int(total * self.validation_ratio)

            train = rows[:train_end]
            validation = rows[train_end:validation_end]
            test = rows[validation_end:]

            logger.info(
                "Split rows into train=%s, validation=%s, test=%s",
                len(train),
                len(validation),
                len(test),
            )

            return train, validation, test

        except Exception as e:
            logger.exception("Failed to split fine-tuning rows.")
            raise RuntimeError(f"Failed to split fine-tuning rows: {e}") from e

    def write_jsonl(self, rows: List[Dict[str, str]], filename: str) -> Path:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / filename

            with output_path.open("w", encoding="utf-8") as file:
                for row in rows:
                    example = self.format_example(row)
                    file.write(json.dumps(example, ensure_ascii=False) + "\n")

            logger.info("Wrote %s examples to %s", len(rows), output_path)
            return output_path

        except Exception as e:
            logger.exception("Failed to write JSONL file.")
            raise RuntimeError(f"Failed to write JSONL file: {e}") from e

    def build(self) -> Dict[str, Path]:
        try:
            rows = self.load_rows()
            train, validation, test = self.split_rows(rows)

            paths = {
                "train": self.write_jsonl(train, "train.jsonl"),
                "validation": self.write_jsonl(validation, "validation.jsonl"),
                "test": self.write_jsonl(test, "test.jsonl"),
            }

            logger.info("Fine-tuning dataset built successfully.")
            return paths

        except Exception as e:
            logger.exception("Failed to build fine-tuning dataset.")
            raise RuntimeError(f"Failed to build fine-tuning dataset: {e}") from e
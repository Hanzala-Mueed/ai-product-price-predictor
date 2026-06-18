import csv
from pathlib import Path
from typing import List

from pricer.items import Item
from config.logging_config import setup_logging

logger = setup_logging()


class DatasetBuilder:
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    def load_items(self, limit: int | None = None) -> List[Item]:
        try:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

            items = []

            with self.csv_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.reader(file)

                for index, row in enumerate(reader):
                    if limit is not None and len(items) >= limit:
                        break

                    if len(row) < 2:
                        continue

                    product_text = row[0].strip()
                    price_text = row[1].strip()

                    if not product_text or not price_text:
                        continue

                    try:
                        price = float(price_text)
                    except ValueError:
                        continue

                    item = Item(
                        id=index,
                        title=self.extract_title(product_text),
                        category=self.extract_category(product_text),
                        price=price,
                        summary=product_text,
                    )
                    items.append(item)

            if not items:
                raise ValueError("No valid items found in dataset.")

            logger.info("Loaded %s items from %s", len(items), self.csv_path)
            return items

        except Exception as e:
            logger.exception("Failed to load items from dataset.")
            raise RuntimeError(f"Failed to load items from dataset: {e}") from e

    def extract_title(self, text: str) -> str:
        try:
            for line in text.splitlines():
                if line.lower().startswith("title:"):
                    title = line.split(":", 1)[1].strip()
                    return title or "Unknown Product"
            return text.splitlines()[0][:80] if text else "Unknown Product"

        except Exception as e:
            logger.exception("Failed to extract title.")
            raise RuntimeError(f"Failed to extract title: {e}") from e

    def extract_category(self, text: str) -> str:
        try:
            for line in text.splitlines():
                if line.lower().startswith("category:"):
                    category = line.split(":", 1)[1].strip()
                    return category or "Unknown"
            return "Unknown"

        except Exception as e:
            logger.exception("Failed to extract category.")
            raise RuntimeError(f"Failed to extract category: {e}") from e
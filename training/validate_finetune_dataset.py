import json
from pathlib import Path
from typing import Dict, List

from config.logging_config import setup_logging

logger = setup_logging()

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.jsonl"
VALIDATION_FILE = BASE_DIR / "data" / "processed" / "validation.jsonl"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.jsonl"


class FineTuneDatasetValidator:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def validate_message(self, message: Dict) -> bool:
        try:
            if not isinstance(message, dict):
                raise ValueError("Message must be a dictionary.")

            if "role" not in message:
                raise ValueError("Message missing role.")

            if "content" not in message:
                raise ValueError("Message missing content.")

            if message["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"Invalid role: {message['role']}")

            if not isinstance(message["content"], str) or not message["content"].strip():
                raise ValueError("Message content cannot be empty.")

            return True

        except Exception as e:
            logger.exception("Invalid message format.")
            raise RuntimeError(f"Invalid message format: {e}") from e

    def validate_example(self, example: Dict) -> bool:
        try:
            if not isinstance(example, dict):
                raise ValueError("Example must be a dictionary.")

            if "messages" not in example:
                raise ValueError("Example missing messages field.")

            messages = example["messages"]

            if not isinstance(messages, list):
                raise ValueError("Messages must be a list.")

            if len(messages) != 3:
                raise ValueError("Each example must contain exactly 3 messages.")

            expected_roles = ["system", "user", "assistant"]
            actual_roles = [message.get("role") for message in messages]

            if actual_roles != expected_roles:
                raise ValueError(
                    f"Invalid message role order. Expected {expected_roles}, got {actual_roles}"
                )

            for message in messages:
                self.validate_message(message)

            assistant_content = messages[2]["content"].strip()

            try:
                price = float(assistant_content)
            except ValueError:
                raise ValueError("Assistant content must be a numeric price.")

            if price < 0:
                raise ValueError("Price cannot be negative.")

            return True

        except Exception as e:
            logger.exception("Invalid fine-tuning example.")
            raise RuntimeError(f"Invalid fine-tuning example: {e}") from e

    def validate_file(self) -> Dict:
        try:
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")

            total = 0
            valid = 0
            errors: List[str] = []

            with self.file_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    total += 1
                    line = line.strip()

                    if not line:
                        errors.append(f"Line {line_number}: empty line")
                        continue

                    try:
                        example = json.loads(line)
                        self.validate_example(example)
                        valid += 1
                    except Exception as e:
                        errors.append(f"Line {line_number}: {e}")

            report = {
                "file": str(self.file_path),
                "total": total,
                "valid": valid,
                "invalid": total - valid,
                "errors": errors,
                "is_valid": total > 0 and total == valid,
            }

            logger.info(
                "Validated %s: total=%s valid=%s invalid=%s",
                self.file_path,
                total,
                valid,
                total - valid,
            )

            return report

        except Exception as e:
            logger.exception("Failed to validate fine-tuning dataset file.")
            raise RuntimeError(f"Failed to validate fine-tuning dataset file: {e}") from e


def validate_all_files() -> Dict:
    try:
        reports = {
            "train": FineTuneDatasetValidator(TRAIN_FILE).validate_file(),
            "validation": FineTuneDatasetValidator(VALIDATION_FILE).validate_file(),
            "test": FineTuneDatasetValidator(TEST_FILE).validate_file(),
        }

        all_valid = all(report["is_valid"] for report in reports.values())

        reports["all_valid"] = all_valid

        logger.info("All fine-tuning dataset validation completed.")
        return reports

    except Exception as e:
        logger.exception("Failed to validate all fine-tuning files.")
        raise RuntimeError(f"Failed to validate all fine-tuning files: {e}") from e


if __name__ == "__main__":
    result = validate_all_files()

    for name, report in result.items():
        if name == "all_valid":
            continue

        print(f"\n{name.upper()} FILE")
        print(f"File: {report['file']}")
        print(f"Total: {report['total']}")
        print(f"Valid: {report['valid']}")
        print(f"Invalid: {report['invalid']}")

        if report["errors"]:
            print("Errors:")
            for error in report["errors"][:10]:
                print(f"- {error}")

    print(f"\nAll files valid: {result['all_valid']}")
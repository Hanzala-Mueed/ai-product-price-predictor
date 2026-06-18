import csv
from pathlib import Path

from pricer.dataset_builder import DatasetBuilder
from pricer.predictor import PricePredictor
from pricer.evaluator import PriceEvaluator
from config.logging_config import setup_logging

logger = setup_logging()

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = BASE_DIR / "data" / "raw" / "human_out.csv"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "base_llama_results.csv"

EVALUATION_LIMIT = 10


def main():
    try:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

        builder = DatasetBuilder(INPUT_CSV)
        items = builder.load_items(limit=EVALUATION_LIMIT)

        predictor = PricePredictor()
        evaluator = PriceEvaluator(predictor.predict)

        evaluation = evaluator.evaluate(items)

        with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "title",
                    "category",
                    "actual",
                    "predicted",
                    "absolute_error",
                    "percentage_error",
                    "within_20_percent",
                ],
            )
            writer.writeheader()
            writer.writerows(evaluation.results)

        print("Base Llama 3.2 Evaluation Completed")
        print(f"Total items: {evaluation.total_items}")
        print(f"Average error: ${evaluation.average_error:.2f}")
        print(f"Median error: ${evaluation.median_error:.2f}")
        print(f"Min error: ${evaluation.min_error:.2f}")
        print(f"Max error: ${evaluation.max_error:.2f}")
        print(f"Accuracy within 20%: {evaluation.accuracy_within_20_percent:.2%}")
        print(f"Results saved to: {OUTPUT_CSV}")

    except Exception as e:
        logger.exception("Base model evaluation failed.")
        raise RuntimeError(f"Base model evaluation failed: {e}") from e


if __name__ == "__main__":
    main()
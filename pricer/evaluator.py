from dataclasses import dataclass
from typing import Callable, List, Dict
import statistics

from pricer.items import Item
from config.logging_config import setup_logging

logger = setup_logging()


@dataclass
class EvaluationResult:
    total_items: int
    average_error: float
    median_error: float
    min_error: float
    max_error: float
    accuracy_within_20_percent: float
    results: List[Dict]


class PriceEvaluator:
    def __init__(self, predictor: Callable[[Item], float]):
        self.predictor = predictor

    def calculate_error(self, predicted: float, actual: float) -> Dict:
        try:
            absolute_error = abs(predicted - actual)

            if actual == 0:
                percentage_error = 0
            else:
                percentage_error = absolute_error / actual

            return {
                "predicted": predicted,
                "actual": actual,
                "absolute_error": absolute_error,
                "percentage_error": percentage_error,
                "within_20_percent": percentage_error <= 0.20,
            }

        except Exception as e:
            logger.exception("Failed to calculate prediction error.")
            raise RuntimeError(f"Failed to calculate prediction error: {e}") from e

    def evaluate_item(self, item: Item) -> Dict:
        try:
            if item.price is None:
                raise ValueError("Item must have an actual price for evaluation.")

            predicted_price = self.predictor(item)
            error_data = self.calculate_error(predicted_price, item.price)

            result = {
                "title": item.title,
                "category": item.category,
                **error_data,
            }

            logger.info(
                "Evaluated item '%s': predicted=%s actual=%s error=%s",
                item.title,
                predicted_price,
                item.price,
                error_data["absolute_error"],
            )

            return result

        except Exception as e:
            logger.exception("Failed to evaluate item: %s", getattr(item, "title", "Unknown"))
            raise RuntimeError(f"Failed to evaluate item: {e}") from e

    def evaluate(self, items: List[Item]) -> EvaluationResult:
        try:
            if not items:
                raise ValueError("Items list cannot be empty.")

            results = [self.evaluate_item(item) for item in items]
            errors = [result["absolute_error"] for result in results]

            total_items = len(results)
            within_20_count = sum(1 for result in results if result["within_20_percent"])

            evaluation = EvaluationResult(
                total_items=total_items,
                average_error=statistics.mean(errors),
                median_error=statistics.median(errors),
                min_error=min(errors),
                max_error=max(errors),
                accuracy_within_20_percent=within_20_count / total_items,
                results=results,
            )

            logger.info(
                "Evaluation completed: total=%s average_error=%s accuracy_within_20_percent=%s",
                evaluation.total_items,
                evaluation.average_error,
                evaluation.accuracy_within_20_percent,
            )

            return evaluation

        except Exception as e:
            logger.exception("Failed to evaluate price predictions.")
            raise RuntimeError(f"Failed to evaluate price predictions: {e}") from e
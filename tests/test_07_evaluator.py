import pytest

from pricer.items import Item
from pricer.evaluator import PriceEvaluator, EvaluationResult


def fake_predictor(item: Item) -> float:
    return 100.0


def test_calculate_error():
    evaluator = PriceEvaluator(fake_predictor)

    result = evaluator.calculate_error(predicted=120, actual=100)

    assert result["predicted"] == 120
    assert result["actual"] == 100
    assert result["absolute_error"] == 20
    assert result["percentage_error"] == 0.20
    assert result["within_20_percent"] is True


def test_calculate_error_outside_20_percent():
    evaluator = PriceEvaluator(fake_predictor)

    result = evaluator.calculate_error(predicted=150, actual=100)

    assert result["absolute_error"] == 50
    assert result["percentage_error"] == 0.50
    assert result["within_20_percent"] is False


def test_evaluate_item():
    item = Item(
        title="Wireless Mouse",
        category="Electronics",
        price=80,
        summary="Wireless mouse with ergonomic design.",
    )

    evaluator = PriceEvaluator(fake_predictor)
    result = evaluator.evaluate_item(item)

    assert result["title"] == "Wireless Mouse"
    assert result["category"] == "Electronics"
    assert result["predicted"] == 100
    assert result["actual"] == 80
    assert result["absolute_error"] == 20


def test_evaluate_item_requires_price():
    item = Item(
        title="Wireless Mouse",
        category="Electronics",
        summary="Wireless mouse with ergonomic design.",
    )

    evaluator = PriceEvaluator(fake_predictor)

    with pytest.raises(RuntimeError):
        evaluator.evaluate_item(item)


def test_evaluate_multiple_items():
    items = [
        Item(title="Product A", category="Electronics", price=100, summary="A"),
        Item(title="Product B", category="Home", price=80, summary="B"),
        Item(title="Product C", category="Office", price=120, summary="C"),
    ]

    evaluator = PriceEvaluator(fake_predictor)
    result = evaluator.evaluate(items)

    assert isinstance(result, EvaluationResult)
    assert result.total_items == 3
    assert result.average_error == pytest.approx(13.3333, rel=1e-3)
    assert result.median_error == 20
    assert result.min_error == 0
    assert result.max_error == 20
    assert result.accuracy_within_20_percent == pytest.approx(2 / 3)


def test_evaluate_rejects_empty_list():
    evaluator = PriceEvaluator(fake_predictor)

    with pytest.raises(RuntimeError):
        evaluator.evaluate([])
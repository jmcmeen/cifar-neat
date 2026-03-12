"""Tests for the evaluation function."""

from test import evaluate


class _FakeNetwork:
    """Minimal mock that returns fixed outputs per input index."""

    def __init__(self, predictions: list[int], num_classes: int) -> None:
        self._predictions = predictions
        self._num_classes = num_classes
        self._call = 0

    def activate(self, inputs: list[float]) -> list[float]:
        pred = self._predictions[self._call]
        self._call += 1
        output = [0.0] * self._num_classes
        output[pred] = 1.0
        return output


class TestEvaluate:
    def test_perfect_accuracy(self) -> None:
        images = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        labels = [0, 1, 0]
        net = _FakeNetwork(predictions=[0, 1, 0], num_classes=2)
        accuracy, confusion = evaluate(net, images, labels, num_classes=2)  # type: ignore[arg-type]
        assert accuracy == 1.0
        assert confusion == [[2, 0], [0, 1]]

    def test_zero_accuracy(self) -> None:
        images = [[0.0], [1.0]]
        labels = [0, 1]
        net = _FakeNetwork(predictions=[1, 0], num_classes=2)
        accuracy, confusion = evaluate(net, images, labels, num_classes=2)  # type: ignore[arg-type]
        assert accuracy == 0.0
        assert confusion == [[0, 1], [1, 0]]

    def test_partial_accuracy(self) -> None:
        images = [[0.0]] * 4
        labels = [0, 0, 1, 1]
        net = _FakeNetwork(predictions=[0, 1, 1, 0], num_classes=2)
        accuracy, confusion = evaluate(net, images, labels, num_classes=2)  # type: ignore[arg-type]
        assert accuracy == 0.5

    def test_empty_labels(self) -> None:
        net = _FakeNetwork(predictions=[], num_classes=2)
        accuracy, confusion = evaluate(net, [], [], num_classes=2)  # type: ignore[arg-type]
        assert accuracy == 0.0

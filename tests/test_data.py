"""Tests for data module utilities."""

from data import CIFAR10_CLASSES, _cache_path


class TestCachePath:
    def test_deterministic(self) -> None:
        p1 = _cache_path([0, 1], 8, train=True)
        p2 = _cache_path([0, 1], 8, train=True)
        assert p1 == p2

    def test_differs_by_split(self) -> None:
        train = _cache_path([0, 1], 8, train=True)
        test = _cache_path([0, 1], 8, train=False)
        assert train != test

    def test_differs_by_classes(self) -> None:
        p1 = _cache_path([0, 1], 8, train=True)
        p2 = _cache_path([0, 2], 8, train=True)
        assert p1 != p2

    def test_differs_by_size(self) -> None:
        p1 = _cache_path([0], 8, train=True)
        p2 = _cache_path([0], 16, train=True)
        assert p1 != p2


class TestCifar10Classes:
    def test_length(self) -> None:
        assert len(CIFAR10_CLASSES) == 10

    def test_known_classes(self) -> None:
        assert CIFAR10_CLASSES[0] == "airplane"
        assert CIFAR10_CLASSES[9] == "truck"

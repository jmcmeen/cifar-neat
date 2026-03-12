"""Shared CIFAR-10 data loading and caching utilities."""

import hashlib
import logging
import pickle
from pathlib import Path

import torchvision
import torchvision.transforms as transforms
from torch import Tensor

logger = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _cache_path(classes: list[int], img_size: int, train: bool) -> Path:
    """Build a deterministic cache filename from the data parameters."""
    split = "train" if train else "test"
    key = f"{split}-{sorted(classes)}-{img_size}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]  # noqa: S324
    return Path("./data") / f"cache_{split}_{digest}.pkl"


def _load_cifar(
    classes: list[int],
    img_size: int,
    train: bool,
    max_per_class: int | None = None,
) -> tuple[list[list[float]], list[int]]:
    """Download CIFAR-10 and return downscaled grayscale arrays for chosen classes.

    Args:
        classes: CIFAR-10 class indices to include.
        img_size: Target side length for square grayscale images.
        train: If True, use training split; otherwise test split.
        max_per_class: Cap per class (None = unlimited, used for test split).

    Returns:
        Tuple of (images, labels) where images are flattened float lists.
    """
    cache = _cache_path(classes, img_size, train)
    if max_per_class is None and cache.exists():
        logger.info("Loading cached dataset from %s", cache)
        with open(cache, "rb") as f:
            return pickle.load(f)  # noqa: S301

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=train, download=True, transform=transform,
    )

    class_map = {c: i for i, c in enumerate(classes)}
    counts = {c: 0 for c in classes}
    images: list[list[float]] = []
    labels: list[int] = []

    for img, label in dataset:
        if label not in class_map:
            continue
        if max_per_class is not None and counts[label] >= max_per_class:
            if all(v >= max_per_class for v in counts.values()):
                break
            continue
        assert isinstance(img, Tensor)
        images.append(img.numpy().flatten().tolist())
        labels.append(class_map[label])
        counts[label] += 1

    if not images:
        split = "train" if train else "test"
        msg = (
            f"No {split} images found for classes {classes}. "
            "Check that class indices are 0-9."
        )
        raise ValueError(msg)

    # Cache the full (uncapped) splits for reuse
    if max_per_class is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump((images, labels), f)
        logger.info("Cached dataset to %s", cache)

    return images, labels


def load_training_data(
    classes: list[int],
    img_size: int,
    samples_per_class: int,
) -> tuple[list[list[float]], list[int], int]:
    """Load CIFAR-10 training subset.

    Returns:
        Tuple of (images, labels, num_classes).
    """
    images, labels = _load_cifar(classes, img_size, train=True, max_per_class=samples_per_class)
    return images, labels, len(classes)


def load_test_data(
    classes: list[int],
    img_size: int,
) -> tuple[list[list[float]], list[int]]:
    """Load full CIFAR-10 test split for the given classes."""
    return _load_cifar(classes, img_size, train=False)

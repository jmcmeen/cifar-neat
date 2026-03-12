"""Shared CIFAR-10 data loading and caching utilities."""

import hashlib
import logging
import pickle
import tarfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_DIR = Path("./data/cifar-10-batches-py")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

TRAIN_BATCHES = [f"data_batch_{i}" for i in range(1, 6)]
TEST_BATCHES = ["test_batch"]


def _ensure_cifar10_downloaded(data_dir: Path = Path("./data")) -> None:
    """Download and extract CIFAR-10 if not already present."""
    if CIFAR10_DIR.exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "cifar-10-python.tar.gz"
    if not archive.exists():
        logger.info("Downloading CIFAR-10...")
        urllib.request.urlretrieve(CIFAR10_URL, archive)  # noqa: S310
    logger.info("Extracting CIFAR-10...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=data_dir)  # noqa: S202


def _load_cifar10_batch(path: Path) -> tuple[np.ndarray, list[int]]:
    """Load a single CIFAR-10 batch file (pickle format)."""
    with open(path, "rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="dtype.*align", category=DeprecationWarning)
            batch = pickle.load(f, encoding="bytes")  # noqa: S301
    images: np.ndarray = batch[b"data"]
    labels: list[int] = batch[b"labels"]
    return images, labels


def _process_image(raw: np.ndarray, img_size: int) -> list[float]:
    """Convert a raw CIFAR-10 image (3072 uint8) to a grayscale, resized, [0,1] float list."""
    # Raw is 3x32x32 flattened in CHW order
    rgb = raw.reshape(3, 32, 32).transpose(1, 2, 0)  # HWC
    img = Image.fromarray(rgb)
    img = img.convert("L")  # grayscale
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten().tolist()


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

    _ensure_cifar10_downloaded()

    batches = TRAIN_BATCHES if train else TEST_BATCHES
    class_map = {c: i for i, c in enumerate(classes)}
    counts = {c: 0 for c in classes}
    images: list[list[float]] = []
    labels: list[int] = []

    for batch_name in batches:
        raw_images, raw_labels = _load_cifar10_batch(CIFAR10_DIR / batch_name)
        for raw, label in zip(raw_images, raw_labels):
            if label not in class_map:
                continue
            if max_per_class is not None and counts[label] >= max_per_class:
                if all(v >= max_per_class for v in counts.values()):
                    break
                continue
            images.append(_process_image(raw, img_size))
            labels.append(class_map[label])
            counts[label] += 1
        else:
            continue
        break  # break outer loop if inner loop broke

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

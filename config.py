"""Configuration loading and validation for CIFAR-NEAT."""

import configparser
import logging
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

CONFIG_PATH = "config.ini"

REQUIRED_TRAINING_KEYS = [
    "classes", "image_size", "samples_per_class", "generations", "winner_file",
]
OPTIONAL_TRAINING_KEYS = {
    "checkpoint_interval": "100",
    "output_dir": "",
}


class TrainingConfig(TypedDict):
    """Typed configuration for the [Training] section."""

    classes: list[int]
    image_size: int
    samples_per_class: int
    generations: int
    winner_file: str
    checkpoint_interval: int
    output_dir: str


def load_training_config(path: str = CONFIG_PATH) -> TrainingConfig:
    """Read and validate the [Training] section from the INI config.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required key is missing from [Training].
        ValueError: If a value cannot be parsed to the expected type.
    """
    if not Path(path).exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    cp = configparser.ConfigParser()
    cp.read(path)

    if "Training" not in cp:
        msg = f"Config file {path} is missing the [Training] section."
        raise KeyError(msg)

    section = cp["Training"]

    # Check required keys
    missing = [k for k in REQUIRED_TRAINING_KEYS if k not in section]
    if missing:
        msg = (
            f"Missing required key(s) in [Training]: {', '.join(missing)}. "
            f"Required keys: {', '.join(REQUIRED_TRAINING_KEYS)}"
        )
        raise KeyError(msg)

    try:
        classes = [int(c.strip()) for c in section["classes"].split(",")]
    except ValueError as e:
        msg = f"Invalid 'classes' value — expected comma-separated integers: {e}"
        raise ValueError(msg) from e

    invalid = [c for c in classes if c < 0 or c > 9]
    if invalid:
        msg = f"Invalid CIFAR-10 class indices (must be 0-9): {invalid}"
        raise ValueError(msg)

    try:
        image_size = int(section["image_size"])
        samples_per_class = int(section["samples_per_class"])
        generations = int(section["generations"])
    except ValueError as e:
        msg = f"Invalid integer value in [Training]: {e}"
        raise ValueError(msg) from e

    checkpoint_interval = int(section.get(
        "checkpoint_interval", OPTIONAL_TRAINING_KEYS["checkpoint_interval"],
    ))
    output_dir = section.get("output_dir", OPTIONAL_TRAINING_KEYS["output_dir"])

    return TrainingConfig(
        classes=classes,
        image_size=image_size,
        samples_per_class=samples_per_class,
        generations=generations,
        winner_file=section["winner_file"],
        checkpoint_interval=checkpoint_interval,
        output_dir=output_dir,
    )

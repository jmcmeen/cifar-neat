"""Load a trained NEAT genome and evaluate it on the CIFAR-10 test set."""

import argparse
import logging
import pickle
from typing import Any

import neat

from config import CONFIG_PATH, load_training_config
from data import CIFAR10_CLASSES, load_test_data

logger = logging.getLogger(__name__)


def evaluate(
    net: neat.nn.FeedForwardNetwork,
    images: list[list[float]],
    labels: list[int],
    num_classes: int,
) -> tuple[float, list[list[int]]]:
    """Compute accuracy and per-class metrics."""
    correct = 0
    confusion: list[list[int]] = [[0] * num_classes for _ in range(num_classes)]

    for img, label in zip(images, labels):
        output: list[float] = net.activate(img)
        prediction = output.index(max(output))
        confusion[label][prediction] += 1
        if prediction == label:
            correct += 1

    accuracy = correct / len(labels) if labels else 0.0
    return accuracy, confusion


def print_results(
    accuracy: float,
    confusion: list[list[int]],
    class_names: list[str],
    total: int,
) -> None:
    """Print accuracy, confusion matrix, and per-class precision."""
    num_classes = len(class_names)
    logger.info("Overall accuracy: %.2f%% (%d/%d)", accuracy * 100, int(accuracy * total), total)
    logger.info("Confusion matrix:")

    header = "".ljust(12) + "".join(name.rjust(12) for name in class_names)
    logger.info(header)

    for i, row in enumerate(confusion):
        row_total = sum(row)
        row_acc = row[i] / row_total if row_total else 0
        line = class_names[i].ljust(12) + "".join(str(v).rjust(12) for v in row)
        line += f"  ({row_acc:.0%})"
        logger.info(line)

    logger.info("Per-class precision:")
    for j, name in enumerate(class_names):
        col_total = sum(confusion[i][j] for i in range(num_classes))
        precision = confusion[j][j] / col_total if col_total else 0
        logger.info("  %s: %.2f%%", name, precision * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test a trained NEAT classifier on CIFAR-10",
    )
    parser.add_argument(
        "--config", default=CONFIG_PATH, help="Path to INI config file",
    )
    parser.add_argument(
        "--genome", required=True, help="Path to saved genome (.pkl)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    training = load_training_config(args.config)
    classes = training["classes"]
    num_classes = len(classes)
    class_names = [CIFAR10_CLASSES[i] for i in classes]

    logger.info("Loading genome from %s...", args.genome)
    with open(args.genome, "rb") as f:
        genome: Any = pickle.load(f)  # noqa: S301

    logger.info("Loading CIFAR-10 test set...")
    images, labels = load_test_data(classes, training["image_size"], training["data_dir"])
    num_inputs = len(images[0])
    logger.info(
        "  %d test samples, %d inputs, %d classes", len(images), num_inputs, num_classes,
    )

    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )
    neat_config.genome_config.num_inputs = num_inputs
    neat_config.genome_config.num_outputs = num_classes
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    accuracy, confusion = evaluate(net, images, labels, num_classes)
    print_results(accuracy, confusion, class_names, len(labels))


if __name__ == "__main__":
    main()

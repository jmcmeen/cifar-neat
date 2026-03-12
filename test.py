"""Load a trained NEAT genome and evaluate it on the CIFAR-10 test set."""

import argparse
import pickle
from typing import Any

import neat
import torchvision
import torchvision.transforms as transforms
from torch import Tensor

from train import CONFIG_PATH, load_training_config

TrainingConfig = dict[str, Any]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_test_data(
    training: TrainingConfig,
) -> tuple[list[list[float]], list[int]]:
    """Load the CIFAR-10 test split for the configured classes."""
    classes: list[int] = training["classes"]
    img_size: int = training["image_size"]

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform,
    )

    class_map = {c: i for i, c in enumerate(classes)}
    images: list[list[float]] = []
    labels: list[int] = []

    for img, label in dataset:
        if label in class_map:
            assert isinstance(img, Tensor)
            images.append(img.numpy().flatten().tolist())
            labels.append(class_map[label])

    return images, labels


def evaluate(
    net: neat.nn.FeedForwardNetwork,
    images: list[list[float]],
    labels: list[int],
    num_classes: int,
) -> tuple[float, list[list[int]]]:
    """Compute accuracy and per-class metrics."""
    correct = 0
    # Confusion matrix: confusion[true][predicted]
    confusion: list[list[int]] = [[0] * num_classes for _ in range(num_classes)]

    for img, label in zip(images, labels):
        output = net.activate(img)
        prediction = output.index(max(output))
        confusion[label][prediction] += 1
        if prediction == label:
            correct += 1

    accuracy = correct / len(labels) if labels else 0.0
    return accuracy, confusion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test a trained NEAT classifier on CIFAR-10",
    )
    parser.add_argument(
        "--config", default=CONFIG_PATH, help="Path to INI config file",
    )
    parser.add_argument(
        "--genome", default=None, help="Path to saved genome (overrides config)",
    )
    args = parser.parse_args()

    training = load_training_config(args.config)
    classes: list[int] = training["classes"]
    num_classes = len(classes)
    class_names = [CIFAR10_CLASSES[i] for i in classes]

    # Load genome
    genome_path: str = args.genome or training["winner_file"]
    print(f"Loading genome from {genome_path}...")
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)  # noqa: S301

    # Load test data
    print("Loading CIFAR-10 test set...")
    images, labels = load_test_data(training)
    num_inputs = len(images[0])
    print(f"  {len(images)} test samples, {num_inputs} inputs, {num_classes} classes")

    # Build network from genome
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

    # Evaluate
    accuracy, confusion = evaluate(net, images, labels, num_classes)

    print(f"\nOverall accuracy: {accuracy:.2%} ({int(accuracy * len(labels))}/{len(labels)})")
    print("\nConfusion matrix:")

    # Header
    header = "".ljust(12) + "".join(name.rjust(12) for name in class_names)
    print(header)

    for i, row in enumerate(confusion):
        row_total = sum(row)
        row_acc = row[i] / row_total if row_total else 0
        line = class_names[i].ljust(12) + "".join(str(v).rjust(12) for v in row)
        line += f"  ({row_acc:.0%})"
        print(line)

    # Per-class precision
    print("\nPer-class precision:")
    for j, name in enumerate(class_names):
        col_total = sum(confusion[i][j] for i in range(num_classes))
        precision = confusion[j][j] / col_total if col_total else 0
        print(f"  {name}: {precision:.2%}")


if __name__ == "__main__":
    main()

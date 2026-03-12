"""Train a NEAT neural network on a subset of CIFAR-10."""

import configparser
import pickle
from typing import Any, Callable

import neat
import torchvision
import torchvision.transforms as transforms
from torch import Tensor

CONFIG_PATH = "config.ini"

TrainingConfig = dict[str, Any]
EvalGenomes = Callable[[list[tuple[int, Any]], neat.Config], None]


def load_training_config(path: str = CONFIG_PATH) -> TrainingConfig:
    """Read the [Training] section from the INI config."""
    cp = configparser.ConfigParser()
    cp.read(path)
    t = cp["Training"]
    return {
        "classes": [int(c) for c in t["classes"].split(",")],
        "image_size": int(t["image_size"]),
        "samples_per_class": int(t["samples_per_class"]),
        "generations": int(t["generations"]),
        "winner_file": t["winner_file"],
    }


def load_cifar_subset(
    training: TrainingConfig,
) -> tuple[list[list[float]], list[int], int]:
    """Download CIFAR-10 and return downscaled grayscale arrays for the chosen classes."""
    classes: list[int] = training["classes"]
    img_size: int = training["image_size"]
    max_per_class: int = training["samples_per_class"]

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform,
    )

    # Map original class indices to 0..N-1
    class_map = {c: i for i, c in enumerate(classes)}
    counts = {c: 0 for c in classes}
    images: list[list[float]] = []
    labels: list[int] = []

    for img, label in dataset:
        if label in class_map and counts[label] < max_per_class:
            # Flatten the grayscale image to a 1-D list of floats in [0, 1]
            assert isinstance(img, Tensor)
            images.append(img.numpy().flatten().tolist())
            labels.append(class_map[label])
            counts[label] += 1
        if all(v >= max_per_class for v in counts.values()):
            break

    return images, labels, len(classes)


def make_eval_function(
    images: list[list[float]], labels: list[int], num_classes: int,
) -> EvalGenomes:
    """Return an eval_genomes function closed over the dataset."""

    def eval_genomes(genomes: list[tuple[int, Any]], config: neat.Config) -> None:
        for _genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            correct = 0
            for img, label in zip(images, labels):
                output = net.activate(img)
                prediction = output.index(max(output))
                if prediction == label:
                    correct += 1
            genome.fitness = correct / len(labels)

    return eval_genomes


def main() -> None:
    training = load_training_config()

    print("Loading CIFAR-10 subset...")
    images, labels, num_classes = load_cifar_subset(training)
    num_inputs = len(images[0])
    print(f"  {len(images)} samples, {num_inputs} inputs, {num_classes} classes")

    print("Loading NEAT configuration...")
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )
    # Override with actual values computed from the data
    neat_config.genome_config.num_inputs = num_inputs
    neat_config.genome_config.num_outputs = num_classes

    # Create population and add reporters
    population = neat.Population(neat_config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(100, filename_prefix="neat-checkpoint-"))

    # Run evolution
    eval_fn = make_eval_function(images, labels, num_classes)
    generations: int = training["generations"]
    print(f"Running evolution for up to {generations} generations...")
    winner = population.run(eval_fn, generations)

    # Save winner
    winner_file: str = training["winner_file"]
    with open(winner_file, "wb") as f:
        pickle.dump(winner, f)
    print(f"\nBest genome saved to {winner_file}")
    print(f"Best fitness: {winner.fitness:.4f}")


if __name__ == "__main__":
    main()

"""Train a NEAT neural network on a subset of CIFAR-10."""

import argparse
import csv
import logging
import multiprocessing
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import neat

from config import CONFIG_PATH, TrainingConfig, load_training_config
from data import load_training_data

logger = logging.getLogger(__name__)

EvalGenomes = Callable[[list[tuple[int, Any]], neat.Config], None]


def _evaluate_genome(
    args: tuple[Any, neat.Config, list[list[float]], list[int]],
) -> float:
    """Evaluate a single genome (worker function for multiprocessing)."""
    genome, config, images, labels = args
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    correct = 0
    for img, label in zip(images, labels):
        output = net.activate(img)
        if output.index(max(output)) == label:
            correct += 1
    return correct / len(labels)


def make_eval_function(
    images: list[list[float]],
    labels: list[int],
    num_classes: int,
) -> EvalGenomes:
    """Return an eval_genomes function closed over the dataset.

    Uses multiprocessing to evaluate genomes in parallel.
    """
    pool = multiprocessing.Pool()

    def eval_genomes(genomes: list[tuple[int, Any]], config: neat.Config) -> None:
        jobs = [
            (genome, config, images, labels)
            for _genome_id, genome in genomes
        ]
        fitnesses = pool.map(_evaluate_genome, jobs)
        for (_genome_id, genome), fitness in zip(genomes, fitnesses):
            genome.fitness = fitness

    return eval_genomes


class CsvReporter(neat.reporting.BaseReporter):
    """Write per-generation fitness stats to a CSV file."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.generation: int | None = None
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "best_fitness", "mean_fitness"])

    def start_generation(self, generation: int) -> None:
        self.generation = generation

    def post_evaluate(
        self,
        config: neat.Config,
        population: dict[int, Any],
        species: Any,
        best_genome: Any,
    ) -> None:
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        if not fitnesses:
            return
        best = max(fitnesses)
        mean = sum(fitnesses) / len(fitnesses)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.generation, f"{best:.6f}", f"{mean:.6f}"])


def setup_output_dir(training: TrainingConfig) -> Path:
    """Create a timestamped output directory for this run."""
    base = training["output_dir"]
    if base:
        output_dir = Path(base)
    else:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_evolution(
    training: TrainingConfig,
    config_path: str,
    checkpoint: str | None = None,
) -> None:
    """Load data, configure NEAT, and run evolution."""
    output_dir = setup_output_dir(training)
    logger.info("Output directory: %s", output_dir)

    logger.info("Loading CIFAR-10 subset...")
    images, labels, num_classes = load_training_data(
        training["classes"], training["image_size"], training["samples_per_class"],
    )
    num_inputs = len(images[0])
    logger.info(
        "  %d samples, %d inputs, %d classes", len(images), num_inputs, num_classes,
    )

    eval_fn = make_eval_function(images, labels, num_classes)

    if checkpoint:
        logger.info("Restoring from checkpoint: %s", checkpoint)
        population = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        logger.info("Loading NEAT configuration from %s", config_path)
        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        neat_config.genome_config.num_inputs = num_inputs
        neat_config.genome_config.num_outputs = num_classes
        population = neat.Population(neat_config)

    verbose = training["verbose"]
    if verbose != "quiet":
        population.add_reporter(neat.StdOutReporter(verbose == "full"))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(CsvReporter(output_dir / "fitness.csv"))

    checkpoint_interval = training["checkpoint_interval"]
    population.add_reporter(
        neat.Checkpointer(
            checkpoint_interval,
            filename_prefix=str(output_dir / "neat-checkpoint-"),
        ),
    )

    generations = training["generations"]
    logger.info("Running evolution for up to %d generations...", generations)
    winner = population.run(eval_fn, generations)

    winner_file = output_dir / training["winner_file"]
    with open(winner_file, "wb") as f:
        pickle.dump(winner, f)
    logger.info("Best genome saved to %s", winner_file)
    logger.info("Best fitness: %.4f", winner.fitness)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a NEAT classifier on CIFAR-10",
    )
    parser.add_argument(
        "--config", default=CONFIG_PATH, help="Path to INI config file",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a neat-checkpoint-* file to resume from",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    training = load_training_config(args.config)
    run_evolution(training, args.config, args.checkpoint)


if __name__ == "__main__":
    main()

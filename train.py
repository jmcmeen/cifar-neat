"""Train a NEAT neural network on a subset of CIFAR-10."""

import argparse
import csv
import logging
import multiprocessing
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from multiprocessing.pool import Pool
from typing import Any, Callable

import neat

from config import CONFIG_PATH, TrainingConfig, load_training_config
from data import load_training_data

logger = logging.getLogger(__name__)

EvalGenomes = Callable[[list[tuple[int, Any]], neat.Config], None]


_worker_images: list[list[float]] = []
_worker_labels: list[int] = []


def _init_worker(images: list[list[float]], labels: list[int]) -> None:
    """Initialize shared data in each worker process."""
    global _worker_images, _worker_labels
    _worker_images = images
    _worker_labels = labels


def _evaluate_genome(args: tuple[Any, neat.Config]) -> float:
    """Evaluate a single genome (worker function for multiprocessing)."""
    genome, config = args
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    correct = 0
    for img, label in zip(_worker_images, _worker_labels):
        output = net.activate(img)
        if output.index(max(output)) == label:
            correct += 1
    return correct / len(_worker_labels)


def make_eval_function(
    images: list[list[float]],
    labels: list[int],
    num_classes: int,
    workers: int | None = None,
) -> tuple[EvalGenomes, Pool]:
    """Return an eval_genomes function closed over the dataset and the pool.

    Uses multiprocessing to evaluate genomes in parallel. The dataset is
    copied once into each worker via an initializer instead of being
    serialized with every task.
    """
    pool = multiprocessing.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(images, labels),
    )

    def eval_genomes(genomes: list[tuple[int, Any]], config: neat.Config) -> None:
        jobs = [(genome, config) for _genome_id, genome in genomes]
        fitnesses = pool.map(_evaluate_genome, jobs)
        for (_genome_id, genome), fitness in zip(genomes, fitnesses):
            genome.fitness = fitness

    return eval_genomes, pool


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


class SummaryReporter(neat.reporting.BaseReporter):
    """Print a one-line summary every N generations."""

    def __init__(self, interval: int) -> None:
        self.interval = interval
        self.generation = 0

    def start_generation(self, generation: int) -> None:
        self.generation = generation

    def post_evaluate(
        self,
        config: neat.Config,
        population: dict[int, Any],
        species: Any,
        best_genome: Any,
    ) -> None:
        if self.generation % self.interval != 0:
            return
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        if not fitnesses:
            return
        best = max(fitnesses)
        mean = sum(fitnesses) / len(fitnesses)
        print(
            f"Gen {self.generation:>6d} | best {best:.4f} | "
            f"mean {mean:.4f} | pop {len(fitnesses)} | "
            f"species {len(species.species)}",
        )


class ProgressReporter(neat.reporting.BaseReporter):
    """Overwrite a single terminal line with current generation stats."""

    def __init__(self, total_generations: int) -> None:
        self.total = total_generations
        self.generation = 0

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
        pct = (self.generation + 1) / self.total * 100 if self.total else 0
        sys.stdout.write(
            f"\rGen {self.generation}/{self.total} ({pct:.0f}%) | best {best:.4f} ",
        )
        sys.stdout.flush()

    def found_solution(self, config: neat.Config, generation: int, best: Any) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    def complete_extinction(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


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

    workers = training["workers"] if training["workers"] > 0 else None
    eval_fn, pool = make_eval_function(images, labels, num_classes, workers)
    logger.info("Worker processes: %d", workers or (multiprocessing.cpu_count() or 1))

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
    if verbose == "full":
        population.add_reporter(neat.StdOutReporter(True))
    elif verbose == "brief":
        population.add_reporter(neat.StdOutReporter(False))
    elif verbose == "summary":
        population.add_reporter(SummaryReporter(training["checkpoint_interval"]))
    elif verbose == "progress":
        population.add_reporter(ProgressReporter(training["generations"]))
    # quiet: no stdout reporter

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
    try:
        winner = population.run(eval_fn, generations)
    finally:
        pool.close()
        pool.join()

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

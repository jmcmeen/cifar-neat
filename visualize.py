"""Visualize a trained NEAT genome's network structure using graphviz."""

import argparse
import logging
import pickle
from typing import Any

import graphviz
import neat

from config import CONFIG_PATH, load_training_config
from data import CIFAR10_CLASSES

logger = logging.getLogger(__name__)


def draw_genome(
    genome: Any,
    config: neat.Config,
    input_labels: list[str] | None = None,
    output_labels: list[str] | None = None,
) -> graphviz.Digraph:
    """Build a graphviz Digraph from a NEAT genome."""
    num_inputs = config.genome_config.num_inputs
    num_outputs = config.genome_config.num_outputs
    input_keys = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys

    if input_labels is None:
        input_labels = [f"I{i}" for i in range(num_inputs)]
    if output_labels is None:
        output_labels = [f"O{i}" for i in range(num_outputs)]

    dot = graphviz.Digraph(
        format="png",
        graph_attr={"rankdir": "LR", "bgcolor": "white"},
        node_attr={"fontsize": "10", "style": "filled"},
    )

    # Input nodes
    with dot.subgraph(name="cluster_inputs") as sub:
        sub.attr(label="Inputs", style="dashed", color="gray")
        for i, key in enumerate(input_keys):
            label = input_labels[i] if i < len(input_labels) else f"I{i}"
            sub.node(str(key), label=label, shape="box", fillcolor="#E8F5E9")

    # Output nodes
    with dot.subgraph(name="cluster_outputs") as sub:
        sub.attr(label="Outputs", style="dashed", color="gray")
        for i, key in enumerate(output_keys):
            label = output_labels[i] if i < len(output_labels) else f"O{i}"
            sub.node(str(key), label=label, shape="box", fillcolor="#FFCDD2")

    # Hidden nodes
    for key, node in genome.nodes.items():
        if key not in input_keys and key not in output_keys:
            dot.node(
                str(key),
                label=f"H{key}\n{node.activation}",
                shape="ellipse",
                fillcolor="#BBDEFB",
            )

    # Connections
    for cg in genome.connections.values():
        if not cg.enabled:
            continue
        color = "#2196F3" if cg.weight > 0 else "#F44336"
        width = str(max(0.5, min(3.0, abs(cg.weight))))
        dot.edge(
            str(cg.key[0]),
            str(cg.key[1]),
            color=color,
            penwidth=width,
        )

    return dot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a trained NEAT genome's network structure",
    )
    parser.add_argument(
        "--config", default=CONFIG_PATH, help="Path to INI config file",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--genome", help="Path to saved genome (.pkl)",
    )
    source.add_argument(
        "--checkpoint", help="Path to a neat-checkpoint-* file",
    )
    parser.add_argument(
        "--output", "-o", default="network", help="Output filename without extension (default: network)",
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
    num_inputs = training["image_size"] ** 2

    if args.checkpoint:
        logger.info("Restoring checkpoint from %s...", args.checkpoint)
        population = neat.Checkpointer.restore_checkpoint(args.checkpoint)
        neat_config = population.config
        neat_config.genome_config.num_inputs = num_inputs
        neat_config.genome_config.num_outputs = num_classes
        genome = max(population.population.values(), key=lambda g: g.fitness)
        logger.info("Best genome fitness: %.4f", genome.fitness)
    else:
        logger.info("Loading genome from %s...", args.genome)
        with open(args.genome, "rb") as f:
            genome = pickle.load(f)  # noqa: S301
        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            args.config,
        )
        neat_config.genome_config.num_inputs = num_inputs
        neat_config.genome_config.num_outputs = num_classes

    input_labels = [f"px{i}" for i in range(num_inputs)]
    dot = draw_genome(genome, neat_config, input_labels, class_names)

    dot.render(args.output, cleanup=True)
    logger.info("Saved network graph to %s.png", args.output)


if __name__ == "__main__":
    main()

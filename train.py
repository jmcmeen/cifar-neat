"""Train a NEAT neural network on a subset of CIFAR-10."""

import os
import pickle
import tempfile
import tomllib

import neat
import numpy as np
import torchvision
import torchvision.transforms as transforms


def load_config(path="config.toml"):
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_cifar_subset(cfg):
    """Download CIFAR-10 and return downscaled grayscale arrays for the chosen classes."""
    classes = cfg["training"]["classes"]
    img_size = cfg["training"]["image_size"]
    max_per_class = cfg["training"]["samples_per_class"]

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
    images, labels = [], []

    for img, label in dataset:
        if label in class_map and counts[label] < max_per_class:
            # Flatten the grayscale image to a 1-D list of floats in [0, 1]
            images.append(img.numpy().flatten().tolist())
            labels.append(class_map[label])
            counts[label] += 1
        if all(v >= max_per_class for v in counts.values()):
            break

    return images, labels, len(classes)


def build_neat_config(cfg, num_inputs, num_outputs):
    """Convert the TOML config into the INI format neat-python expects and load it."""
    neat_cfg = cfg["neat"]
    genome = neat_cfg["genome"]
    species = neat_cfg["species"]
    stagnation = neat_cfg["stagnation"]
    reproduction = neat_cfg["reproduction"]

    ini_lines = [
        "[NEAT]",
        f"fitness_criterion     = {neat_cfg['fitness_criterion']}",
        f"fitness_threshold     = {neat_cfg['fitness_threshold']}",
        f"pop_size              = {neat_cfg['pop_size']}",
        f"reset_on_extinction   = {str(neat_cfg['reset_on_extinction']).capitalize()}",
        f"no_fitness_termination = {str(neat_cfg.get('no_fitness_termination', False)).capitalize()}",
        "",
        "[DefaultGenome]",
        f"num_inputs              = {num_inputs}",
        f"num_outputs             = {num_outputs}",
        f"num_hidden              = {genome['num_hidden']}",
        f"feed_forward            = {str(genome['feed_forward']).capitalize()}",
        f"initial_connection      = {genome['initial_connection']}",
        f"activation_default      = {genome['activation_default']}",
        f"activation_mutate_rate  = {genome['activation_mutate_rate']}",
        f"activation_options      = {genome['activation_options']}",
        f"aggregation_default     = {genome['aggregation_default']}",
        f"aggregation_mutate_rate = {genome['aggregation_mutate_rate']}",
        f"aggregation_options     = {genome['aggregation_options']}",
        f"bias_init_mean          = {genome['bias_init_mean']}",
        f"bias_init_stdev         = {genome['bias_init_stdev']}",
        f"bias_init_type          = {genome['bias_init_type']}",
        f"bias_max_value          = {genome['bias_max_value']}",
        f"bias_min_value          = {genome['bias_min_value']}",
        f"bias_mutate_power       = {genome['bias_mutate_power']}",
        f"bias_mutate_rate        = {genome['bias_mutate_rate']}",
        f"bias_replace_rate       = {genome['bias_replace_rate']}",
        f"response_init_mean      = {genome['response_init_mean']}",
        f"response_init_stdev     = {genome['response_init_stdev']}",
        f"response_init_type      = {genome['response_init_type']}",
        f"response_max_value      = {genome['response_max_value']}",
        f"response_min_value      = {genome['response_min_value']}",
        f"response_mutate_power   = {genome['response_mutate_power']}",
        f"response_mutate_rate    = {genome['response_mutate_rate']}",
        f"response_replace_rate   = {genome['response_replace_rate']}",
        f"weight_init_mean        = {genome['weight_init_mean']}",
        f"weight_init_stdev       = {genome['weight_init_stdev']}",
        f"weight_init_type        = {genome['weight_init_type']}",
        f"weight_max_value        = {genome['weight_max_value']}",
        f"weight_min_value        = {genome['weight_min_value']}",
        f"weight_mutate_power     = {genome['weight_mutate_power']}",
        f"weight_mutate_rate      = {genome['weight_mutate_rate']}",
        f"weight_replace_rate     = {genome['weight_replace_rate']}",
        f"conn_add_prob           = {genome['conn_add_prob']}",
        f"conn_delete_prob        = {genome['conn_delete_prob']}",
        f"node_add_prob           = {genome['node_add_prob']}",
        f"node_delete_prob        = {genome['node_delete_prob']}",
        f"single_structural_mutation = {str(genome.get('single_structural_mutation', False)).capitalize()}",
        f"structural_mutation_surer  = {genome.get('structural_mutation_surer', 'default')}",
        f"enabled_default         = {str(genome['enabled_default']).capitalize()}",
        f"enabled_mutate_rate     = {genome['enabled_mutate_rate']}",
        f"compatibility_disjoint_coefficient = {genome['compatibility_disjoint_coefficient']}",
        f"compatibility_weight_coefficient   = {genome['compatibility_weight_coefficient']}",
        "",
        "[DefaultSpeciesSet]",
        f"compatibility_threshold = {species['compatibility_threshold']}",
        "",
        "[DefaultStagnation]",
        f"species_fitness_func = {stagnation['species_fitness_func']}",
        f"max_stagnation       = {stagnation['max_stagnation']}",
        f"species_elitism      = {stagnation['species_elitism']}",
        "",
        "[DefaultReproduction]",
        f"elitism            = {reproduction['elitism']}",
        f"survival_threshold = {reproduction['survival_threshold']}",
        f"min_species_size   = {reproduction['min_species_size']}",
    ]

    # Write to a temp file and load via neat.Config
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False)
    tmp.write("\n".join(ini_lines))
    tmp.close()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        tmp.name,
    )
    os.unlink(tmp.name)
    return config


def make_eval_function(images, labels, num_classes):
    """Return an eval_genomes function closed over the dataset."""

    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            correct = 0
            for img, label in zip(images, labels):
                output = net.activate(img)
                prediction = output.index(max(output))
                if prediction == label:
                    correct += 1
            genome.fitness = correct / len(labels)

    return eval_genomes


def main():
    cfg = load_config()
    training = cfg["training"]

    print("Loading CIFAR-10 subset...")
    images, labels, num_classes = load_cifar_subset(cfg)
    num_inputs = len(images[0])
    print(f"  {len(images)} samples, {num_inputs} inputs, {num_classes} classes")

    print("Building NEAT configuration...")
    neat_config = build_neat_config(cfg, num_inputs, num_classes)

    # Create population and add reporters
    population = neat.Population(neat_config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(10, filename_prefix="neat-checkpoint-"))

    # Run evolution
    eval_fn = make_eval_function(images, labels, num_classes)
    generations = training["generations"]
    print(f"Running evolution for up to {generations} generations...")
    winner = population.run(eval_fn, generations)

    # Save winner
    winner_file = training["winner_file"]
    with open(winner_file, "wb") as f:
        pickle.dump(winner, f)
    print(f"\nBest genome saved to {winner_file}")
    print(f"Best fitness: {winner.fitness:.4f}")


if __name__ == "__main__":
    main()

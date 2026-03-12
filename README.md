# cifar-neat

NEAT (NeuroEvolution of Augmenting Topologies) classifier for CIFAR-10, built with [neat-python](https://github.com/CodeReclaimers/neat-python).

NEAT evolves neural network topology and weights simultaneously through a genetic algorithm. Since NEAT networks are small by design, this project downscales CIFAR-10 images to grayscale and uses a subset of classes to keep the problem tractable.

## Setup

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

## Usage

### Train

```bash
python train.py
```

This will:

1. Download CIFAR-10 (first run only, saved to `./data/`)
2. Run NEAT evolution using the parameters in `config.toml`
3. Save the best genome to `winner-genome.pkl`
4. Write checkpoints every 10 generations as `neat-checkpoint-*`

### Test

```bash
python test.py
```

Loads the winner genome and evaluates it on the CIFAR-10 test split, printing overall accuracy, a confusion matrix, and per-class precision.

To test a specific genome or checkpoint:

```bash
python test.py --genome neat-checkpoint-50
```

## Configuration

All parameters live in `config.toml`:

| Section               | Key settings                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| `[training]`          | `classes` (CIFAR-10 class indices), `image_size` (downscale target), `samples_per_class`, `generations` |
| `[neat]`              | `pop_size`, `fitness_threshold`, `fitness_criterion`                                                    |
| `[neat.genome]`       | Mutation rates, weight/bias ranges, activation functions, structural mutation probabilities             |
| `[neat.species]`      | `compatibility_threshold` for speciation                                                                |
| `[neat.stagnation]`   | `max_stagnation`, `species_elitism`                                                                     |
| `[neat.reproduction]` | `elitism`, `survival_threshold`                                                                         |

CIFAR-10 class indices: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck.

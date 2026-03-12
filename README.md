# cifar-neat

NEAT (NeuroEvolution of Augmenting Topologies) classifier for CIFAR-10, built with [neat-python](https://github.com/CodeReclaimers/neat-python).

NEAT evolves neural network topology and weights simultaneously through a genetic algorithm. Since NEAT networks are small by design, this project downscales CIFAR-10 images to grayscale and uses a subset of classes to keep the problem tractable.

## Setup

Requires Python 3.11+.

```bash
pip install -r requirements.txt
pip install pytest  # for running tests
```

## Usage

### Train

```bash
python train.py
```

This will:

1. Download CIFAR-10 (first run only, cached to `./data/`)
2. Run NEAT evolution using the parameters in `config.ini`
3. Save outputs to a timestamped directory under `runs/` (genome, checkpoints, fitness CSV)

Options:

```bash
python train.py --config path/to/config.ini       # use a different config
python train.py --checkpoint runs/.../neat-checkpoint-50  # resume from checkpoint
```

Genome evaluation is parallelized across CPU cores via `multiprocessing`. Set `workers` in `config.ini` to limit the number of processes.

### Test

```bash
python test.py --genome runs/.../winner-genome.pkl
```

Loads the winner genome and evaluates it on the CIFAR-10 test split, printing overall accuracy, a confusion matrix, and per-class precision.

```bash
python test.py --genome runs/.../winner-genome.pkl --config path/to/config.ini
```

### Visualize

Render the evolved network structure as a graph image.

Requires [Graphviz](https://graphviz.org/download/) installed on your system:

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz
```

```bash
python visualize.py --genome runs/.../winner-genome.pkl
```

Outputs `network.png` showing input nodes, hidden nodes (with activation function), and output nodes with color-coded weighted connections.

```bash
python visualize.py --genome runs/.../winner-genome.pkl --output my-network
python visualize.py --genome runs/.../winner-genome.pkl --config path/to/config.ini
```

## Project Structure

| File             | Purpose                                                      |
| ---------------- | ------------------------------------------------------------ |
| `config.py`      | Config loading, validation, and `TrainingConfig` TypedDict   |
| `data.py`        | Shared CIFAR-10 loading, transforms, and caching             |
| `train.py`       | Training CLI with parallel eval, CSV logging, checkpointing  |
| `test.py`        | Evaluation CLI with confusion matrix and per-class metrics   |
| `visualize.py`   | Network structure visualization using Graphviz               |
| `config.ini`     | All NEAT and training parameters                             |
| `tests/`         | Unit tests for config, data, and evaluation                  |

## Configuration

All parameters live in `config.ini` (INI format, read directly by neat-python):

| Section                 | Key settings                                                                                                                |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `[Training]`            | `classes`, `image_size`, `samples_per_class`, `generations`, `winner_file`, `checkpoint_interval`, `output_dir`, `verbose`  |
| `[NEAT]`                | `pop_size`, `fitness_threshold`, `fitness_criterion`                                                                        |
| `[DefaultGenome]`       | Mutation rates, weight/bias ranges, activation functions, structural mutation probabilities                                 |
| `[DefaultSpeciesSet]`   | `compatibility_threshold` for speciation                                                                                    |
| `[DefaultStagnation]`   | `max_stagnation`, `species_elitism`                                                                                         |
| `[DefaultReproduction]` | `elitism`, `survival_threshold`                                                                                             |

Optional `[Training]` keys:

- `checkpoint_interval` — generations between checkpoints (default: 100, set in `config.ini`)
- `output_dir` — fixed output directory (default: auto-generated `runs/YYYYMMDD_HHMMSS`)
- `verbose` — NEAT stdout reporter verbosity (default: `full`):
  - `full` — all output including per-species detail
  - `brief` — generation summaries, no species breakdown
  - `summary` — one line every `checkpoint_interval` generations
  - `progress` — single updating line with generation and best fitness
  - `quiet` — no stdout output (CSV still logs)
- `workers` — number of worker processes for genome evaluation (default: `0` = all CPU cores)
- `data_dir` — directory for CIFAR-10 downloads and caches (default: `data`)

CIFAR-10 class indices: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck.

## Output

Each training run creates a timestamped directory under `runs/`:

```text
runs/20260312_143000/
├── winner-genome.pkl    # best genome
├── fitness.csv          # per-generation best/mean fitness
└── neat-checkpoint-*    # periodic checkpoints
```

## CI

GitHub Actions runs `ruff check` and `pytest` on every push and PR to `main`. See `.github/workflows/ci.yml`.

## Running Tests

```bash
python -m pytest tests/ -v
```

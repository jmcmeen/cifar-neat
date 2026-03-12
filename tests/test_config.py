"""Tests for config loading and validation."""

import tempfile
from pathlib import Path

import pytest

from config import load_training_config

VALID_INI = """\
[Training]
classes = 0, 3, 5
image_size = 8
samples_per_class = 100
generations = 50
winner_file = winner.pkl

[NEAT]
fitness_criterion = max
fitness_threshold = 0.95
pop_size = 10
reset_on_extinction = False
no_fitness_termination = False

[DefaultGenome]
num_inputs = 64
num_outputs = 3
num_hidden = 0
feed_forward = True
initial_connection = full_direct
activation_default = relu
activation_mutate_rate = 0.0
activation_options = relu
aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_init_type = gaussian
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1
response_init_mean = 1.0
response_init_stdev = 0.0
response_init_type = gaussian
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_init_type = gaussian
weight_max_value = 30.0
weight_min_value = -30.0
weight_mutate_power = 0.5
weight_mutate_rate = 0.8
weight_replace_rate = 0.1
conn_add_prob = 0.5
conn_delete_prob = 0.5
node_add_prob = 0.2
node_delete_prob = 0.2
single_structural_mutation = False
structural_mutation_surer = default
enabled_default = True
enabled_mutate_rate = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 20
species_elitism = 2

[DefaultReproduction]
elitism = 2
survival_threshold = 0.2
min_species_size = 2
"""


def _write_ini(content: str) -> str:
    """Write content to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False)
    f.write(content)
    f.close()
    return f.name


class TestLoadTrainingConfig:
    def test_valid_config(self) -> None:
        path = _write_ini(VALID_INI)
        cfg = load_training_config(path)
        assert cfg["classes"] == [0, 3, 5]
        assert cfg["image_size"] == 8
        assert cfg["samples_per_class"] == 100
        assert cfg["generations"] == 50
        assert cfg["winner_file"] == "winner.pkl"
        assert cfg["checkpoint_interval"] == 100  # default
        assert cfg["verbose"] == "full"  # default
        assert cfg["workers"] == 0  # default
        assert cfg["data_dir"] == "data"  # default
        Path(path).unlink()

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_training_config("/nonexistent/path.ini")

    def test_missing_training_section(self) -> None:
        path = _write_ini("[NEAT]\npop_size = 10\n")
        with pytest.raises(KeyError, match="Training"):
            load_training_config(path)
        Path(path).unlink()

    def test_missing_required_key(self) -> None:
        ini = "[Training]\nclasses = 0,1\nimage_size = 8\n"
        path = _write_ini(ini)
        with pytest.raises(KeyError, match="samples_per_class"):
            load_training_config(path)
        Path(path).unlink()

    def test_invalid_class_index(self) -> None:
        ini = VALID_INI.replace("classes = 0, 3, 5", "classes = 0, 15")
        path = _write_ini(ini)
        with pytest.raises(ValueError, match="0-9"):
            load_training_config(path)
        Path(path).unlink()

    def test_non_integer_class(self) -> None:
        ini = VALID_INI.replace("classes = 0, 3, 5", "classes = cat, dog")
        path = _write_ini(ini)
        with pytest.raises(ValueError, match="comma-separated integers"):
            load_training_config(path)
        Path(path).unlink()

    def test_invalid_verbose(self) -> None:
        ini = VALID_INI.replace(
            "winner_file = winner.pkl",
            "winner_file = winner.pkl\nverbose = loud",
        )
        path = _write_ini(ini)
        with pytest.raises(ValueError, match="full, brief, summary, progress, quiet"):
            load_training_config(path)
        Path(path).unlink()

    def test_custom_verbose(self) -> None:
        ini = VALID_INI.replace(
            "winner_file = winner.pkl",
            "winner_file = winner.pkl\nverbose = quiet",
        )
        path = _write_ini(ini)
        cfg = load_training_config(path)
        assert cfg["verbose"] == "quiet"
        Path(path).unlink()

    def test_custom_workers(self) -> None:
        ini = VALID_INI.replace(
            "winner_file = winner.pkl",
            "winner_file = winner.pkl\nworkers = 4",
        )
        path = _write_ini(ini)
        cfg = load_training_config(path)
        assert cfg["workers"] == 4
        Path(path).unlink()

    def test_invalid_workers(self) -> None:
        ini = VALID_INI.replace(
            "winner_file = winner.pkl",
            "winner_file = winner.pkl\nworkers = -1",
        )
        path = _write_ini(ini)
        with pytest.raises(ValueError, match="workers"):
            load_training_config(path)
        Path(path).unlink()

    def test_custom_data_dir(self) -> None:
        ini = VALID_INI.replace(
            "winner_file = winner.pkl",
            "winner_file = winner.pkl\ndata_dir = /tmp/cifar",
        )
        path = _write_ini(ini)
        cfg = load_training_config(path)
        assert cfg["data_dir"] == "/tmp/cifar"
        Path(path).unlink()

    def test_custom_checkpoint_interval(self) -> None:
        ini = VALID_INI.replace(
            "winner_file = winner.pkl",
            "winner_file = winner.pkl\ncheckpoint_interval = 50",
        )
        path = _write_ini(ini)
        cfg = load_training_config(path)
        assert cfg["checkpoint_interval"] == 50
        Path(path).unlink()

"""Tests for the draw_genome visualization function."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import graphviz
import pytest

from visualize import draw_genome, main


def _make_config(num_inputs: int, num_outputs: int) -> SimpleNamespace:
    """Build a minimal fake config matching neat.Config.genome_config shape."""
    input_keys = list(range(-num_inputs, 0))
    output_keys = list(range(num_outputs))
    genome_config = SimpleNamespace(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        input_keys=input_keys,
        output_keys=output_keys,
    )
    return SimpleNamespace(genome_config=genome_config)


def _make_connection(key: tuple[int, int], weight: float, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(key=key, weight=weight, enabled=enabled)


def _make_node(activation: str = "relu") -> SimpleNamespace:
    return SimpleNamespace(activation=activation)


class TestDrawGenome:
    def test_returns_digraph(self) -> None:
        config = _make_config(2, 2)
        genome = SimpleNamespace(nodes={0: _make_node(), 1: _make_node()}, connections={})
        dot = draw_genome(genome, config)
        assert isinstance(dot, graphviz.Digraph)

    def test_input_and_output_nodes(self) -> None:
        config = _make_config(2, 2)
        genome = SimpleNamespace(nodes={0: _make_node(), 1: _make_node()}, connections={})
        dot = draw_genome(genome, config)
        source = dot.source
        # Input keys are -2 and -1
        assert "-2" in source
        assert "-1" in source
        # Output keys are 0 and 1 (with label attributes)
        assert "0 [label=O0" in source
        assert "1 [label=O1" in source

    def test_custom_labels(self) -> None:
        config = _make_config(2, 2)
        genome = SimpleNamespace(nodes={0: _make_node(), 1: _make_node()}, connections={})
        dot = draw_genome(genome, config, input_labels=["px0", "px1"], output_labels=["cat", "dog"])
        source = dot.source
        assert "px0" in source
        assert "px1" in source
        assert "cat" in source
        assert "dog" in source

    def test_hidden_node(self) -> None:
        config = _make_config(1, 1)
        genome = SimpleNamespace(
            nodes={0: _make_node(), 5: _make_node("sigmoid")},
            connections={},
        )
        dot = draw_genome(genome, config)
        source = dot.source
        assert "H5" in source
        assert "sigmoid" in source

    def test_connections(self) -> None:
        config = _make_config(1, 1)
        conn = _make_connection((-1, 0), weight=1.5)
        genome = SimpleNamespace(
            nodes={0: _make_node()},
            connections={(-1, 0): conn},
        )
        dot = draw_genome(genome, config)
        source = dot.source
        assert "-1 -> 0" in source

    def test_disabled_connections_excluded(self) -> None:
        config = _make_config(1, 1)
        conn = _make_connection((-1, 0), weight=1.0, enabled=False)
        genome = SimpleNamespace(
            nodes={0: _make_node()},
            connections={(-1, 0): conn},
        )
        dot = draw_genome(genome, config)
        source = dot.source
        assert "-1 -> 0" not in source

    def test_positive_weight_color(self) -> None:
        config = _make_config(1, 1)
        conn = _make_connection((-1, 0), weight=2.0)
        genome = SimpleNamespace(
            nodes={0: _make_node()},
            connections={(-1, 0): conn},
        )
        dot = draw_genome(genome, config)
        # Blue for positive weights
        assert "#2196F3" in dot.source

    def test_negative_weight_color(self) -> None:
        config = _make_config(1, 1)
        conn = _make_connection((-1, 0), weight=-2.0)
        genome = SimpleNamespace(
            nodes={0: _make_node()},
            connections={(-1, 0): conn},
        )
        dot = draw_genome(genome, config)
        # Red for negative weights
        assert "#F44336" in dot.source


class TestMainCheckpoint:
    """Tests for the --checkpoint path through main()."""

    def _make_population(self) -> MagicMock:
        """Build a fake population returned by restore_checkpoint."""
        genome_a = SimpleNamespace(
            fitness=0.6,
            nodes={0: _make_node()},
            connections={},
        )
        genome_b = SimpleNamespace(
            fitness=0.9,
            nodes={0: _make_node()},
            connections={},
        )
        pop = MagicMock()
        pop.population = {1: genome_a, 2: genome_b}
        pop.config.genome_config.num_inputs = 4
        pop.config.genome_config.num_outputs = 2
        pop.config.genome_config.input_keys = [-4, -3, -2, -1]
        pop.config.genome_config.output_keys = [0, 1]
        return pop

    @patch("visualize.load_training_config")
    @patch("visualize.neat.Checkpointer.restore_checkpoint")
    def test_checkpoint_selects_best_genome(
        self, mock_restore: MagicMock, mock_load_config: MagicMock, tmp_path: pytest.TempPathFactory,
    ) -> None:
        mock_load_config.return_value = {
            "classes": [3, 5],
            "image_size": 2,
        }
        pop = self._make_population()
        mock_restore.return_value = pop

        output = str(tmp_path / "test-net")
        with patch("sys.argv", ["visualize.py", "--checkpoint", "fake-checkpoint", "--output", output]):
            main()

        mock_restore.assert_called_once_with("fake-checkpoint")
        # Should have rendered an output file
        assert (tmp_path / "test-net.png").exists()

    @patch("visualize.load_training_config")
    @patch("visualize.neat.Checkpointer.restore_checkpoint")
    def test_checkpoint_uses_training_config_dimensions(
        self, mock_restore: MagicMock, mock_load_config: MagicMock, tmp_path: pytest.TempPathFactory,
    ) -> None:
        mock_load_config.return_value = {
            "classes": [3, 5],
            "image_size": 2,
        }
        pop = self._make_population()
        mock_restore.return_value = pop

        output = str(tmp_path / "test-net")
        with patch("sys.argv", ["visualize.py", "--checkpoint", "fake-checkpoint", "--output", output]):
            main()

        # num_inputs = image_size**2 = 4, num_outputs = len(classes) = 2
        assert pop.config.genome_config.num_inputs == 4
        assert pop.config.genome_config.num_outputs == 2

    def test_genome_and_checkpoint_mutually_exclusive(self) -> None:
        with patch("sys.argv", ["visualize.py", "--genome", "a.pkl", "--checkpoint", "b"]):
            with pytest.raises(SystemExit):
                main()

    def test_requires_genome_or_checkpoint(self) -> None:
        with patch("sys.argv", ["visualize.py"]):
            with pytest.raises(SystemExit):
                main()

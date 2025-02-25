"""
Tests for the Observation Basenet implementation.
"""

from typing import Tuple

import pytest
import torch
from torch import Tensor

from SampleEfficientRL.Agents.RL.Networks.ObservationBasenet import (
    ObservationBasenet,
    ObservationBasenetConfig,
    observation_basenet_small,
)
from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import (
    NUMBER_ENCODING_DIMS,
)


def create_mock_state_tuple(
    seq_len: int = 10,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create a mock state tuple for testing."""
    token_types = torch.randint(0, 7, (seq_len,))
    card_uid_indices = torch.randint(0, 4, (seq_len,))
    status_uid_indices = torch.randint(0, 7, (seq_len,))
    enemy_intent_indices = torch.randint(0, 3, (seq_len,))
    # Create encoded numbers with the new format (12 dimensions)
    encoded_numbers = torch.rand((seq_len, NUMBER_ENCODING_DIMS))

    return (
        token_types,
        card_uid_indices,
        status_uid_indices,
        enemy_intent_indices,
        encoded_numbers,
    )


class TestObservationBasenet:
    """Test the ObservationBasenet implementation."""

    def test_init(self) -> None:
        """Test that initialization works correctly."""
        model = observation_basenet_small()
        assert isinstance(model, ObservationBasenet)
        assert isinstance(model.config, ObservationBasenetConfig)

    def test_forward_pass(self) -> None:
        """Test that forward pass produces output of expected shape and type."""
        model = observation_basenet_small()
        state_tuple = create_mock_state_tuple()

        with torch.no_grad():
            latent = model(state_tuple)

        assert isinstance(latent, torch.Tensor)
        assert latent.shape == (1, model.config.latent_dim)
        assert not torch.isnan(latent).any()

    def test_batch_processing(self) -> None:
        """Test that batch processing works correctly."""
        model = observation_basenet_small()
        batch_size = 3
        seq_len = 10

        # Create a batch of state tuples
        token_types = torch.randint(0, 7, (batch_size, seq_len))
        card_uid_indices = torch.randint(0, 4, (batch_size, seq_len))
        status_uid_indices = torch.randint(0, 7, (batch_size, seq_len))
        enemy_intent_indices = torch.randint(0, 3, (batch_size, seq_len))
        # Create encoded numbers with the new format (12 dimensions)
        encoded_numbers = torch.rand((batch_size, seq_len, NUMBER_ENCODING_DIMS))

        state_tuple = (
            token_types,
            card_uid_indices,
            status_uid_indices,
            enemy_intent_indices,
            encoded_numbers,
        )

        with torch.no_grad():
            latent = model(state_tuple)

        assert latent.shape == (batch_size, model.config.latent_dim)
        assert not torch.isnan(latent).any()

    def test_padding_handling(self) -> None:
        """Test that padding is handled correctly."""
        model = observation_basenet_small()
        seq_len = 10

        # Create a state tuple with padding (zeros)
        token_types = torch.zeros(seq_len, dtype=torch.long)
        token_types[0:5] = torch.randint(1, 7, (5,))  # Only first 5 are non-padding

        card_uid_indices = torch.randint(0, 4, (seq_len,))
        status_uid_indices = torch.randint(0, 7, (seq_len,))
        enemy_intent_indices = torch.randint(0, 3, (seq_len,))
        # Create encoded numbers with the new format (12 dimensions)
        encoded_numbers = torch.rand((seq_len, NUMBER_ENCODING_DIMS))

        state_tuple = (
            token_types,
            card_uid_indices,
            status_uid_indices,
            enemy_intent_indices,
            encoded_numbers,
        )

        with torch.no_grad():
            latent = model(state_tuple)

        assert latent.shape == (1, model.config.latent_dim)
        assert not torch.isnan(latent).any()


if __name__ == "__main__":
    pytest.main()

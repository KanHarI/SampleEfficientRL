"""
Observation Basenet Module

This module defines a configurable neural network architecture for processing
game state observations from the Deckbuilder environment and producing a latent
representation that can be used by downstream RL components.
"""

from dataclasses import dataclass
from typing import Tuple, cast

import torch.nn as nn
from torch import Tensor

from SampleEfficientRL.Agents.RL.Networks.ObservationEmbedder import (
    ObservationEmbedder,
    ObservationEmbedderConfig,
)
from SampleEfficientRL.Agents.RL.Networks.ObservationNetwork import (
    ObservationNetwork,
    ObservationNetworkConfig,
)


@dataclass
class ObservationBasenetConfig:
    """Configuration for the observation basenet."""

    # Embedding dimensions
    token_type_embedding_dim: int = 32
    card_embedding_dim: int = 32
    status_embedding_dim: int = 16
    intent_embedding_dim: int = 16

    # Hidden dimensions for the different processing pathways
    hidden_dim: int = 128

    # Output latent dimension
    latent_dim: int = 256

    # Architecture configuration
    num_layers: int = 2
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Number of heads in multi-head attention
    num_heads: int = 4

    # Maximum sequence length
    max_seq_len: int = 128

    @property
    def embedder_config(self) -> ObservationEmbedderConfig:
        """Generate embedder config from the basenet config."""
        return ObservationEmbedderConfig(
            token_type_embedding_dim=self.token_type_embedding_dim,
            card_embedding_dim=self.card_embedding_dim,
            status_embedding_dim=self.status_embedding_dim,
            intent_embedding_dim=self.intent_embedding_dim,
            numerical_hidden_dim=self.hidden_dim // 2,
            max_seq_len=self.max_seq_len,
        )

    def network_config(self, input_dim: int) -> ObservationNetworkConfig:
        """Generate network config from the basenet config."""
        return ObservationNetworkConfig(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_layer_norm=self.use_layer_norm,
            num_heads=self.num_heads,
        )


class ObservationBasenet(nn.Module):
    """
    Neural network that processes game state observations and produces a latent representation.

    This network takes the tensor representation from SingleBattleEnvTensorizer and processes
    it through several embedding and processing layers to produce a fixed-size latent vector.
    """

    def __init__(self, config: ObservationBasenetConfig):
        """
        Initialize the observation basenet.

        Args:
            config: Configuration parameters for the network.
        """
        super().__init__()
        self.config = config

        # Create embedder
        self.embedder = ObservationEmbedder(config.embedder_config)

        # Create network
        network_config = config.network_config(self.embedder.embedding_dim)
        self.network = ObservationNetwork(network_config)

    def forward(
        self,
        state_tuple: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> Tensor:
        """
        Process the game state into a latent representation.

        Args:
            state_tuple: A tuple of tensors from SingleBattleEnvTensorizer:
                - token_types: [batch_size, seq_len] tensor of token types
                - card_uid_indices: [batch_size, seq_len] tensor of card UID indices
                - status_uid_indices: [batch_size, seq_len] tensor of status UID indices
                - enemy_intent_indices: [batch_size, seq_len] tensor of enemy intent indices
                - encoded_numbers: [batch_size, seq_len] tensor of numerical values

        Returns:
            A [batch_size, latent_dim] tensor representing the game state.
        """
        # Get embedded features and padding mask from embedder
        embedded_features, padding_mask = self.embedder(state_tuple)

        # Process embedded features through network
        latent_representation = self.network(embedded_features, padding_mask)

        return cast(Tensor, latent_representation)


def observation_basenet_small() -> ObservationBasenet:
    """Create a small observation basenet for testing."""
    config = ObservationBasenetConfig(
        token_type_embedding_dim=16,
        card_embedding_dim=16,
        status_embedding_dim=8,
        intent_embedding_dim=8,
        hidden_dim=64,
        latent_dim=128,
        num_layers=1,
        dropout=0.1,
        use_layer_norm=True,
        num_heads=2,
        max_seq_len=128,
    )
    return ObservationBasenet(config)


def observation_basenet_medium() -> ObservationBasenet:
    """Create a medium-sized observation basenet for general use."""
    config = ObservationBasenetConfig(
        token_type_embedding_dim=32,
        card_embedding_dim=32,
        status_embedding_dim=16,
        intent_embedding_dim=16,
        hidden_dim=128,
        latent_dim=256,
        num_layers=2,
        dropout=0.1,
        use_layer_norm=True,
        num_heads=4,
        max_seq_len=128,
    )
    return ObservationBasenet(config)


def observation_basenet_large() -> ObservationBasenet:
    """Create a large observation basenet for complex environments."""
    config = ObservationBasenetConfig(
        token_type_embedding_dim=64,
        card_embedding_dim=64,
        status_embedding_dim=32,
        intent_embedding_dim=32,
        hidden_dim=256,
        latent_dim=512,
        num_layers=4,
        dropout=0.1,
        use_layer_norm=True,
        num_heads=8,
        max_seq_len=256,
    )
    return ObservationBasenet(config)

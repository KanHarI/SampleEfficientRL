"""
Observation Embedder Module

This module defines the embedder component for processing raw game state inputs
and converting them into embeddings that can be used by the observation network.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import (
    MAX_ENCODED_NUMBER,
    SUPPORTED_ENEMY_INTENT_TYPES,
    SUPPORTED_CARDS_UIDs,
    SUPPORTED_STATUS_UIDs,
    TokenType,
)


@dataclass
class ObservationEmbedderConfig:
    """Configuration for the observation embedder."""

    # Embedding dimensions
    token_type_embedding_dim: int = 32
    card_embedding_dim: int = 32
    status_embedding_dim: int = 16
    intent_embedding_dim: int = 16

    # Hidden dimensions for the numerical encoder
    numerical_hidden_dim: int = (
        64  # Will be used as half of the hidden_dim in the network
    )

    # Maximum sequence length
    max_seq_len: int = 128


class ObservationEmbedder(nn.Module):
    """
    Neural network component that embeds raw game state data.

    This network takes the tensor representation from SingleBattleEnvTensorizer and
    processes it through several embedding layers to produce embedded features.
    """

    def __init__(self, config: ObservationEmbedderConfig):
        """
        Initialize the observation embedder.

        Args:
            config: Configuration parameters for the embedder.
        """
        super().__init__()
        self.config = config

        # Token type embedding
        self.token_type_embedding = nn.Embedding(
            len(TokenType) + 1, config.token_type_embedding_dim  # +1 for padding
        )

        # Card embedding
        # Add 1 to accommodate padding at index 0
        self.card_embedding = nn.Embedding(
            len(SUPPORTED_CARDS_UIDs) + 1, config.card_embedding_dim  # +1 for padding
        )

        # Status embedding
        self.status_embedding = nn.Embedding(
            len(SUPPORTED_STATUS_UIDs) + 1,  # +1 for padding
            config.status_embedding_dim,
        )

        # Intent embedding
        self.intent_embedding = nn.Embedding(
            len(SUPPORTED_ENEMY_INTENT_TYPES) + 1,  # +1 for padding
            config.intent_embedding_dim,
        )

        # Numerical value encoder
        self.numerical_encoder = nn.Sequential(
            nn.Linear(1, config.numerical_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.numerical_hidden_dim // 2, config.numerical_hidden_dim),
            nn.ReLU(),
        )

    @property
    def embedding_dim(self) -> int:
        """Total dimension of the combined embeddings."""
        return (
            self.config.token_type_embedding_dim
            + self.config.card_embedding_dim
            + self.config.status_embedding_dim
            + self.config.intent_embedding_dim
            + self.config.numerical_hidden_dim
        )

    def forward(
        self,
        state_tuple: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Embed the game state data.

        Args:
            state_tuple: A tuple of tensors from SingleBattleEnvTensorizer:
                - token_types: [batch_size, seq_len] tensor of token types
                - card_uid_indices: [batch_size, seq_len] tensor of card UID indices
                - status_uid_indices: [batch_size, seq_len] tensor of status UID indices
                - enemy_intent_indices: [batch_size, seq_len] tensor of enemy intent indices
                - encoded_numbers: [batch_size, seq_len] tensor of numerical values

        Returns:
            A tuple containing:
                - embedded_features: [batch_size, seq_len, embedding_dim] tensor of embedded features
                - padding_mask: [batch_size, seq_len] boolean tensor indicating padding positions
        """
        (
            token_types,
            card_uid_indices,
            status_uid_indices,
            enemy_intent_indices,
            encoded_numbers,
        ) = state_tuple

        # Get original shape and ensure sequence length doesn't exceed max_seq_len
        seq_len = min(token_types.size(-1), self.config.max_seq_len)

        # Reshape inputs to include batch dimension if needed
        if token_types.dim() == 1:  # If no batch dimension
            token_types = token_types.unsqueeze(0)
            card_uid_indices = card_uid_indices.unsqueeze(0)
            status_uid_indices = status_uid_indices.unsqueeze(0)
            enemy_intent_indices = enemy_intent_indices.unsqueeze(0)
            encoded_numbers = encoded_numbers.unsqueeze(0)

        # Limit sequence length to max_seq_len
        token_types = token_types[:, :seq_len]
        card_uid_indices = card_uid_indices[:, :seq_len]
        status_uid_indices = status_uid_indices[:, :seq_len]
        enemy_intent_indices = enemy_intent_indices[:, :seq_len]
        encoded_numbers = encoded_numbers[:, :seq_len]

        # Create mask for padding positions (where token_types is 0)
        padding_mask = token_types == 0

        # Apply embeddings
        token_embeddings = self.token_type_embedding(token_types)
        card_embeddings = self.card_embedding(card_uid_indices)
        status_embeddings = self.status_embedding(status_uid_indices)
        intent_embeddings = self.intent_embedding(enemy_intent_indices)

        # Process numerical values
        # Normalize to [0, 1] range based on MAX_ENCODED_NUMBER
        normalized_numbers = encoded_numbers.float() / MAX_ENCODED_NUMBER
        normalized_numbers = normalized_numbers.unsqueeze(-1)  # Add feature dimension
        numeric_embeddings = self.numerical_encoder(normalized_numbers)

        # Combine all embeddings
        combined_embeddings = torch.cat(
            [
                token_embeddings,
                card_embeddings,
                status_embeddings,
                intent_embeddings,
                numeric_embeddings,
            ],
            dim=-1,
        )

        return combined_embeddings, padding_mask


def observation_embedder_small() -> ObservationEmbedder:
    """Create a small observation embedder for testing."""
    config = ObservationEmbedderConfig(
        token_type_embedding_dim=16,
        card_embedding_dim=16,
        status_embedding_dim=8,
        intent_embedding_dim=8,
        numerical_hidden_dim=32,
        max_seq_len=128,
    )
    return ObservationEmbedder(config)


def observation_embedder_medium() -> ObservationEmbedder:
    """Create a medium-sized observation embedder for general use."""
    config = ObservationEmbedderConfig(
        token_type_embedding_dim=32,
        card_embedding_dim=32,
        status_embedding_dim=16,
        intent_embedding_dim=16,
        numerical_hidden_dim=64,
        max_seq_len=128,
    )
    return ObservationEmbedder(config)


def observation_embedder_large() -> ObservationEmbedder:
    """Create a large observation embedder for complex environments."""
    config = ObservationEmbedderConfig(
        token_type_embedding_dim=64,
        card_embedding_dim=64,
        status_embedding_dim=32,
        intent_embedding_dim=32,
        numerical_hidden_dim=128,
        max_seq_len=256,
    )
    return ObservationEmbedder(config)

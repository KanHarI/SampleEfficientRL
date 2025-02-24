"""
Observation Basenet Module

This module defines a configurable neural network architecture for processing
game state observations from the Deckbuilder environment and producing a latent
representation that can be used by downstream RL components.
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
            nn.Linear(1, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim),
            nn.ReLU(),
        )

        # Combined embedding dimension
        combined_embedding_dim = (
            config.token_type_embedding_dim
            + config.card_embedding_dim
            + config.status_embedding_dim
            + config.intent_embedding_dim
            + config.hidden_dim  # From numerical encoder
        )

        # Projection layer to unified hidden dimension
        self.projection = nn.Linear(combined_embedding_dim, config.hidden_dim)

        # Layer normalization
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_dim)

        # Transformer or GRU layers for sequence processing
        if config.num_layers > 0:
            # Self-attention layers
            self.self_attention_layers = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=config.hidden_dim,
                        num_heads=config.num_heads,
                        dropout=config.dropout,
                        batch_first=True,
                    )
                    for _ in range(config.num_layers)
                ]
            )

            # Feed-forward networks after attention
            self.feed_forward_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                        nn.ReLU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.hidden_dim * 4, config.hidden_dim),
                        nn.Dropout(config.dropout),
                    )
                    for _ in range(config.num_layers)
                ]
            )

            if config.use_layer_norm:
                self.attention_layer_norms = nn.ModuleList(
                    [nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)]
                )

                self.ffn_layer_norms = nn.ModuleList(
                    [nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)]
                )

        # Final output head
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.latent_dim),
        )

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

        # Project to hidden dimension
        hidden_states = self.projection(combined_embeddings)

        # Apply layer normalization
        if self.config.use_layer_norm:
            hidden_states = self.layer_norm(hidden_states)

        # Process through transformer/self-attention layers
        if self.config.num_layers > 0:
            # Using key_padding_mask is sufficient for handling padding in MultiheadAttention
            # The key_padding_mask will prevent attention to padded positions

            for i in range(self.config.num_layers):
                # Self-attention layer
                attn_output, _ = self.self_attention_layers[i](
                    query=hidden_states,
                    key=hidden_states,
                    value=hidden_states,
                    key_padding_mask=padding_mask,  # This handles the padding correctly
                    need_weights=False,
                )

                # Residual connection and layer norm
                hidden_states = hidden_states + attn_output
                if self.config.use_layer_norm:
                    hidden_states = self.attention_layer_norms[i](hidden_states)

                # Feed-forward network
                ffn_output = self.feed_forward_layers[i](hidden_states)

                # Residual connection and layer norm
                hidden_states = hidden_states + ffn_output
                if self.config.use_layer_norm:
                    hidden_states = self.ffn_layer_norms[i](hidden_states)

        # Create a context vector by averaging the non-padded tokens
        # Mask padded positions
        mask = ~padding_mask  # True for non-padded positions
        mask = mask.float().unsqueeze(-1)

        # Apply mask and compute mean
        context_vector = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(
            min=1
        )

        # Apply output head to get final latent representation
        latent_representation = self.output_head(context_vector)

        return Tensor(latent_representation)


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

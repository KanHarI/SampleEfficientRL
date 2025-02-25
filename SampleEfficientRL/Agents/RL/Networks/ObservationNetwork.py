"""
Observation Network Module

This module defines the network component for processing embedded game state features
and producing a latent representation that can be used by downstream RL components.
"""

from dataclasses import dataclass
from typing import cast

import torch.nn as nn
from torch import Tensor


@dataclass
class ObservationNetworkConfig:
    """Configuration for the observation network."""

    # Input and output dimensions
    input_dim: int  # Will be set based on embedder output
    hidden_dim: int = 128
    latent_dim: int = 256

    # Architecture configuration
    num_layers: int = 2
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Number of heads in multi-head attention
    num_heads: int = 4


class ObservationNetwork(nn.Module):
    """
    Neural network that processes embedded game state features and produces a latent representation.

    This network takes the embedded features from ObservationEmbedder and processes
    them through several layers to produce a fixed-size latent vector.
    """

    def __init__(self, config: ObservationNetworkConfig):
        """
        Initialize the observation network.

        Args:
            config: Configuration parameters for the network.
        """
        super().__init__()
        self.config = config

        # Projection layer to unified hidden dimension
        self.projection = nn.Linear(config.input_dim, config.hidden_dim)

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
        embedded_features: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """
        Process the embedded features into a latent representation.

        Args:
            embedded_features: [batch_size, seq_len, embedding_dim] tensor of embedded features
            padding_mask: [batch_size, seq_len] boolean tensor indicating padding positions

        Returns:
            A [batch_size, latent_dim] tensor representing the game state.
        """
        # Project to hidden dimension
        hidden_states = self.projection(embedded_features)

        # Apply layer normalization
        if self.config.use_layer_norm:
            hidden_states = self.layer_norm(hidden_states)

        # Process through transformer/self-attention layers
        if self.config.num_layers > 0:
            for i in range(self.config.num_layers):
                # Self-attention layer
                attn_output, _ = self.self_attention_layers[i](
                    query=hidden_states,
                    key=hidden_states,
                    value=hidden_states,
                    key_padding_mask=padding_mask,
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

        return cast(Tensor, latent_representation)


def observation_network_small(input_dim: int) -> ObservationNetwork:
    """Create a small observation network for testing."""
    config = ObservationNetworkConfig(
        input_dim=input_dim,
        hidden_dim=64,
        latent_dim=128,
        num_layers=1,
        dropout=0.1,
        use_layer_norm=True,
        num_heads=2,
    )
    return ObservationNetwork(config)


def observation_network_medium(input_dim: int) -> ObservationNetwork:
    """Create a medium-sized observation network for general use."""
    config = ObservationNetworkConfig(
        input_dim=input_dim,
        hidden_dim=128,
        latent_dim=256,
        num_layers=2,
        dropout=0.1,
        use_layer_norm=True,
        num_heads=4,
    )
    return ObservationNetwork(config)


def observation_network_large(input_dim: int) -> ObservationNetwork:
    """Create a large observation network for complex environments."""
    config = ObservationNetworkConfig(
        input_dim=input_dim,
        hidden_dim=256,
        latent_dim=512,
        num_layers=4,
        dropout=0.1,
        use_layer_norm=True,
        num_heads=8,
    )
    return ObservationNetwork(config)

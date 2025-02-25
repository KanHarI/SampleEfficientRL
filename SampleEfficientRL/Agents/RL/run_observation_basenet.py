"""
Demo script for the Observation Basenet implementation.

This script demonstrates loading replay data and processing it through the
observation basenet to generate latent representations. It's intended
for manual testing and demonstrations.
"""

import argparse
import os
from typing import List, cast

import torch

from SampleEfficientRL.Agents.RL.Networks.ObservationBasenet import (
    observation_basenet_large,
    observation_basenet_medium,
    observation_basenet_small,
)
from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import (
    ActionType,
    PlaythroughStep,
)


def load_replay_data(replay_path: str) -> List[PlaythroughStep]:
    """
    Load replay data from a file.

    Args:
        replay_path: Path to the replay file.

    Returns:
        List of PlaythroughStep objects.
    """
    print(f"Loading replay data from {replay_path}...")
    # Add necessary classes to safe globals for loading
    torch.serialization.add_safe_globals([PlaythroughStep, ActionType])
    replay_data = torch.load(replay_path, weights_only=False)
    print(f"Loaded {len(replay_data)} steps.")
    return cast(List[PlaythroughStep], replay_data)


def run_observation_basenet(replay_path: str, model_size: str = "medium") -> None:
    """
    Demonstrate the observation basenet by processing replay data.

    Args:
        replay_path: Path to the replay file.
        model_size: Size of the model to use ("small", "medium", or "large").
    """
    # Load replay data
    replay_data = load_replay_data(replay_path)

    # Create model based on size
    if model_size.lower() == "small":
        model = observation_basenet_small()
    elif model_size.lower() == "large":
        model = observation_basenet_large()
    else:  # Default to medium
        model = observation_basenet_medium()

    print(f"Created {model_size} observation basenet")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Process a sample of steps
    num_samples = min(5, len(replay_data))

    print("\nProcessing sample steps:")
    for i in range(num_samples):
        step = replay_data[i]

        # Get state tensors
        state_tuple = step.state

        # Process through model
        with torch.no_grad():
            latent = model(state_tuple)

        # Print information
        print(f"\nStep {i}:")
        print(f"  Action type: {step.action_type}")
        print(f"  Latent shape: {latent.shape}")
        print(f"  Latent mean: {latent.mean().item():.6f}")
        print(f"  Latent std: {latent.std().item():.6f}")

        # Print a sample of the latent values
        sample_size = min(5, latent.shape[1])
        print(f"  Sample latent values: {latent[0, :sample_size].tolist()}")


def main() -> None:
    """Parse arguments and run the demonstration."""
    parser = argparse.ArgumentParser(description="Demonstrate the observation basenet.")
    parser.add_argument(
        "--replay",
        type=str,
        default="playthrough_data/random_walk_with_actions.pt",
        help="Path to the replay file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Size of the model to use.",
    )

    args = parser.parse_args()

    # Check if replay file exists
    if not os.path.exists(args.replay):
        print(f"Replay file not found: {args.replay}")
        print("Please run the random agent first or provide a valid replay file path.")
        return

    # Run demonstration
    run_observation_basenet(args.replay, args.model)


if __name__ == "__main__":
    main()

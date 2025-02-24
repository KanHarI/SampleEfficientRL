# Sample Efficient RL

This repo is eventually intended to be an RL agent playing Slay The Spire.

Currently, what I have is a CLI of a simple version (Ironclad starter vs Cultist) that I can use to test the framework.

## How to run
* Create a venv:
```bash
python -m venv venv
```
* Activate the venv:
```bash
source venv/bin/activate
```
* Install the package:
```bash
./build_lint_and_test.sh
```
* Run the CLI:
```bash
play_simple
```

## Random Agent and Replay Explorer

### Generating Random Playthroughs
The project includes a random walk agent that makes random decisions and records the gameplay for later analysis:

```bash
# Generate a random playthrough with default settings
random_walk

# Specify a custom output file
random_walk --output playthrough_data/my_custom_playthrough.pt
```

The random agent will:
* Choose random cards to play (from those that it can afford with current energy)
* Sometimes randomly end turns early
* Record all states and actions in a tensor format
* Save the entire playthrough to a binary file

### Replaying Saved Playthroughs
You can view saved playthroughs in a turn-by-turn format using the replay explorer:

```bash
# View a saved playthrough in turn-by-turn format
replay_explorer playthrough_data/random_walk_playthrough.pt

# Save the replay output to a file
replay_explorer playthrough_data/random_walk_playthrough.pt > replay_output.txt
```

The replay explorer displays the entire playthrough chronologically, showing:
* Turn structure (start turn, player actions, enemy phase, etc.)
* Player and enemy stats at key points
* All actions taken during the game

## Reinforcement Learning Components

The project now includes neural network architectures for reinforcement learning:

### Observation Basenet
The observation basenet is a neural network model that encodes game state observations into a fixed-size latent representation. It serves as the foundation for the RL agent, processing the raw game state into a form that can be used by policy and value networks.

Features:
- Configurable embedding dimensions and network size (small, medium, large)
- Transformer-based architecture for processing game state sequences
- Processes all relevant game state information (player/enemy stats, cards, etc.)

Testing:
```bash
# Test the observation basenet with a replay file
python SampleEfficientRL/Agents/RL/run_observation_basenet.py --replay playthrough_data/random_walk_with_actions.pt

# Optional: specify the model size
python SampleEfficientRL/Agents/RL/run_observation_basenet.py --replay playthrough_data/random_walk_with_actions.pt --model [small|medium|large]
```

The model outputs a latent representation of shape [batch_size, latent_dim] where latent_dim is 128, 256, or 512 depending on the model size.

### Dynamics Network
The dynamics network is a neural network that returns policy + estimated value. It receives γ, as a discount factor, and supports multiple values. It will be trained on 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999 (See Agent57 paper for motivation).

It also receives β, the curiosity reward factor. It will be trained on 0.0, 0.5, 1.0, 2.0, 5.0 of normalized reward (See Agent57 paper for motivation).

It predicts curiosity and external reward on different heads (See Agent57 paper for motivation).

### Model Network
The model network is a neural network that takes the latent representation and action as input and outputs the predicted next state.

### Discriminator Network
The discriminator network is a neural network that takes the predicted future latent representation or observed next state as input and outputs whether the predicted future state is correct.

It derives a curiosity reward from the discriminator network. If reality confuses us or the predicted future state is systematically wrong, we are in interesting territory.

### MCTS Searching
The project implements MCTS searching with the model network and the dynamics network.

### Self-Consistency Loss
The project implements a self-consistency loss (See EfficientZero).

## Game Mechanics
* Teach an agent to win with minimal HP loss as Ironclad starter VS a single cultist
* Add support for Exhaust mechanics
* Add support for artifacts
* Give Frozen Eye to the agent 50% of times and let it know it is there, to allow it learn to use deck mechanics
* Give Charon's Ashes to the agent 50% of times and let it know it is there, to allow it learn to use exhaust and "understand artifacts"
* Create a battle with a "real" deck against 3 cultists
* Create a sequence of battles with card rewards chosen in between

## Connecting to Slay The Spire
* Connect to the engine via CommunicationMode
* Connect to lightning_sts ?

---
# Roadmap

## RL
* ✅ Implement a configurable observation basenet taking in the in-battle game state and returning a latent representation
* Create a dynamics network that returns policy + estimated value
    * Will receive γ, as a the discount factor and support multiple values, will be trained on 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999 (See Agent57 paper for motivation)
    * Will receive β, the curiosity reward factor. Will be trained on 0.0, 0.5, 1.0, 2.0, 5.0 of normalized reward  (See Agent57 paper for motivation)
    * Will predict curiosity and external reward on different heads (See Agent57 paper for motivation)
* Create a model network (See EfficientZero, MuZero, etc.):
    * Input: latent representation + action
    * Output: predicted next state
* Create a discriminator network (See EfficientZero; Also - GANs, BYOL, SPR[Self-Predictive Representations]. We are using it rather then a projection head as unlike ATARI and go, we have inherent randomness in the game. *This is the experimental part of this repo*):
    * Input: Predicted future latent representation OR observed next state
    * Output: Is the predicted future state correct?
    * Derive a curiosity reward from the discriminator network - if reality confuses us OR the the predicted future state is systematically wrong, we are in interesting territory
* Implement MCTS searching with the model network and the dynamics network
* Implement a self-consistency loss (See EfficientZero)

## Game Mechanics
* Teach an agent to win with minimal HP loss as Ironclad starter VS a single cultist
* Add support for Exhaust mechanics
* Add support for artifacts
* Give Frozen Eye to the agent 50% of times and let it know it is there, to allow it learn to use deck mechanics
* Give Charon's Ashes to the agent 50% of times and let it know it is there, to allow it learn to use exhaust and "understand artifacts"
* Create a battle with a "real" deck against 3 cultists
* Create a sequence of battles with card rewards chosen in between

## Connecting to Slay The Spire
* Connect to the engine via CommunicationMode
* Connect to lightning_sts ?

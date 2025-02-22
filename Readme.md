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
---
# Roadmap

## RL
* Implement a configurable observation basenet taking in the in-battle game state and returning a latent representation
* Create a dynamics network that returns policy + estimated value
    * Will receive γ, as a the discount factor and support multiple values, will be trained on 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999 (See Agent57 paper for motivation)
    * Will receive β, the curiosity reward factor. Will be trained on 0.0, 0.5, 1.0, 2.0, 5.0 of normalized reward  (See Agent57 paper for motivation)
* Create a model network (See EfficientZero, MuZero, etc.):
    * Input: latent representation + action
    * Output: predicted next state
* Create a discriminator network (See GANs, BYOL, SPR - Self-Predictive Representations. We are using it rather then a projection head as unlike ATARI and go, we have inherent randomness in the game. *This is the experimental part of this repo*):
    * Input: Predicted future latent representation OR observed next state
    * Output: Is the predicted future state correct?

## Game Mechanics
* Teach an agent to win with minimal HP loss as Ironclad starter VS a single cultist
* Add support for Exhaust mechanics
* Add support for artifacts
* Give Frozen Eye to the agent 50% of times and let it know it is there, to allow it learn to use deck mechanics
* Give Charon's Ashes to the agent 50% of times and let it know it is there, to allow it learn to use exhaust and "understand artifacts"
* Create a battle with a "real" deck against 3 cultists
* Create a sequence of battles with card rewards chosen in between

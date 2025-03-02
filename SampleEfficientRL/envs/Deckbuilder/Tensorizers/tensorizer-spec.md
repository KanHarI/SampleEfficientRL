# Tensorizer & De-Tensorizer Specification

## Overview
The tensorizer converts every game step into a binary tensor, capturing essential game state information in a fixed, network-friendly format. The de-tensorizer reverses this process to reconstruct structured game states.

## Recorded Data on Each Step
Each tensor record must include:
- **HP Values:**
  - Player's HP.
  - Each opponent's HP.
- **Statuses:**
  - All statuses applied to the player and each opponent (status ID and corresponding value).
- **Deck Information:**
  - Draw deck.
  - Hand.
  - Discard pile.
- **Metadata:**
  - Turn number.
  - Action type (e.g., PLAY_CARD, END_TURN, NO_OP).
  - Reward (if applicable).

## Number Serialization Format
Every numeric value must be serialized using the following structure:
1. **Scalar Number:** The original value.
2. **Sign Bit:** A bit indicating the sign (0 for positive, 1 for negative).
3. **10 Binary Bits:** A fixed-length array (10 elements) of binary digits (0 or 1) representing the number's magnitude in a simple, digestible way for the network.

*Example:*  
A number `N` is represented as:
[scalar_value, sign_bit, [b1, b2, ..., b10]]
where `[b1, b2, ..., b10]` is the 10-bit binary representation of the absolute value of `N`.

## Key Interfaces

### Tensorizer
- **`tensorize(env) -> tensor_state`**  
  Converts the current game state into a tensor record that includes HP values, statuses, and deck information following the specified number format.

- **`record_play_card(env, card_idx, target_idx, reward) -> None`**  
  Logs a tensorized record for a card play action.

- **`record_end_turn(env, reward) -> None`**  
  Logs a tensorized record for ending a turn.

- **`record_action(state_tensor, action_type, reward, turn_number) -> None`**  
  Records a generic action (e.g., NO_OP) along with the tensorized state.

- **`record_enemy_action(env, enemy_idx, move_type, amount, reward) -> None`**  
  Logs enemy actions with the corresponding tensorized game state.

- **`save_playthrough(filename) -> None`**  
  Serializes and saves the tensorized playthrough data to a binary file (including a header with versioning, tensor size, timestamp, etc.).

- **`get_playthrough_data() -> List[Any]`**  
  Returns the in-memory tensor records for further processing or inspection.

### De-Tensorizer
- **`load_playthrough(filename) -> List[tensor_state]`**  
  Loads and deserializes the binary tensor data.

- **`reconstruct_state(tensor_state) -> Dict`**  
  Reconstructs a tensor record back into a structured, human-readable game state.

- **`replay_playthrough(playthrough_data) -> None`** *(Optional)*  
  Iterates through tensor records to simulate game progression for debugging or analysis.

## Data Format & Storage
- **Tensor Format:**  
  A fixed-size vector (e.g., NumPy array or torch.Tensor) with fields in a consistent order: HP values, statuses, deck information, followed by metadata.
  
- **Binary Serialization:**  
  Records are stored in a binary format with a header (version, tensor size, timestamp) ensuring data integrity and compatibility.

## Error Handling & Extensibility
- **Error Handling:**  
  - Validate tensor dimensions and binary representation lengths.
  - Check header version compatibility.
  - Gracefully handle corrupted or incomplete data with descriptive errors.
  
- **Extensibility:**  
  - Design for adding new game features (e.g., additional statuses or actions) without breaking backward compatibility.
  - Use versioning in the header to support future changes.

## Conclusion
This specification ensures that every game step is recorded with detailed, fixed-format tensors optimized for neural network processing and later reconstruction, enabling efficient training, analysis, and debugging of the RL agent.

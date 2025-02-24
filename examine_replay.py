import torch
from SampleEfficientRL.Envs.Deckbuilder.Tensorizers.SingleBattleEnvTensorizer import PlaythroughStep, ActionType

# Add safe globals for loading
torch.serialization.add_safe_globals([PlaythroughStep, ActionType])

# Load the replay data
data = torch.load('playthrough_data/random_walk_with_actions.pt', weights_only=False)

print(f'Number of steps: {len(data)}')
print(f'First step action type: {data[0].action_type}')
print('\nAction types in sequence:')
for i, step in enumerate(data):
    print(f'{i}: {step.action_type}')

# Look for END_TURN followed by non-NO_OP to find turn boundaries
print('\nPotential turn boundaries:')
for i in range(1, len(data)):
    prev_step = data[i-1]
    step = data[i]
    if prev_step.action_type == ActionType.END_TURN and step.action_type != ActionType.NO_OP:
        print(f'Possible turn boundary at step {i}, action: {step.action_type}') 
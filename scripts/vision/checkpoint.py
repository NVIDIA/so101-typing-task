"""Preserve a trained wrist encoder when extending its goal vocabulary."""

import torch


def expand_goal_vocabulary(actor_state, target_state, *, old_goals=26, new_goals=38):
    """Insert new one-hot goal channels between the old goals and phase features.

    The CNN latent follows the normalized proprioception vector in CNNModel's
    first MLP layer. Existing A-Z, phase, camera, and downstream weights keep
    their exact values; the additional goal channels start with zero weights.
    """
    prefix, suffix = 15, 4  # joint pos/vel/command, then phase and elapsed time
    old_proprio = prefix + old_goals + suffix
    new_proprio = prefix + new_goals + suffix
    if new_goals <= old_goals:
        raise ValueError('goal vocabulary must grow')
    if actor_state['obs_normalizer._mean'].shape != (1, old_proprio):
        raise ValueError('source actor observation contract is not the 26-key vision policy')
    if target_state['obs_normalizer._mean'].shape != (1, new_proprio):
        raise ValueError('target actor observation contract does not match phrase key vocabulary')
    old_weight = actor_state['mlp.0.weight']
    target_weight = target_state['mlp.0.weight']
    if old_weight.shape[0] != target_weight.shape[0] or old_weight.shape[1] - old_proprio != target_weight.shape[1] - new_proprio:
        raise ValueError('CNN latent or MLP width changed; cannot preserve checkpoint')
    if set(actor_state) != set(target_state):
        raise ValueError('actor parameters changed beyond the goal vocabulary')

    adapted = {}
    for name, target in target_state.items():
        source = actor_state[name]
        if name == 'mlp.0.weight':
            value = torch.zeros_like(target)
            value[:, :prefix + old_goals] = source[:, :prefix + old_goals]
            value[:, prefix + new_goals: new_proprio] = source[:, prefix + old_goals: old_proprio]
            value[:, new_proprio:] = source[:, old_proprio:]
        elif name in ('obs_normalizer._mean', 'obs_normalizer._var', 'obs_normalizer._std'):
            value = target.clone()
            value[:, :prefix + old_goals] = source[:, :prefix + old_goals]
            value[:, prefix + new_goals:] = source[:, prefix + old_goals:]
        else:
            if source.shape != target.shape:
                raise ValueError(f'unexpected parameter shape change: {name}')
            value = source.clone()
        adapted[name] = value
    return adapted

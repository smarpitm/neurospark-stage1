import numpy as np

def update_weights(pre_spikes, post_spikes, weights, free_energy, learning_rate=0.01, max_weight=10.0):
    """
    If free energy is high (prediction error), weaken connections that caused surprise.
    If free energy is low, strengthen connections that led to good predictions.
    Simplified Hebbian / STDP update with safety clipping.
    
    :param pre_spikes: Array of spike activities/counts for pre-synaptic neurons of each synapse
    :param post_spikes: Array of spike activities/counts for post-synaptic neurons of each synapse
    :param weights: Current weights of the synapses
    :param free_energy: Free energy of the current transition
    :param learning_rate: Learning rate for updating weights
    :param max_weight: Maximum weight limit for clipping
    :return: Updated weights array (clipped to [0.0, max_weight] for stability)
    """
    # Normalize/clip free energy before learning update (Phase 4 requirement)
    normalized_fe = np.clip(free_energy, 0.0, 2.0)
    raw_delta = learning_rate * (1.0 - normalized_fe)
    
    # Clip delta_w to protect SNN weights from huge prediction errors
    delta_w = np.clip(raw_delta, -0.05, 0.05)
    
    # Copy weights to make updates
    updated_weights = np.copy(weights)
    
    # Vectorized Hebbian: update only if both pre and post spiked
    coactive = (pre_spikes > 0) & (post_spikes > 0)
    updated_weights[coactive] += delta_w
    
    # Clip to keep weights positive and stable
    updated_weights = np.clip(updated_weights, 0.0, max_weight)
    return updated_weights

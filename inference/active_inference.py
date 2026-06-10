from brian2 import ms
from inference.free_energy import compute_expected_free_energy

def select_action(snn, current_state, possible_actions, encoder, decoder, generative_model, beta=15.0):
    """
    Selects the action using a Boltzmann softmax distribution over the negative
    Expected Free Energy (EFE) values combined with policy preference logits.
    
    :param snn: FlyBrainSNN instance
    :param current_state: 11-dimensional observation state vector
    :param possible_actions: list of valid action indices (e.g. [0, 1, 2, 3])
    :param encoder: Encoder instance
    :param decoder: Decoder instance
    :param generative_model: GenerativeModel instance containing preferences
    :param beta: Precision parameter (inverse temperature) for softmax
    :return: Selected action index
    """
    import numpy as np
    from brian2 import ms, Hz, nA
    
    preferred_state = generative_model.get_preferred_state()
    preferred_reward = generative_model.get_preferred_reward()
    
    # Store the SNN network state before we begin "imagining"
    snn.net.store('before_imagination')
    
    # Phase 3: Disable noise during imagination for cleaner signal propagation
    has_noise = hasattr(snn, 'noise') and snn.noise is not None
    if has_noise:
        snn.noise.active = False
        
    efe_list = []
    policy_logits = []
    
    for action in possible_actions:
        # Restore SNN state for same baseline
        snn.net.restore('before_imagination')
        snn.sensory.I_inj = 0 * nA  # Cleaner reset between rollouts
        
        # 1. Encode state and action
        intention = encoder.encode_intention(current_state, action)
        
        # 2. Inject intention
        snn.inject_sensory(intention)
        
        t_start = snn.net.t
        
        # 3. Simulate forward for 50 ms
        snn.run(50 * ms)
        
        t_end = snn.net.t
        
        # 4. Get motor spikes
        motor_spikes = snn.get_motor_spikes_in_window(t_start, t_end)
        
        # 5. Decode predicted next state and reward
        predicted_next, predicted_reward = decoder.decode(motor_spikes)
        
        # 6. Compute Expected Free Energy (EFE)
        efe = compute_expected_free_energy(
            predicted_next, preferred_state, 
            predicted_reward, preferred_reward
        )
        efe_list.append(efe)
        
        # Decode action preference from the resulting motor spikes
        policy_logit = decoder.decode_action(motor_spikes)[action]
        policy_logits.append(policy_logit)
            
    # Restore state and original noise rate
    snn.net.restore('before_imagination')
    if has_noise:
        snn.noise.active = True
    snn.sensory.I_inj = 0 * nA
    
    efe_arr = np.array(efe_list)
    policy_arr = np.array(policy_logits)
    combined_scores = -efe_arr + policy_arr
    
    # Greedy fallback when max(EFE) - min(EFE) < threshold (flat signal)
    threshold = 0.02
    efe_spread = np.max(efe_arr) - np.min(efe_arr)
    
    if efe_spread < threshold:
        # Check if all combined scores are equal to avoid deterministic oscillation
        if np.all(np.abs(combined_scores - combined_scores[0]) < 1e-5):
            selected_action = np.random.choice(possible_actions)
        else:
            selected_index = np.argmax(combined_scores)
            selected_action = possible_actions[selected_index]
        is_greedy = True
    else:
        # Softmax over combined scores
        shifted_scores = combined_scores - np.max(combined_scores)
        exp_scores = np.exp(beta * shifted_scores)
        probs = exp_scores / np.sum(exp_scores)
        selected_action = np.random.choice(possible_actions, p=probs)
        is_greedy = False
        
    # Diagnostics print
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
    efe_dict = {action_names[act]: round(val, 4) for act, val in zip(possible_actions, efe_arr)}
    combined_dict = {action_names[act]: round(val, 4) for act, val in zip(possible_actions, combined_scores)}
    print(f"  [Plan] Actions: {list(efe_dict.keys())} | EFE: {list(efe_dict.values())} | Combined: {list(combined_dict.values())} | Greedy: {is_greedy}")
    
    return selected_action


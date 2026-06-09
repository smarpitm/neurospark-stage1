from brian2 import ms
from inference.free_energy import compute_expected_free_energy

def select_action(snn, current_state, possible_actions, encoder, decoder, generative_model):
    """
    Selects the action that minimizes Expected Free Energy (EFE) by projecting
    each action's outcome using the SNN.
    
    :param snn: FlyBrainSNN instance
    :param current_state: 11-dimensional observation state vector
    :param possible_actions: list of valid action indices (e.g. [0, 1, 2, 3])
    :param encoder: Encoder instance
    :param decoder: Decoder instance
    :param generative_model: GenerativeModel instance containing preferences
    :return: Selected action index
    """
    best_action = None
    min_free_energy = float('inf')
    
    preferred_state = generative_model.get_preferred_state()
    preferred_reward = generative_model.get_preferred_reward()
    
    # Store the SNN network state before we begin "imagining"
    snn.net.store('before_imagination')
    
    for action in possible_actions:
        # Restore the SNN state so that the simulation of this action starts 
        # from the exact same baseline condition as other actions
        snn.net.restore('before_imagination')
        
        # 1. Encode the current state and intended action into stimulus currents
        intention = encoder.encode_intention(current_state, action)
        
        # 2. Inject this intention into the sensory neurons
        snn.inject_sensory(intention)
        
        # Keep track of the simulation time window
        t_start = snn.net.t
        
        # 3. Simulate the network forward for 50 ms to compute the dynamics
        snn.run(50 * ms)
        
        t_end = snn.net.t
        
        # 4. Read the output motor spikes from the simulation
        motor_spikes = snn.get_motor_spikes_in_window(t_start, t_end)
        
        # 5. Decode the spikes into a predicted next state and a predicted reward
        # (The decoder predicts: next state vector, and next step reward)
        predicted_next, predicted_reward = decoder.decode(motor_spikes)
        
        # 6. Compute how surprising this predicted state is (Expected Free Energy)
        efe = compute_expected_free_energy(
            predicted_next, preferred_state, 
            predicted_reward, preferred_reward
        )
        
        # 7. Select the action that minimizes Expected Free Energy
        if efe < min_free_energy:
            min_free_energy = efe
            best_action = action
            
    # Restore state one final time before returning so that the actual state 
    # of the SNN matches what it was before action selection started.
    snn.net.restore('before_imagination')
    
    return best_action

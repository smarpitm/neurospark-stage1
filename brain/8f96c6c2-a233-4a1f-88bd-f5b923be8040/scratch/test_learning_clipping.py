import numpy as np
from brian2 import ms, second, nA
from brain.network import FlyBrainSNN
from bridge.encoder import Encoder
from bridge.decoder import Decoder
from inference.generative_model import GenerativeModel
from inference.active_inference import select_action
from inference.free_energy import compute_free_energy

# Seed
np.random.seed(42)

def run_test_with_clipping(clip_val=0.1):
    snn = FlyBrainSNN(use_noise=True)
    encoder = Encoder()
    decoder = Decoder()
    gen_model = GenerativeModel()
    
    state = np.zeros(11)
    state[:5] = 1.0
    state[9:11] = [0.707, 0.707]
    
    action = 0  # Fixed action for testing
    
    for step in range(5):
        # Actual step transition
        intention = encoder.encode_intention(state, action)
        snn.inject_sensory(intention)
        
        t_start = snn.net.t
        snn.run(50 * ms)
        t_end = snn.net.t
        
        motor_spikes = snn.get_motor_spikes_in_window(t_start, t_end)
        predicted_next, predicted_reward = decoder.decode(motor_spikes)
        
        # In a real environment, state would change, but let's assume it remains same for simplicity
        next_state = state
        reward = -0.01
        
        fe = compute_free_energy(predicted_next, next_state, predicted_reward, reward)
        
        # Calculate delta_w with clipping
        raw_delta = 0.01 * (1.0 - fe)
        delta_w = np.clip(raw_delta, -clip_val, clip_val)
        
        # Let's see what delta_w is
        print(f"Step {step+1} | Motor Spikes: {np.sum(motor_spikes):3d} | FE: {fe:6.2f} | Raw delta_w: {raw_delta:7.4f} | Clipped delta_w: {delta_w:7.4f}")
        
        # Update weights (we implement the update manually here to test the math)
        sensory_spikes = snn.get_spikes_in_window(snn.spike_mon_sensory, 100, t_start, t_end)
        recurrent_spikes = snn.get_spikes_in_window(snn.spike_mon_recurrent, 800, t_start, t_end)
        
        # Update S_rec as a proxy
        pre_rec = recurrent_spikes[snn.S_rec.i]
        post_rec = recurrent_spikes[snn.S_rec.j]
        coactive = (pre_rec > 0) & (post_rec > 0)
        
        snn.S_rec.w[coactive] = np.clip(snn.S_rec.w[coactive] + delta_w, 0.0, 10.0)
        
        # Update decoder
        decoder.update(motor_spikes, next_state, reward, learning_rate=0.01)

print("--- Testing Learning with Clipped delta_w = 0.01 ---")
run_test_with_clipping(clip_val=0.01)

print("\n--- Testing Learning with Clipped delta_w = 0.05 ---")
run_test_with_clipping(clip_val=0.05)

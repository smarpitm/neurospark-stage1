import numpy as np
from brian2 import ms, second, nA
from brain.network import FlyBrainSNN
from bridge.encoder import Encoder
from bridge.decoder import Decoder
from inference.generative_model import GenerativeModel
from inference.free_energy import compute_free_energy

# Seed
np.random.seed(42)

def run_binary_test():
    snn = FlyBrainSNN(use_noise=True)
    encoder = Encoder()
    
    # Let's customize a Decoder that uses binary spikes
    class BinaryDecoder(Decoder):
        def decode(self, spike_counts):
            binary_spikes = (spike_counts > 0).astype(float)
            predicted_state = np.dot(self.W_out, binary_spikes)
            predicted_reward = float(np.dot(self.W_reward, binary_spikes))
            return predicted_state, predicted_reward
            
        def decode_action(self, spike_counts):
            binary_spikes = (spike_counts > 0).astype(float)
            return np.dot(self.W_action, binary_spikes)
            
        def update(self, spike_counts, actual_state, actual_reward, learning_rate=0.005):
            binary_spikes = (spike_counts > 0).astype(float)
            pred_state, pred_reward = self.decode(spike_counts)
            
            error_state = actual_state - pred_state
            self.W_out += learning_rate * np.outer(error_state, binary_spikes)
            
            error_reward = actual_reward - pred_reward
            self.W_reward += learning_rate * error_reward * binary_spikes

    decoder = BinaryDecoder(motor_neurons=100, output_dim=11)
    
    state = np.zeros(11)
    state[:5] = 1.0
    state[9:11] = [0.707, 0.707]
    
    action = 0  # Fixed action for testing
    
    for step in range(10):
        # Actual step transition
        intention = encoder.encode_intention(state, action)
        snn.inject_sensory(intention)
        
        t_start = snn.net.t
        snn.run(50 * ms)
        t_end = snn.net.t
        
        motor_spikes = snn.get_motor_spikes_in_window(t_start, t_end)
        predicted_next, predicted_reward = decoder.decode(motor_spikes)
        
        # Target next state
        next_state = state
        reward = -0.01
        
        fe = compute_free_energy(predicted_next, next_state, predicted_reward, reward)
        
        # Calculate delta_w with clipping for SNN weights
        # Using a safe learning rate and clipping to preserve network activity
        raw_delta = 0.001 * (1.0 - fe)
        delta_w = np.clip(raw_delta, -0.01, 0.01)
        
        print(f"Step {step+1:02d} | Motor Spikes: {np.sum(motor_spikes):3d} | FE: {fe:7.2f} | delta_w: {delta_w:7.4f}")
        
        # Update SNN S_rec weights
        sensory_spikes = snn.get_spikes_in_window(snn.spike_mon_sensory, 100, t_start, t_end)
        recurrent_spikes = snn.get_spikes_in_window(snn.spike_mon_recurrent, 800, t_start, t_end)
        
        pre_rec = recurrent_spikes[snn.S_rec.i]
        post_rec = recurrent_spikes[snn.S_rec.j]
        coactive = (pre_rec > 0) & (post_rec > 0)
        snn.S_rec.w[coactive] = np.clip(snn.S_rec.w[coactive] + delta_w, 0.0, 10.0)
        
        # Update decoder weights
        decoder.update(motor_spikes, next_state, reward, learning_rate=0.005)

run_binary_test()

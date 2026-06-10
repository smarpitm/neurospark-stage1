import numpy as np
from brian2 import ms, second, nA
from brain.network import FlyBrainSNN
from bridge.encoder import Encoder

# Seed
np.random.seed(42)

snn = FlyBrainSNN(use_noise=True)
encoder = Encoder()

state = np.zeros(11)
state[:5] = 1.0  # Walls
state[9:11] = [0.707, 0.707]  # Goal direction

# Create intention vector
intention_vector = np.zeros(15)
intention_vector[:11] = state
intention_vector[11 + 0] = 1.0  # Action UP

print("--- Testing Multi-step Spiking Activity ---")
for scale in [1.0, 5.0, 15.0]:
    print(f"\n--- Testing with scale = {scale} ---")
    snn = FlyBrainSNN(use_noise=True)
    currents = np.dot(encoder.W_intention, intention_vector) * scale
    
    for step in range(5):
        snn.inject_sensory(currents)
        t_start = snn.net.t
        snn.run(50 * ms)
        t_end = snn.net.t
        
        sens_spikes = snn.get_spikes_in_window(snn.spike_mon_sensory, 100, t_start, t_end)
        rec_spikes = snn.get_spikes_in_window(snn.spike_mon_recurrent, 800, t_start, t_end)
        mot_spikes = snn.get_spikes_in_window(snn.spike_mon_motor, 100, t_start, t_end)
        
        print(f"Step {step+1} | Sensory: {np.sum(sens_spikes):4d} | Recurrent: {np.sum(rec_spikes):4d} | Motor: {np.sum(mot_spikes):3d}")

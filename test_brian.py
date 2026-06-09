from brian2 import *
import matplotlib.pyplot as plt
import os

# Set codegen target to numpy to avoid C++ compilation warnings on machines without build tools
prefs.codegen.target = 'numpy'

# Simple 10-neuron LIF simulation
start_scope()

# Parameters
tau = 20*ms
v_rest = -70*mV
v_threshold = -50*mV
v_reset = -80*mV
R = 10*Mohm
I = 2.5*nA  # Current to drive spiking

# Equations
eqs = '''
dv/dt = (v_rest - v + R*I)/tau : volt (unless refractory)
'''

# Population
neurons = NeuronGroup(10, eqs, threshold='v > v_threshold', reset='v = v_reset', refractory=5*ms, method='exact')
# Initialize membrane potential randomly between v_rest and v_threshold
neurons.v = 'v_rest + rand()*(v_threshold - v_rest)'

# Monitor spikes and membrane potentials
state_mon = StateMonitor(neurons, 'v', record=True)
spike_mon = SpikeMonitor(neurons)

# Run for 100 ms
run(100*ms)

print(f"Brian2 simulation finished.")
print(f"Total spikes recorded: {spike_mon.num_spikes}")
spike_trains = spike_mon.spike_trains()
for i in range(10):
    spike_count = len(spike_trains[i])
    print(f"Neuron {i} spiked {spike_count} times")

# Save a plot to verify output
plt.figure(figsize=(10, 5))
for i in range(min(5, len(neurons))):
    plt.plot(state_mon.t/ms, state_mon.v[i]/mV, label=f'Neuron {i}')
plt.axhline(v_threshold/mV, color='r', linestyle='--', label='Threshold')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Brian2 10-Neuron LIF Simulation Verification')
plt.legend()
plt.grid(True)
plt.savefig('brian2_test_potentials.png')
print("Saved membrane potential plot to 'brian2_test_potentials.png'.")


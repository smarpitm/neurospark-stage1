from brian2 import *
import numpy as np

# Set codegen target
prefs.codegen.target = 'numpy'

# Define network
G = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1', threshold='v > 1', reset='v = 0')
noise = PoissonInput(G, 'v', N=10, rate=100*Hz, weight=0.5)
spike_mon = SpikeMonitor(G)
net = Network(G, noise, spike_mon)

# Run step 1
print("Step 1...")
net.run(10*ms)
print(f"Spike monitor has: {len(spike_mon.t)} spikes")

# Reset monitor dynamically
print("Resetting monitor...")
net.remove(spike_mon)
spike_mon = SpikeMonitor(G)
net.add(spike_mon)

# Run step 2
print("Step 2...")
net.run(10*ms)
print(f"New spike monitor has: {len(spike_mon.t)} spikes")

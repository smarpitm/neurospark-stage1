from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import os

from brain.lif_params import tau, v_rest, v_threshold, v_reset, refractory_period, C, R
from brain.connectome_loader import load_or_generate_connectome

# Set codegen target to numpy to prevent compilation warnings
prefs.codegen.target = 'numpy'

class FlyBrainSNN:
    def __init__(self, use_noise=True):
        """
        Initializes the 1,000-neuron SNN representing the fly brain.
        - 100 Sensory neurons (input)
        - 800 Recurrent neurons (brain)
        - 100 Motor neurons (output)
        """
        start_scope()
        
        # 1. Define LIF equations
        # I_inj will be used to inject sensory stimulus current or test currents
        self.eqs = '''
        dv/dt = (v_rest - v + R * I_inj) / tau : volt (unless refractory)
        I_inj : amp
        '''
        
        # 2. Create neuron groups
        self.sensory = NeuronGroup(100, self.eqs, threshold='v > v_threshold', reset='v = v_reset', refractory=refractory_period, method='exact')
        self.recurrent = NeuronGroup(800, self.eqs, threshold='v > v_threshold', reset='v = v_reset', refractory=refractory_period, method='exact')
        self.motor = NeuronGroup(100, self.eqs, threshold='v > v_threshold', reset='v = v_reset', refractory=refractory_period, method='exact')
        
        # Initialize membrane potentials randomly between v_rest and v_threshold to start from an active state
        self.sensory.v = 'v_rest + rand() * (v_threshold - v_rest)'
        self.recurrent.v = 'v_rest + rand() * (v_threshold - v_rest)'
        self.motor.v = 'v_rest + rand() * (v_threshold - v_rest)'
        
        self.sensory.I_inj = 0 * nA
        self.recurrent.I_inj = 0 * nA
        self.motor.I_inj = 0 * nA
        
        # 3. Connect populations
        # Sensory -> Recurrent (8% connection probability, random weight)
        self.S_in = Synapses(self.sensory, self.recurrent, model='w : 1', on_pre='v_post += w * mV')
        self.S_in.connect(p=0.08)
        self.S_in.w = '0.5 + rand() * 1.5'  # 0.5 to 2.0 mV EPSP
        
        # Recurrent -> Recurrent (Connected via the small-world fly connectome)
        sources, targets, weights = load_or_generate_connectome()
        self.S_rec = Synapses(self.recurrent, self.recurrent, model='w : 1', on_pre='v_post += w * mV')
        self.S_rec.connect(i=sources, j=targets)
        self.S_rec.w = weights  # Synaptic conductances scale defined in connectome loader
        
        # Recurrent -> Motor (8% connection probability, random weight)
        self.S_out = Synapses(self.recurrent, self.motor, model='w : 1', on_pre='v_post += w * mV')
        self.S_out.connect(p=0.08)
        self.S_out.w = '0.5 + rand() * 1.5'  # 0.5 to 2.0 mV EPSP
        
        # Store initial weights and set maximum weight limits to prevent saturation
        self.S_in_initial = np.array(self.S_in.w)
        self.S_rec_initial = np.array(self.S_rec.w)
        self.S_out_initial = np.array(self.S_out.w)
        
        self.max_in_w = self.S_in_initial * 1.5
        self.max_rec_w = self.S_rec_initial * 1.02
        self.max_out_w = self.S_out_initial * 1.5
        
        # 4. Set up Monitors
        self.spike_mon_sensory = SpikeMonitor(self.sensory)
        self.spike_mon_recurrent = SpikeMonitor(self.recurrent)
        self.spike_mon_motor = SpikeMonitor(self.motor)
        
        self.state_mon_rec = StateMonitor(self.recurrent, 'v', record=[0, 1, 2])
        
        # 5. Poisson Background Noise (To trigger spontaneous background firing)
        if use_noise:
            # Connect background Poisson inputs to recurrent neurons
            # Lowered rate and weight (Phase 3) to prevent saturation during actual steps
            self.noise = PoissonInput(self.recurrent, 'v', N=100, rate=2*Hz, weight=1.0*mV)
        else:
            self.noise = None
            
        # 6. Initialize Network
        self.net = Network(
            self.sensory, self.recurrent, self.motor,
            self.S_in, self.S_rec, self.S_out,
            self.spike_mon_sensory, self.spike_mon_recurrent, self.spike_mon_motor,
            self.state_mon_rec
        )
        if self.noise:
            self.net.add(self.noise)
            
    def run(self, duration):
        """
        Runs the simulation for a specific duration.
        """
        self.net.run(duration)
        
    def inject_sensory(self, currents):
        """
        Injects sensory currents into the sensory population.
        :param currents: 100-dimensional numpy array of current values (in nA or with Brian2 units)
        """
        from brian2 import Quantity
        if isinstance(currents, Quantity):
            self.sensory.I_inj = currents
        else:
            self.sensory.I_inj = currents * nA
        
    def get_spikes_in_window(self, monitor, num_neurons, t_start, t_end):
        """
        Calculates the spike counts of a given population within the given time window.
        """
        times = monitor.t
        indices = monitor.i
        
        # Mask spikes within the time window
        mask = (times >= t_start) & (times <= t_end)
        recent_indices = indices[mask]
        
        # Count occurrences for each neuron
        counts = np.bincount(recent_indices, minlength=num_neurons)
        return counts

    def get_motor_spikes_in_window(self, t_start, t_end):
        """
        Calculates the spike counts of the 100 motor neurons within the given time window.
        """
        return self.get_spikes_in_window(self.spike_mon_motor, 100, t_start, t_end)

    def update_weights(self, free_energy, t_start, t_end, learning_rate=0.01):
        """
        Updates synaptic weights in the network using the FEP-modulated Hebbian learning rule.
        """
        from bridge.learning_rules import update_weights
        
        # Get spike counts for each population in the last step
        sensory_spikes = self.get_spikes_in_window(self.spike_mon_sensory, 100, t_start, t_end)
        recurrent_spikes = self.get_spikes_in_window(self.spike_mon_recurrent, 800, t_start, t_end)
        motor_spikes = self.get_spikes_in_window(self.spike_mon_motor, 100, t_start, t_end)
        
        print(f"  [Learning] Spikes - Sensory: {np.sum(sensory_spikes)} | Recurrent: {np.sum(recurrent_spikes)} | Motor: {np.sum(motor_spikes)}")
        print(f"  [Learning] S_in.w mean: {np.mean(self.S_in.w):.4f} | S_rec.w mean: {np.mean(self.S_rec.w):.4f} | S_out.w mean: {np.mean(self.S_out.w):.4f}")
        
        # 1. Update Sensory -> Recurrent weights
        pre_in = sensory_spikes[self.S_in.i]
        post_in = recurrent_spikes[self.S_in.j]
        self.S_in.w = update_weights(pre_in, post_in, self.S_in.w, free_energy, learning_rate, max_weight=self.max_in_w)
        
        # 2. Update Recurrent -> Recurrent weights
        # pre_rec = recurrent_spikes[self.S_rec.i]
        # post_rec = recurrent_spikes[self.S_rec.j]
        # self.S_rec.w = update_weights(pre_rec, post_rec, self.S_rec.w, free_energy, learning_rate, max_weight=self.max_rec_w)
        
        # 3. Update Recurrent -> Motor weights
        pre_out = recurrent_spikes[self.S_out.i]
        post_out = motor_spikes[self.S_out.j]
        self.S_out.w = update_weights(pre_out, post_out, self.S_out.w, free_energy, learning_rate, max_weight=self.max_out_w)
        print(f"  [Learning] Updated means - S_in.w: {np.mean(self.S_in.w):.4f} | S_rec.w: {np.mean(self.S_rec.w):.4f} | S_out.w: {np.mean(self.S_out.w):.4f}")

    def reset_monitors(self):
        """
        Recreates and adds monitors to clear accumulated spike/state histories.
        This speeds up simulations significantly by preventing memory and search overhead growth.
        """
        # 1. Remove old monitors from the network
        self.net.remove(self.spike_mon_sensory)
        self.net.remove(self.spike_mon_recurrent)
        self.net.remove(self.spike_mon_motor)
        self.net.remove(self.state_mon_rec)
        
        # 2. Re-create monitors
        self.spike_mon_sensory = SpikeMonitor(self.sensory)
        self.spike_mon_recurrent = SpikeMonitor(self.recurrent)
        self.spike_mon_motor = SpikeMonitor(self.motor)
        self.state_mon_rec = StateMonitor(self.recurrent, 'v', record=[0, 1, 2])
        
        # 3. Add them back to the network
        self.net.add(self.spike_mon_sensory)
        self.net.add(self.spike_mon_recurrent)
        self.net.add(self.spike_mon_motor)
        self.net.add(self.state_mon_rec)


def plot_spontaneous_activity(brain, duration=200*ms):
    """
    Simulates spontaneous activity and plots a spike raster of all populations.
    """
    print(f"Running spontaneous activity simulation for {duration}...")
    brain.run(duration)
    
    # Extract spike data
    t_sens = brain.spike_mon_sensory.t / ms
    i_sens = brain.spike_mon_sensory.i
    
    t_rec = brain.spike_mon_recurrent.t / ms
    i_rec = brain.spike_mon_recurrent.i + 100  # Offset by sensory population size
    
    t_mot = brain.spike_mon_motor.t / ms
    i_mot = brain.spike_mon_motor.i + 900  # Offset by sensory + recurrent population size
    
    plt.figure(figsize=(12, 7))
    
    # Plot spike trains
    plt.scatter(t_sens, i_sens, s=2, color='#38bdf8', label='Sensory (0-99)', alpha=0.6)
    plt.scatter(t_rec, i_rec, s=1, color='#c084fc', label='Recurrent (100-899)', alpha=0.4)
    plt.scatter(t_mot, i_mot, s=2, color='#10b981', label='Motor (900-999)', alpha=0.6)
    
    plt.axhline(100, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(900, color='gray', linestyle='--', alpha=0.5)
    
    plt.title("FlyBrain SNN: Spontaneous Spiking Activity (1,000 Neurons)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Neuron Index", fontsize=12)
    plt.legend(loc='upper right', frameon=True, facecolor='#1e293b', edgecolor='none', labelcolor='white')
    
    # Aesthetic dark-theme adjustments
    ax = plt.gca()
    ax.set_facecolor('#0f172a')
    plt.gcf().patch.set_facecolor('#0f172a')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xlim(0, duration / ms)
    plt.ylim(0, 1000)
    plt.grid(True, color='#334155', linestyle=':', alpha=0.5)
    
    output_path = "snn_spontaneous_raster.png"
    plt.savefig(output_path, dpi=300, facecolor=plt.gcf().get_facecolor(), bbox_inches='tight')
    print(f"Spontaneous spike raster saved successfully as '{output_path}'.")
    
    # Output spike statistics
    print("\n--- Simulation Stats ---")
    print(f"Sensory spikes: {len(t_sens)} (Mean rate: {len(t_sens)/100/(duration/second):.2f} Hz)")
    print(f"Recurrent spikes: {len(t_rec)} (Mean rate: {len(t_rec)/800/(duration/second):.2f} Hz)")
    print(f"Motor spikes: {len(t_mot)} (Mean rate: {len(t_mot)/100/(duration/second):.2f} Hz)")

if __name__ == "__main__":
    # Instantiate brain with background noise enabled
    brain = FlyBrainSNN(use_noise=True)
    plot_spontaneous_activity(brain, duration=200*ms)

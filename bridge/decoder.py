import numpy as np

class Decoder:
    def __init__(self, motor_neurons=100, output_dim=11):
        """
        Initializes the decoder with random weights.
        :param motor_neurons: Number of motor neurons (default 100)
        :param output_dim: Observation output dimension (default 11)
        """
        self.motor_neurons = motor_neurons
        self.output_dim = output_dim
        
        # Initialize weights with smaller standard deviation for stability
        self.W_out = np.random.randn(output_dim, motor_neurons) * 0.05
        self.W_reward = np.random.randn(motor_neurons) * 0.05
        self.W_action = np.random.randn(4, motor_neurons) * 0.05

    def decode(self, spike_counts):
        """
        Spike counts over 50ms -> predicted next state and predicted reward
        Uses normalized rate coding for numerical stability and discriminability.
        :param spike_counts: 100-dimensional motor neuron spike counts
        :return: (predicted_next_state, predicted_reward)
        """
        normalized_spikes = spike_counts / (np.max(spike_counts) + 1e-8)
        predicted_state = np.dot(self.W_out, normalized_spikes)
        predicted_reward = float(np.dot(self.W_reward, normalized_spikes))
        return predicted_state, predicted_reward

    def decode_action(self, spike_counts):
        """
        Spike counts -> action preference vector
        Uses normalized rate coding for numerical stability and discriminability.
        :param spike_counts: 100-dimensional motor neuron spike counts
        :return: Action preference vector (4-dimensional)
        """
        normalized_spikes = spike_counts / (np.max(spike_counts) + 1e-8)
        return np.dot(self.W_action, normalized_spikes)

    def update(self, spike_counts, actual_state, actual_reward, learning_rate=0.005):
        """
        Updates decoder prediction weights (W_out and W_reward) using local prediction error delta rules.
        Uses normalized rate coding for numerical stability and discriminability.
        :param spike_counts: 100-dimensional motor neuron spike counts
        :param actual_state: 11-dimensional actual next state
        :param actual_reward: Actual reward received
        :param learning_rate: Learning rate for updating weights
        """
        normalized_spikes = spike_counts / (np.max(spike_counts) + 1e-8)
        pred_state, pred_reward = self.decode(spike_counts)
        
        # Delta rule for state prediction weights W_out:
        error_state = actual_state - pred_state
        self.W_out += learning_rate * np.outer(error_state, normalized_spikes)
        
        # Delta rule for reward prediction weights W_reward:
        error_reward = actual_reward - pred_reward
        self.W_reward += learning_rate * error_reward * normalized_spikes

    def update_action_weights(self, motor_spikes, action, free_energy, learning_rate=0.01):
        """
        Updates policy weights W_action using Free Energy modulated Hebbian learning.
        Uses normalized rate coding for numerical stability and discriminability.
        :param motor_spikes: 100-dimensional motor neuron spike counts
        :param action: Selected action index
        :param free_energy: Free energy of the current transition
        :param learning_rate: Learning rate for updating weights
        """
        # Normalize/clip free energy before learning update (Phase 4 requirement)
        normalized_fe = np.clip(free_energy, 0.0, 2.0)
        
        # If free energy is low (prediction error is low), we strengthen the pathway
        delta_w = learning_rate * (1.0 - normalized_fe)
        
        # Clip the delta_w to a safe range to prevent policy explosion/collapse
        delta_w = np.clip(delta_w, -0.05, 0.05)
        
        normalized_spikes = motor_spikes / (np.max(motor_spikes) + 1e-8)
        self.W_action[action] += delta_w * normalized_spikes
        
        # Clip action weights to prevent divergence
        self.W_action = np.clip(self.W_action, -10.0, 10.0)

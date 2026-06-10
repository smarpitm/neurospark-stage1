import numpy as np

INPUT_DIM = 11
NUM_ACTIONS = 4


class Decoder:
    def __init__(self, motor_neurons=100, output_dim=11):
        self.motor_neurons = motor_neurons
        self.output_dim = output_dim
        self.intention_dim = INPUT_DIM + NUM_ACTIONS + (INPUT_DIM * NUM_ACTIONS)

        self.W_out = np.random.randn(output_dim, motor_neurons) * 0.05
        self.W_reward = np.random.randn(motor_neurons) * 0.05
        self.W_action = np.random.randn(NUM_ACTIONS, motor_neurons) * 0.05

        # Explicit forward model: (state, action) -> next state / reward (used for planning)
        self.W_forward = np.random.randn(output_dim, self.intention_dim) * 0.1
        self.W_reward_forward = np.random.randn(self.intention_dim) * 0.05

    def _intention_vector(self, state, action):
        intention = np.zeros(self.intention_dim, dtype=np.float32)
        intention[:INPUT_DIM] = state
        intention[INPUT_DIM + action] = 1.0
        
        cross_idx = INPUT_DIM + NUM_ACTIONS + action * INPUT_DIM
        intention[cross_idx : cross_idx + INPUT_DIM] = state
        return intention

    def _normalize_spikes(self, spike_counts):
        """Rate-code spikes; uniform prior when motor layer is silent."""
        if np.sum(spike_counts) == 0:
            return np.ones(self.motor_neurons, dtype=np.float32) / self.motor_neurons
        return spike_counts.astype(np.float32) / (np.max(spike_counts) + 1e-8)

    def predict_forward(self, state, action):
        """Counterfactual next-state prediction for active inference planning."""
        return np.dot(self.W_forward, self._intention_vector(state, action))

    def predict_reward_forward(self, state, action):
        """Counterfactual reward prediction for active inference planning."""
        return float(np.dot(self.W_reward_forward, self._intention_vector(state, action)))

    def update_forward(self, state, action, actual_next_state, actual_reward, learning_rate=0.01):
        """Train the forward model on observed transitions."""
        x = self._intention_vector(state, action)
        pred_state = self.predict_forward(state, action)
        
        l2_lambda = 0.001
        self.W_forward += learning_rate * (np.outer(actual_next_state - pred_state, x) - l2_lambda * self.W_forward)
        
        pred_reward = self.predict_reward_forward(state, action)
        self.W_reward_forward += learning_rate * ((actual_reward - pred_reward) * x - l2_lambda * self.W_reward_forward)

    def decode(self, spike_counts, current_state=None):
        """
        Decode motor spike counts into predicted next state and reward.

        Zero-spike fallback: if all motors are silent return current_state (or zeros)
        and a mild negative reward signal (-0.5) so the learning signal stays meaningful
        instead of collapsing to whatever the random weights × uniform-prior happen to give.
        """
        if np.max(spike_counts) == 0:
            fallback_state = np.array(current_state, dtype=np.float32) if current_state is not None \
                             else np.zeros(self.output_dim, dtype=np.float32)
            return fallback_state, -0.5

        normalized_spikes = self._normalize_spikes(spike_counts)
        predicted_state = np.dot(self.W_out, normalized_spikes)
        predicted_reward = float(np.dot(self.W_reward, normalized_spikes))
        return predicted_state, predicted_reward

    def decode_action(self, spike_counts):
        """
        Decode motor spikes into per-action logits.

        Zero-spike fallback: return uniform logits so all actions are equally likely
        and downstream softmax / argmax doesn't amplify noise from a silent motor layer.
        """
        if np.max(spike_counts) == 0:
            return np.zeros(NUM_ACTIONS, dtype=np.float32)

        normalized_spikes = self._normalize_spikes(spike_counts)
        return np.dot(self.W_action, normalized_spikes)

    def predict_for_planning(self, state, action, motor_spikes, motor_blend=0.35):
        """
        Blend SNN motor decode with the trained forward model.
        Forward model dominates when motor output is weak or silent.
        Zero-spike case falls directly through to the forward model.
        """
        fwd_state = self.predict_forward(state, action)
        fwd_reward = self.predict_reward_forward(state, action)

        motor_total = float(np.sum(motor_spikes))
        if motor_total < 5.0:
            return fwd_state, fwd_reward

        snn_state, snn_reward = self.decode(motor_spikes, current_state=state)
        alpha = motor_blend * min(1.0, motor_total / 50.0)
        predicted_state = alpha * snn_state + (1.0 - alpha) * fwd_state
        predicted_reward = alpha * snn_reward + (1.0 - alpha) * fwd_reward
        return predicted_state, predicted_reward

    def update(self, spike_counts, actual_state, actual_reward, learning_rate=0.005):
        normalized_spikes = self._normalize_spikes(spike_counts)
        pred_state, pred_reward = self.decode(spike_counts, current_state=actual_state)

        error_state = actual_state - pred_state
        self.W_out += learning_rate * np.outer(error_state, normalized_spikes)

        error_reward = actual_reward - pred_reward
        self.W_reward += learning_rate * error_reward * normalized_spikes

    def update_action_weights(self, motor_spikes, action, free_energy, learning_rate=0.01):
        normalized_fe = np.clip(free_energy, 0.0, 2.0)
        delta_w = learning_rate * (1.0 - normalized_fe)
        delta_w = np.clip(delta_w, -0.05, 0.05)

        normalized_spikes = self._normalize_spikes(motor_spikes)
        self.W_action[action] += delta_w * normalized_spikes
        self.W_action = np.clip(self.W_action, -10.0, 10.0)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path):
        """Persist all decoder weight matrices to a .npz file."""
        np.savez(
            path,
            W_out=self.W_out,
            W_reward=self.W_reward,
            W_action=self.W_action,
            W_forward=self.W_forward,
            W_reward_forward=self.W_reward_forward,
        )

    def load(self, path):
        """
        Restore decoder weights from a .npz checkpoint.
        Shape mismatches are silently ignored so that a checkpoint from a
        different architecture never crashes a new run.
        """
        data = np.load(path)
        for attr, key in [
            ('W_out',            'W_out'),
            ('W_reward',         'W_reward'),
            ('W_action',         'W_action'),
            ('W_forward',        'W_forward'),
            ('W_reward_forward', 'W_reward_forward'),
        ]:
            if key in data and data[key].shape == getattr(self, attr).shape:
                setattr(self, attr, data[key])

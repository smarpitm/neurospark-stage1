import unittest
import numpy as np
from brian2 import ms, second

from inference.free_energy import compute_free_energy, compute_expected_free_energy
from inference.generative_model import GenerativeModel
from inference.active_inference import select_action

class MockNetwork:
    def __init__(self):
        self.t = 0 * second
        self.stored_states = {}

    def store(self, name):
        self.stored_states[name] = self.t

    def restore(self, name):
        self.t = self.stored_states.get(name, 0 * second)

class MockSNN:
    def __init__(self):
        self.net = MockNetwork()

    def inject_sensory(self, currents):
        pass

    def run(self, duration):
        self.net.t += duration

    def get_motor_spikes_in_window(self, t_start, t_end):
        return np.ones(100, dtype=np.int32)

class MockEncoder:
    def encode_intention(self, state, action):
        return np.zeros(100, dtype=np.float32)

class MockDecoder:
    def __init__(self, target_action):
        self.target_action = target_action

    def decode(self, spikes):
        # Return a predicted state based on the current context or action
        # In a real system, the SNN dynamics dictate the output spikes.
        # Since this is a mock, we will check which action is being evaluated
        # during the test and return states accordingly.
        pass

class TestActiveInference(unittest.TestCase):
    def test_free_energy_calculations(self):
        # 1. Test Variational Free Energy (VFE)
        pred_state = np.array([0.5, 0.5, 0.0, 0.0])
        act_state = np.array([0.0, 0.5, 0.0, 0.0])
        pred_rew = 0.5
        act_rew = 1.0
        
        # State MSE = (0.5-0.0)**2 = 0.25
        # Reward MSE = (0.5-1.0)**2 = 0.25
        # VFE = 0.5
        vfe = compute_free_energy(pred_state, act_state, pred_rew, act_rew)
        self.assertAlmostEqual(vfe, 0.5)

    def test_action_selection(self):
        snn = MockSNN()
        encoder = MockEncoder()
        
        # We want the agent to prefer the goal state (where goal direction is 0.0)
        # We will configure the decoder so that when action 2 is taken, it predicts a perfect state.
        # For other actions, it predicts high error states.
        gen_model = GenerativeModel()
        preferred_state = gen_model.get_preferred_state() # zeros
        
        class ScenarioDecoder:
            def __init__(self):
                self.current_action_idx = 0
                
            def decode(self, spikes):
                # We simulate that the action selection loop increments a counter or we map actions.
                # Since select_action loops through actions, we will return a perfect match for action 2.
                act = self.current_action_idx
                self.current_action_idx = (self.current_action_idx + 1) % 4
                
                if act == 2:
                    # Perfect match to preferred_state (all zeros)
                    return np.zeros(11, dtype=np.float32), 1.0
                else:
                    # High error state
                    return np.ones(11, dtype=np.float32) * 5.0, 0.0

        decoder = ScenarioDecoder()
        
        # Mock the encoder's encode_intention to track the action being simulated
        # to ensure the decoder knows which action is running.
        # Since select_action runs actions in order [0, 1, 2, 3]:
        # action 0 -> decode returns action 0's prediction
        # action 1 -> decode returns action 1's prediction
        # action 2 -> decode returns action 2's prediction (perfect state)
        # action 3 -> decode returns action 3's prediction
        
        best_action = select_action(
            snn=snn,
            current_state=np.random.randn(11),
            possible_actions=[0, 1, 2, 3],
            encoder=encoder,
            decoder=decoder,
            generative_model=gen_model
        )
        
        # Action 2 must be selected because it minimizes EFE
        self.assertEqual(best_action, 2)

if __name__ == '__main__':
    unittest.main()

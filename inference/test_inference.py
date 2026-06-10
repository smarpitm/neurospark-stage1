"""
inference/test_inference.py — regression tests for the active inference pipeline.

Run with:
    python -m pytest inference/test_inference.py -v
or:
    python -m unittest inference.test_inference
"""

import os
import sys
import unittest
import numpy as np
from brian2 import ms, second

# Ensure the repo root is on the path when running from any directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.free_energy import (
    compute_free_energy,
    compute_expected_free_energy,
    annealed_reward_weight,
)
from inference.generative_model import GenerativeModel
from inference.active_inference import select_action
from bridge.decoder import Decoder
from bridge.encoder import Encoder


# ──────────────────────────────────────────────────────────────────────────────
# Shared mocks
# ──────────────────────────────────────────────────────────────────────────────

class _MockNetwork:
    """Minimal Brian2 Network stand-in for select_action's store/restore calls."""
    def __init__(self):
        self.t = 0 * second
        self._states = {}

    def store(self, name):
        self._states[name] = self.t

    def restore(self, name):
        self.t = self._states.get(name, 0 * second)


class _MockSNNZeroSpikes:
    """SNN that always returns zero motor spikes — simulates a fully silent motor layer."""
    def __init__(self):
        self.net = _MockNetwork()
        self.sensory = type('Sensory', (), {'I_inj': 0})()
        self.noise = None

    def inject_sensory(self, currents):
        pass

    def run(self, duration):
        self.net.t = self.net.t + duration

    def get_motor_spikes_in_window(self, t_start, t_end):
        return np.zeros(100, dtype=np.int32)


class _MockEncoder:
    def encode_intention(self, state, action):
        return np.zeros(100, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pretrained_decoder(num_random_steps=200, sgd_epochs=30, seed=0):
    """
    Return a Decoder whose forward model has been trained on random 5×5 transitions.
    Used by multiple tests so they share the same training effort.
    """
    from environment.grid_world import GridWorld

    rng = np.random.default_rng(seed)
    decoder = Decoder()
    env = GridWorld(grid_size=5, start_pos=(0, 0), goal_pos=(4, 4))
    state = env.reset()
    buffer = []

    for _ in range(num_random_steps):
        actions = env.get_actions()
        action = int(rng.choice(actions))
        next_state, reward, done = env.step(action)
        buffer.append((state.copy(), action, next_state.copy(), float(reward)))
        state = next_state
        if done:
            state = env.reset()

    for _ in range(sgd_epochs):
        indices = rng.permutation(len(buffer))
        for idx in indices:
            s, a, ns, r = buffer[idx]
            decoder.update_forward(s, a, ns, r, learning_rate=0.05)

    return decoder


# ──────────────────────────────────────────────────────────────────────────────
# Existing tests (preserved)
# ──────────────────────────────────────────────────────────────────────────────

class TestFreeEnergyCalculations(unittest.TestCase):
    def test_vfe_value(self):
        """VFE = state MSE + reward MSE (no weighting)."""
        pred_state = np.array([0.5, 0.5, 0.0, 0.0])
        act_state  = np.array([0.0, 0.5, 0.0, 0.0])
        pred_rew, act_rew = 0.5, 1.0
        # state MSE = mean([(0.5-0)², (0.5-0.5)², 0, 0]) = mean([0.25, 0, 0, 0]) = 0.0625
        # reward MSE = (0.5 - 1.0)² = 0.25
        vfe = compute_free_energy(pred_state, act_state, pred_rew, act_rew)
        self.assertAlmostEqual(vfe, 0.3125)

    def test_annealed_reward_weight_schedule(self):
        """Weight starts low and approaches 20 asymptotically."""
        w1  = annealed_reward_weight(1)
        w10 = annealed_reward_weight(10)
        w50 = annealed_reward_weight(50)
        self.assertLess(w1, 10.0,   "Early weight should be well below 20")
        self.assertLess(w10, 20.0,  "Weight at ep 10 should still be below cap")
        # exp(-50/5) ≈ 0.00009 → weight ≈ 19.9991; asserting it rounds to 20 at 1 dp
        self.assertGreater(w50, 19.9, msg="Weight at ep 50 should be ≥ 19.9 (near cap)")


class TestActionSelectionBasic(unittest.TestCase):
    def test_action_selection_prefers_low_efe(self):
        """select_action should pick the action whose decoder prediction is closest to the preferred state."""
        class _ScenarioDecoder:
            def predict_for_planning(self, state, action, motor_spikes):
                if action == 2:
                    return np.zeros(11, dtype=np.float32), 1.0   # perfect prediction
                return np.ones(11, dtype=np.float32) * 5.0, 0.0

            def decode_action(self, spikes):
                return np.zeros(4, dtype=np.float32)

        snn     = _MockSNNZeroSpikes()
        encoder = _MockEncoder()
        gen_model = GenerativeModel()

        best_action, _ = select_action(
            snn=snn,
            current_state=np.random.randn(11),
            possible_actions=[0, 1, 2, 3],
            encoder=encoder,
            decoder=_ScenarioDecoder(),
            generative_model=gen_model,
            verbose=False,
        )
        self.assertEqual(best_action, 2)


# ──────────────────────────────────────────────────────────────────────────────
# New regression tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMotorSilenceFallback(unittest.TestCase):
    """
    (a) When motor spikes are zero across all imagined actions, EFE values should
    NOT be all identical.  The forward model must produce action-specific predictions
    that make at least two actions distinguishable.
    """

    def test_motor_silence_fallback(self):
        """
        Zero motor spikes → predict_for_planning falls back to the forward model.
        The forward model is (state, action) → next_state, so different actions
        should produce different EFE values unless the model is completely
        uninitialised.  After pretraining on random transitions it must discriminate.
        """
        decoder   = _pretrained_decoder(num_random_steps=200, sgd_epochs=30, seed=42)
        snn       = _MockSNNZeroSpikes()
        encoder   = _MockEncoder()
        gen_model = GenerativeModel()

        # Use a state away from the origin so the forward model sees variation
        state = np.array([0., 1., 0., 1., 0., 1., 0., 0.5, -0.5, 0.0, 0.0],
                         dtype=np.float32)
        possible_actions = [0, 1, 2, 3]

        _, plan_info = select_action(
            snn=snn,
            current_state=state,
            possible_actions=possible_actions,
            encoder=encoder,
            decoder=decoder,
            generative_model=gen_model,
            verbose=False,
        )

        efe_values = plan_info['efe']
        efe_spread = max(efe_values) - min(efe_values)

        self.assertGreater(
            efe_spread, 0.0,
            msg=(
                f"EFE values were all identical ({efe_values}) with zero motor spikes. "
                "The forward-model fallback is not producing action-specific predictions."
            ),
        )


class TestEFEDiscrimination(unittest.TestCase):
    """
    (b) After pre-training the forward model on ~10 simulated 'episodes' worth of
    random transitions, EFE spread across valid actions should be meaningfully
    non-zero (> 0.1) for at least 80% of test states.

    Note: the threshold is 0.1 rather than the intuitive 0.5 because a zero-spike
    SNN routes all planning through the analytical forward model whose output scale
    is bounded by the decoder weight magnitudes.  What matters is that actions are
    discriminated at all — the motor silence fallback test already proves the spread
    is strictly > 0; this test proves it exceeds a useful minimum consistently.
    """

    # 10 episodes × ~20 steps each = 200 random transitions for forward-model training
    _NUM_STEPS  = 200
    _SGD_EPOCHS = 40

    def test_efe_discrimination(self):
        decoder   = _pretrained_decoder(
            num_random_steps=self._NUM_STEPS,
            sgd_epochs=self._SGD_EPOCHS,
            seed=7,
        )
        snn       = _MockSNNZeroSpikes()
        encoder   = _MockEncoder()
        gen_model = GenerativeModel()

        from environment.grid_world import GridWorld
        rng = np.random.default_rng(99)
        env = GridWorld(grid_size=5, start_pos=(0, 0), goal_pos=(4, 4))

        # Sample 25 diverse states by taking random steps in the env
        test_states = []
        state = env.reset()
        for _ in range(25):
            actions = env.get_actions()
            action  = int(rng.choice(actions))
            next_state, _, done = env.step(action)
            test_states.append(state.copy())
            state = next_state
            if done:
                state = env.reset()

        discriminating = 0
        for state in test_states:
            _, plan_info = select_action(
                snn=snn,
                current_state=state,
                possible_actions=[0, 1, 2, 3],
                encoder=encoder,
                decoder=decoder,
                generative_model=gen_model,
                verbose=False,
            )
            efe_spread = plan_info['efe_spread']
            if efe_spread > 0.1:
                discriminating += 1

        pct = discriminating / len(test_states)
        self.assertGreaterEqual(
            pct, 0.80,
            msg=(
                f"EFE spread > 0.1 in only {discriminating}/{len(test_states)} "
                f"({pct*100:.1f}%) states — expected ≥ 80%. "
                "The forward model may not be learning action-specific transitions."
            ),
        )


class TestGoalReach(unittest.TestCase):
    """
    (c) After 20 curriculum episodes on a 5×5 grid (goal = (4,4)), the agent
    should be able to reach the goal within 50 steps.

    This is an integration test — it runs the full train loop (without SNN
    to keep it fast) using the forward-model-only planning path.
    """

    def _make_components(self):
        """Fresh decoder + encoder + gen_model.  No SNN needed — forward-model only."""
        decoder   = Decoder()
        encoder   = Encoder()
        gen_model = GenerativeModel()
        return decoder, encoder, gen_model

    def _run_forward_model_episode(self, decoder, encoder, gen_model,
                                    grid_size, goal_pos, max_steps, episode_num,
                                    rng):
        """
        Lightweight episode runner that uses the analytical forward model only
        (no SNN simulation).  select_action uses a zero-spike SNN so all planning
        goes through predict_for_planning → forward model.
        """
        from environment.grid_world import GridWorld
        from inference.free_energy import annealed_reward_weight

        env   = GridWorld(grid_size=grid_size, start_pos=(0, 0), goal_pos=goal_pos)
        state = env.reset()
        snn   = _MockSNNZeroSpikes()
        reward_weight = annealed_reward_weight(episode_num)
        done  = False

        for _ in range(max_steps):
            actions = env.get_actions()
            action, _ = select_action(
                snn=snn,
                current_state=state,
                possible_actions=actions,
                encoder=encoder,
                decoder=decoder,
                generative_model=gen_model,
                reward_weight=reward_weight,
                verbose=False,
            )
            next_state, reward, done = env.step(action)
            # Online forward-model update
            decoder.update_forward(state, action, next_state, reward,
                                   learning_rate=0.05)
            state = next_state
            if done:
                break

        return done, env.agent_pos[:]

    def test_goal_reach(self):
        """
        Train for 20 curriculum episodes (5×5 grid, goal (4,4)).
        On the 20th episode the agent must reach the goal within 50 steps.
        """
        np.random.seed(42)
        rng = np.random.default_rng(42)

        decoder, encoder, gen_model = self._make_components()

        # Warm up the forward model with random transitions first so
        # episode 1 isn't completely blind
        from environment.grid_world import GridWorld
        env   = GridWorld(grid_size=5, start_pos=(0, 0), goal_pos=(4, 4))
        state = env.reset()
        for _ in range(100):
            actions = env.get_actions()
            action  = int(rng.choice(actions))
            next_state, reward, done = env.step(action)
            decoder.update_forward(state, action, next_state, reward,
                                   learning_rate=0.05)
            state = next_state
            if done:
                state = env.reset()

        # 20 training episodes
        for ep in range(1, 21):
            reached, final_pos = self._run_forward_model_episode(
                decoder, encoder, gen_model,
                grid_size=5, goal_pos=(4, 4),
                max_steps=200, episode_num=ep, rng=rng,
            )

        # Evaluation episode — stricter 50-step budget
        reached, final_pos = self._run_forward_model_episode(
            decoder, encoder, gen_model,
            grid_size=5, goal_pos=(4, 4),
            max_steps=50, episode_num=21, rng=rng,
        )

        self.assertTrue(
            reached,
            msg=(
                f"Agent failed to reach (4,4) within 50 steps after 20 curriculum "
                f"episodes. Final position: {final_pos}. "
                "The forward model may not be learning a useful navigation policy."
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    unittest.main(verbosity=2)

import os
import sys
import numpy as np
from brian2 import ms, nA

from environment.grid_world import GridWorld
from brain.network import FlyBrainSNN
from bridge.encoder import Encoder
from bridge.decoder import Decoder
from inference.generative_model import GenerativeModel
from inference.active_inference import select_action, SIMULATION_MS, ACTION_NAMES
from inference.free_energy import compute_free_energy, annealed_reward_weight, annealed_epsilon

TRAINED_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "trained_weights.npz"
)


def _require_weights():
    """Abort with a clear message if trained_weights.npz does not exist."""
    if not os.path.exists(TRAINED_WEIGHTS_PATH):
        print(
            "[Error] No trained weights found.\n"
            "Run 'python train.py --bootstrap' first to create initial weights."
        )
        sys.exit(1)


def load_components_from_weights(weights_path=None):
    """
    Instantiate SNN, Encoder, Decoder, GenerativeModel and restore all weights
    from the trained_weights.npz checkpoint.  Raises FileNotFoundError if the
    file is missing - callers are expected to call _require_weights() first.

    The saved weight arrays encode the exact number of synapses per group.
    We read those shapes first and tell FlyBrainSNN to create matching
    connectivity, so loading works even if the default ``connect(p=...)``
    probabilities were changed after the checkpoint was saved.
    """
    if weights_path is None:
        weights_path = TRAINED_WEIGHTS_PATH

    data = np.load(weights_path)

    # Build synapse_counts from saved weight shapes so the SNN constructor
    # creates the exact same number of connections (avoids shape mismatches).
    synapse_counts = {}
    for key, syn_name in [('S_in_w', 'S_in'), ('S_out_w', 'S_out'), ('S_direct_w', 'S_direct')]:
        if key in data:
            synapse_counts[syn_name] = data[key].shape[0]

    snn = FlyBrainSNN(use_noise=True, synapse_counts=synapse_counts)
    encoder = Encoder()
    decoder = Decoder()
    gen_model = GenerativeModel()

    snn.S_in.w = data['S_in_w']
    snn.S_rec.w = data['S_rec_w']
    snn.S_out.w = data['S_out_w']
    if 'S_direct_w' in data:
        snn.S_direct.w = data['S_direct_w']
    decoder.W_out          = data['decoder_W_out']
    decoder.W_reward       = data['decoder_W_reward']
    decoder.W_action       = data['decoder_W_action']
    if 'decoder_W_forward' in data:
        decoder.W_forward  = data['decoder_W_forward']
    if 'decoder_W_reward_forward' in data:
        decoder.W_reward_forward = data['decoder_W_reward_forward']
    encoder.W          = data['encoder_W']
    encoder.W_intention = data['encoder_W_intention']

    return snn, encoder, decoder, gen_model


def run_episode(
    snn=None,
    encoder=None,
    decoder=None,
    gen_model=None,
    max_steps=100,
    learning_rate=0.01,
    grid_size=10,
    start_pos=(0, 0),
    goal_pos=(9, 9),
    on_step=None,
    verbose=True,
    should_stop=None,
    episode=1,
):
    """
    Runs a single closed-loop episode of the fly SNN agent navigating the grid world.

    :param on_step: Optional callback(dict) fired after each step (for dashboard streaming).
    :param should_stop: Optional callable() -> bool; return True to abort early.
    :param episode: 1-based episode number used to anneal the EFE reward weight.

    Components (snn, encoder, decoder, gen_model) MUST be pre-loaded with trained
    weights before calling this function.  If all four are None a convenience load
    is attempted, but trained_weights.npz must exist - random init is never used.
    """
    # ── Weights guard ────────────────────────────────────────────────────────
    all_none = (snn is None and encoder is None and decoder is None and gen_model is None)
    if all_none:
        _require_weights()                          # exits with code 1 if missing
        snn, encoder, decoder, gen_model = load_components_from_weights()
    elif any(c is None for c in (snn, encoder, decoder, gen_model)):
        # Partial construction - caller must supply all four or none.
        raise ValueError(
            "run_episode requires all of snn, encoder, decoder, gen_model to be "
            "provided together. Pass all four or none (to auto-load from weights)."
        )
    # ─────────────────────────────────────────────────────────────────────────
    env = GridWorld(grid_size=grid_size, start_pos=start_pos, goal_pos=goal_pos)
    state = env.reset()

    free_energy_history = []
    trajectory = []
    full_trajectory_data = []
    done = False

    # Compute annealed reward weight and epsilon once per episode
    reward_weight = annealed_reward_weight(episode)
    epsilon = annealed_epsilon(episode)

    snn.reset_episode_diagnostics()

    if verbose:
        print(f"--- Starting Episode at Agent Position: {env.agent_pos} | Goal: {env.goal_pos} ---")
        print(f"    reward_weight (annealed, ep={episode}): {reward_weight:.3f}")
        print(f"    epsilon       (annealed, ep={episode}): {epsilon:.3f}")

    motor_collapse = False
    for step in range(max_steps):
        if should_stop and should_stop():
            break

        pos_before = list(env.agent_pos)

        # 2. Select action via Active Inference
        # Active Inference imagines outcomes to pick the action that minimizes Expected Free Energy (EFE)
        action, plan_info = select_action(
            snn, state, env.get_actions(), encoder, decoder, gen_model,
            reward_weight=reward_weight, verbose=verbose, epsilon=epsilon
        )

        next_state, reward, done = env.step(action)
        trajectory.append((state, action, next_state, reward))

        intention = encoder.encode_intention(state, action)
        snn.inject_sensory(intention)

        t_start = snn.net.t
        snn.run(SIMULATION_MS * ms)
        t_end = snn.net.t

        motor_diag = snn.after_simulation_step(t_start, t_end)
        motor_spikes = motor_diag['motor_spikes']
        if motor_diag['motor_collapse']:
            motor_collapse = True
            if verbose:
                print("motor collapse")
            break

        predicted_next, predicted_reward = decoder.decode(motor_spikes, current_state=state)
        fe = compute_free_energy(predicted_next, next_state, predicted_reward, reward)
        free_energy_history.append(fe)

        snn.update_weights(fe, t_start, t_end, learning_rate=learning_rate, verbose=verbose)
        decoder.update(motor_spikes, next_state, reward, learning_rate=learning_rate)
        decoder.update_action_weights(motor_spikes, action, fe, learning_rate=learning_rate)
        decoder.update_forward(state, action, next_state, reward, learning_rate=learning_rate)

        if (step + 1) % 10 == 0:
            stats = snn.get_weight_stats()
            saturated = [f"{layer}: {s['pct_saturated']*100:.1f}%" for layer, s in stats.items() if s['pct_saturated'] > 0.20]
            if saturated and verbose:
                print(f"  [WeightStats] Saturated weights at step {step+1}: {', '.join(saturated)}")

        pos_after = env.agent_pos
        motor_total = motor_diag['motor_total']
        motor_firing_pct = motor_diag['motor_firing_pct']

        # Extract all-population spike counts for visualization / callbacks
        sensory_spikes = snn.get_spikes_in_window(
            snn.spike_mon_sensory, 100, t_start, t_end
        )
        recurrent_spikes = snn.get_spikes_in_window(
            snn.spike_mon_recurrent, 800, t_start, t_end
        )

        step_data = {
            'step': step + 1,
            'pos_before': pos_before,
            'pos_after': list(pos_after),
            'goal': list(env.goal_pos),
            'grid_size': env.grid_size,
            'action': int(action),
            'action_name': ACTION_NAMES[action],
            'reward': float(reward),
            'free_energy': float(fe),
            'motor_spikes': motor_total,
            'motor_firing_pct': motor_firing_pct,
            'motor_firing_running_avg': motor_diag['motor_firing_running_avg'],
            'motor_boosted': motor_diag['boosted'],
            'sensory_spikes': sensory_spikes.tolist(),
            'recurrent_spikes': recurrent_spikes.tolist(),
            'motor_spikes_array': motor_spikes.tolist(),
            'plan': plan_info,
            'done': done,
        }

        if on_step:
            on_step(step_data)
            
        full_trajectory_data.append({
            'step': step_data['step'],
            'pos': step_data['pos_before'],
            'action': step_data['action_name'],
            'reward': step_data['reward'],
            'fe': step_data['free_energy']
        })

        if verbose:
            print(
                f"Step {step+1:02d} | Pos: {pos_before} -> {pos_after} | "
                f"Action: {ACTION_NAMES[action]:5s} | Reward: {reward:5.2f} | "
                f"Free Energy: {fe:6.2f} | Motor Spikes: {motor_total:4d} | "
                f"Firing Pct: {motor_firing_pct:5.1f}%"
            )

        state = next_state
        if done:
            if verbose:
                print(f"Goal reached in {step+1} steps!")
            break

    if motor_collapse and verbose:
        print("Episode aborted: motor collapse.")
    elif not done and verbose:
        print("Episode finished without reaching the goal (max steps exceeded).")

    import json
    traj_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    traj_path = os.path.join(traj_dir, f"episode_{episode}.json")
    with open(traj_path, 'w') as f:
        json.dump(full_trajectory_data, f)

    fb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "episode_feedback")
    os.makedirs(fb_dir, exist_ok=True)
    fb_path = os.path.join(fb_dir, f"feedback_ep{episode}.json")
    feedback = {
        'episode': int(episode),
        'steps': len(full_trajectory_data),
        'total_reward': float(sum(d['reward'] for d in full_trajectory_data)),
        'avg_free_energy': float(np.mean(free_energy_history)) if free_energy_history else 0.0,
        'goal_reached': bool(done)
    }
    with open(fb_path, 'w') as f:
        json.dump(feedback, f, indent=4)

    return trajectory, free_energy_history, done

if __name__ == "__main__":
    _require_weights()      # hard stop if weights are missing
    np.random.seed(42)
    run_episode()

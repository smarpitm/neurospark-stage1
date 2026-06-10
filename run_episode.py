import numpy as np
from brian2 import ms, second

from environment.grid_world import GridWorld
from brain.network import FlyBrainSNN
from bridge.encoder import Encoder
from bridge.decoder import Decoder
from inference.generative_model import GenerativeModel
from inference.active_inference import select_action
from inference.free_energy import compute_free_energy

def run_episode(snn=None, encoder=None, decoder=None, gen_model=None, max_steps=100, learning_rate=0.01, grid_size=10, start_pos=(0, 0), goal_pos=(9, 9)):
    """
    Runs a single closed-loop episode of the fly SNN agent navigating the grid world.
    """
    # 1. Initialize environment
    env = GridWorld(grid_size=grid_size, start_pos=start_pos, goal_pos=goal_pos)
    state = env.reset()
    
    # 2. Initialize SNN, encoder, decoder, and generative model if not provided
    if snn is None:
        snn = FlyBrainSNN(use_noise=True)
    if encoder is None:
        encoder = Encoder()
    if decoder is None:
        decoder = Decoder()
    if gen_model is None:
        gen_model = GenerativeModel()

    # Reset any spike monitor data or start from current time
    t_episode_start = snn.net.t
    
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
    free_energy_history = []
    trajectory = []
    
    print(f"--- Starting Episode at Agent Position: {env.agent_pos} | Goal: {env.goal_pos} ---")
    
    for step in range(max_steps):
        # Cache current position for logging
        pos_before = list(env.agent_pos)
        
        # 2. Select action via Active Inference
        # Active Inference imagines outcomes to pick the action that minimizes Expected Free Energy (EFE)
        action = select_action(snn, state, env.get_actions(), encoder, decoder, gen_model)
        
        # 3. Execute action in environment
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, next_state, reward))
        
        # 4. Run SNN for the actual step with the chosen intention
        # Encode the actual intention (state + chosen action) and inject it
        intention = encoder.encode_intention(state, action)
        snn.inject_sensory(intention)
        
        t_start = snn.net.t
        snn.run(50 * ms)
        t_end = snn.net.t
        
        # Retrieve actual spikes generated during this step
        motor_spikes = snn.get_motor_spikes_in_window(t_start, t_end)
        
        # 5. Decode predicted next state and predicted reward
        predicted_next, predicted_reward = decoder.decode(motor_spikes)
        
        # 6. Compute prediction error (Free Energy)
        fe = compute_free_energy(predicted_next, next_state, predicted_reward, reward)
        free_energy_history.append(fe)
        
        # 7. Update network weights (learning)
        # Update SNN internal weights
        snn.update_weights(fe, t_start, t_end, learning_rate=learning_rate)
        # Update Decoder weights (State and Reward predictions)
        decoder.update(motor_spikes, next_state, reward, learning_rate=learning_rate)
        # Update Decoder action policy weights
        decoder.update_action_weights(motor_spikes, action, fe, learning_rate=learning_rate)
        
        # Print telemetry log
        pos_after = env.agent_pos
        motor_firing_pct = (np.sum(motor_spikes > 0) / len(motor_spikes)) * 100
        print(f"Step {step+1:02d} | Pos: {pos_before} -> {pos_after} | Action: {action_names[action]:5s} | Reward: {reward:5.2f} | Free Energy: {fe:6.2f} | Motor Spikes: {np.sum(motor_spikes):4d} | Firing Pct: {motor_firing_pct:5.1f}%")
        
        state = next_state
        if done:
            print(f"Goal reached in {step+1} steps!")
            break
            
    if not done:
        print("Episode finished without reaching the goal (max steps exceeded).")
        
    return trajectory, free_energy_history

if __name__ == "__main__":
    # Seed numpy random number generator for reproducibility
    np.random.seed(42)
    run_episode()

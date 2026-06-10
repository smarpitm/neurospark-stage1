import os
import sys
import numpy as np
from brian2 import ms
import json

# Force headless rendering for Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import imageio

from environment.grid_world import GridWorld
from environment.renderer import Renderer
from run_episode import load_components_from_weights, _require_weights
from inference.active_inference import select_action, SIMULATION_MS, ACTION_NAMES
from inference.free_energy import compute_free_energy, annealed_reward_weight

def get_current_training_episode():
    meta_path = os.path.join("data", "training_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            data = json.load(f)
            return data.get("total_episodes", 109)
    return 109

def main():
    _require_weights()
    print("Loading network weights...")
    snn, encoder, decoder, gen_model = load_components_from_weights()
    
    videos_dir = "videos"
    os.makedirs(videos_dir, exist_ok=True)
    
    current_ep = get_current_training_episode()
    print(f"Starting rendering for {current_ep} episodes into {videos_dir}/")
    
    for ep in range(1, current_ep + 1):
        vid_path = os.path.join(videos_dir, f"episode_{ep:03d}.mp4")
        if os.path.exists(vid_path):
            print(f"Skipping Episode {ep:03d}, video already exists.")
            continue
            
        print(f"Rendering Episode {ep:03d} to {vid_path}...")
        
        # Determine grid size based on curriculum
        grid_size = 5 if ep <= 75 else 7 if ep <= 95 else 10
        start_pos = (0, 0)
        goal_pos = (4, 4) if grid_size == 5 else (6, 6) if grid_size == 7 else (9, 9)
        
        env = GridWorld(grid_size=grid_size, start_pos=start_pos, goal_pos=goal_pos)
        state = env.reset()
        
        renderer = Renderer(env)
        
        # 15 fps gives a good balance of speed vs file size
        writer = imageio.get_writer(vid_path, fps=15)
        
        reward_weight = annealed_reward_weight(ep)
        snn.reset_episode_diagnostics()
        
        motor_collapse = False
        
        for step in range(150): # max 150 steps
            action, plan_info = select_action(
                snn, state, env.get_actions(), encoder, decoder, gen_model,
                reward_weight=reward_weight, verbose=False
            )
            
            next_state, reward, done = env.step(action)
            
            intention = encoder.encode_intention(state, action)
            snn.inject_sensory(intention)
            
            t_start = snn.net.t
            snn.run(SIMULATION_MS * ms)
            t_end = snn.net.t
            
            motor_diag = snn.after_simulation_step(t_start, t_end)
            motor_spikes = motor_diag['motor_spikes']
            
            if motor_diag['motor_collapse']:
                motor_collapse = True
                break
                
            predicted_next, predicted_reward = decoder.decode(motor_spikes, current_state=state)
            fe = compute_free_energy(predicted_next, next_state, predicted_reward, reward)
            
            sensory_spikes = snn.get_spikes_in_window(snn.spike_mon_sensory, 100, t_start, t_end)
            recurrent_spikes = snn.get_spikes_in_window(snn.spike_mon_recurrent, 800, t_start, t_end)
            
            renderer.update_snn_data({
                'sensory_spikes': sensory_spikes,
                'recurrent_spikes': recurrent_spikes,
                'motor_spikes': motor_spikes,
                'motor_total': motor_diag['motor_total'],
                'action_name': ACTION_NAMES[action],
                'free_energy': float(fe),
                'motor_firing_pct': motor_diag['motor_firing_pct'],
                'step': step + 1,
            })
            
            renderer.step_count = step + 1
            renderer.cumulative_reward += reward
            renderer.last_action = ACTION_NAMES[action]
            renderer.last_reward = reward
            
            curr_pos = tuple(env.agent_pos)
            if not renderer.trail or renderer.trail[-1] != curr_pos:
                renderer.trail.append(curr_pos)
                
            # Snap agent visual position instantly for video
            renderer.visual_x = float(env.agent_pos[0])
            renderer.visual_y = float(env.agent_pos[1])
            
            # Draw frame
            renderer.draw(render_only=True)
            
            # Capture frame and convert color format
            # Pygame array3d is (x, y, rgb), imageio expects (y, x, rgb)
            frame = pygame.surfarray.array3d(renderer.screen)
            frame = np.transpose(frame, (1, 0, 2))
            writer.append_data(frame)
            
            state = next_state
            
            if done:
                # Add 2 seconds (30 frames) of hold time at the end to show goal reached
                for _ in range(30):
                    writer.append_data(frame)
                break
                
        writer.close()
        
        # Important: clean up Brian2 monitors to prevent Memory leaks
        snn.reset_monitors()
        pygame.quit() # cleanup this headless renderer instance

if __name__ == "__main__":
    main()

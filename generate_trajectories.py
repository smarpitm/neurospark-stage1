import os
import sys

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from run_episode import load_components_from_weights, run_episode, _require_weights
import json

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
    
    current_ep = get_current_training_episode()
    print(f"Generating missing trajectories up to episode {current_ep}...")
    
    for ep in range(2, current_ep + 1):
        traj_file = os.path.join("data", "trajectories", f"episode_{ep}.json")
        if os.path.exists(traj_file):
            continue
            
        print(f"Generating trajectory for Episode {ep}...")
        
        # Adjust grid size based on episode number to match curriculum
        grid_size = 5 if ep <= 75 else 7 if ep <= 95 else 10
        goal_pos = (4, 4) if grid_size == 5 else (6, 6) if grid_size == 7 else (9, 9)
        
        run_episode(
            snn=snn,
            encoder=encoder,
            decoder=decoder,
            gen_model=gen_model,
            grid_size=grid_size,
            goal_pos=goal_pos,
            episode=ep,
            verbose=False,
            max_steps=100
        )
        print(f"-> Saved {traj_file}")
        snn.reset_monitors()
        
if __name__ == "__main__":
    main()

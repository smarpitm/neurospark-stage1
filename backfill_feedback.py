import os
import glob
import json
import numpy as np

def main():
    traj_dir = os.path.join('data', 'trajectories')
    fb_dir = os.path.join('data', 'episode_feedback')
    os.makedirs(fb_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(traj_dir, 'episode_*.json'))
    print(f"Found {len(files)} trajectory files to backfill.")
    
    for file in files:
        filename = os.path.basename(file)
        # extract episode number from episode_X.json
        ep_num = filename.replace('episode_', '').replace('.json', '')
        
        with open(file, 'r') as f:
            data = json.load(f)
            
        if not data:
            continue
            
        steps = len(data)
        total_reward = sum(step.get('reward', 0) for step in data)
        avg_fe = np.mean([step.get('free_energy', 0) for step in data])
        
        # Check if goal reached based on last position
        # Depending on grid curriculum, goal is different. But we can just check if last reward was >= 0 (since collisions are -1, steps are -0.01 or similar, and goal is 1.0 or 10.0 or similar)
        last_reward = data[-1].get('reward', 0)
        goal_reached = last_reward > 0
        
        feedback = {
            'episode': int(ep_num),
            'steps': steps,
            'total_reward': float(total_reward),
            'avg_free_energy': float(avg_fe),
            'goal_reached': goal_reached
        }
        
        fb_path = os.path.join(fb_dir, f'feedback_ep{ep_num}.json')
        with open(fb_path, 'w') as f:
            json.dump(feedback, f, indent=4)
            
    print("Backfill complete.")

if __name__ == '__main__':
    main()

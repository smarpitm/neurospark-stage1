import os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import ms

from brain.network import FlyBrainSNN
from bridge.encoder import Encoder
from bridge.decoder import Decoder
from inference.generative_model import GenerativeModel
from run_episode import run_episode

def train(num_episodes=20, max_steps=100, learning_rate=0.01):
    """
    Trains the SNN and decoder over multiple episodes and saves the resulting weights.
    """
    print("==========================================")
    print(f"Starting SNN Training for {num_episodes} Episodes (Curriculum)")
    print("==========================================\n")
    
    # 1. Initialize persistent agent components
    # Using noise so that recurrent dynamics are active
    snn = FlyBrainSNN(use_noise=True)
    encoder = Encoder()
    decoder = Decoder()
    gen_model = GenerativeModel()
    
    steps_history = []
    avg_fe_history = []
    
    for ep in range(num_episodes):
        # Reset SNN monitors at the start of each episode to clear spike history.
        # This keeps SNN simulations fast by preventing monitor size accumulation.
        if ep > 0:
            snn.reset_monitors()
            
        # Curriculum: 5x5 grid for the first half, 10x10 grid for the second half
        if ep < num_episodes // 2:
            grid_size = 5
            goal_pos = (4, 4)
            stage_name = "5x5 Grid"
        else:
            grid_size = 10
            goal_pos = (9, 9)
            stage_name = "10x10 Grid"
            
        print(f"\n--- Episode {ep+1}/{num_episodes} ({stage_name} | Goal: {goal_pos}) ---")
        
        # Run the episode
        trajectory, fe_history = run_episode(
            snn=snn,
            encoder=encoder,
            decoder=decoder,
            gen_model=gen_model,
            max_steps=max_steps,
            learning_rate=learning_rate,
            grid_size=grid_size,
            start_pos=(0, 0),
            goal_pos=goal_pos
        )
        
        steps_taken = len(fe_history)
        avg_fe = np.mean(fe_history)
        
        steps_history.append(steps_taken)
        avg_fe_history.append(avg_fe)
        
        print(f"Episode {ep+1} Summary ({stage_name}): Steps = {steps_taken} | Avg Free Energy = {avg_fe:.4f}")
        
    # 2. Plot and save training curves
    plt.figure(figsize=(12, 5))
    
    # Plot Steps to Goal
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_episodes + 1), steps_history, marker='o', color='#38bdf8', linewidth=2)
    plt.title("Steps to Goal per Episode", fontsize=12, fontweight='bold', color='white')
    plt.xlabel("Episode", fontsize=10, color='white')
    plt.ylabel("Steps Taken", fontsize=10, color='white')
    plt.grid(True, color='#334155', linestyle=':', alpha=0.5)
    plt.xticks(range(1, num_episodes + 1))
    
    # Plot Average Free Energy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_episodes + 1), avg_fe_history, marker='s', color='#10b981', linewidth=2)
    plt.title("Avg Free Energy per Episode", fontsize=12, fontweight='bold', color='white')
    plt.xlabel("Episode", fontsize=10, color='white')
    plt.ylabel("Variational Free Energy", fontsize=10, color='white')
    plt.grid(True, color='#334155', linestyle=':', alpha=0.5)
    plt.xticks(range(1, num_episodes + 1))
    
    # Aesthetic dark theme adjustments
    for ax in plt.gcf().axes:
        ax.set_facecolor('#0f172a')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.gcf().patch.set_facecolor('#0f172a')
    plt.tight_layout()
    
    plot_path = "training_curves.png"
    plt.savefig(plot_path, dpi=300, facecolor=plt.gcf().get_facecolor(), bbox_inches='tight')
    print(f"\nTraining curves saved to '{plot_path}'")
    
    # 3. Save the trained weights to data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    weights_path = os.path.join(data_dir, "trained_weights.npz")
    
    np.savez(
        weights_path,
        S_in_w=np.array(snn.S_in.w),
        S_rec_w=np.array(snn.S_rec.w),
        S_out_w=np.array(snn.S_out.w),
        decoder_W_out=decoder.W_out,
        decoder_W_reward=decoder.W_reward,
        decoder_W_action=decoder.W_action,
        encoder_W=encoder.W,
        encoder_W_intention=encoder.W_intention
    )
    print(f"Trained weights saved to '{weights_path}'")
    
    print("\nTraining complete! The agent should show improving navigation efficiency (fewer steps and lower free energy).")

if __name__ == "__main__":
    np.random.seed(42)
    train()

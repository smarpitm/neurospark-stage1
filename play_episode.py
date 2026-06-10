# play_episode.py
"""Run a full episode with the SNN agent and record a video of the game.
This script uses the existing Renderer (pygame) to display each step and
captures each frame to produce an MP4 video using imageio.
"""

import os
import numpy as np
from brian2 import ms
import pygame
import imageio

from environment.grid_world import GridWorld
from brain.network import FlyBrainSNN
from bridge.encoder import Encoder
from bridge.decoder import Decoder
from inference.generative_model import GenerativeModel
from inference.active_inference import select_action
from inference.free_energy import compute_free_energy
from environment.renderer import Renderer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_VIDEO = "episode.mp4"
FPS = 30  # frames per second for the saved video
MAX_STEPS = 200
LEARNING_RATE = 0.01

def run_and_record():
    # 1️⃣ Initialise environment and components
    env = GridWorld()
    state = env.reset()
    snn = FlyBrainSNN(use_noise=True)
    encoder = Encoder()
    decoder = Decoder()
    gen_model = GenerativeModel()

    # 2️⃣ Initialise the visualiser (pygame)
    renderer = Renderer(env)
    renderer.draw()  # initial draw
    pygame.display.flip()

    frames = []  # collected RGB frames for video

    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    for step in range(MAX_STEPS):
        # ---- Select action via Active Inference ----
        action = select_action(snn, state, env.get_actions(), encoder, decoder, gen_model)

        # ---- Execute action in the GridWorld ----
        next_state, reward, done = env.step(action)

        # ---- Run the SNN for this step ----
        intention = encoder.encode_intention(state, action)
        snn.inject_sensory(intention)
        t_start = snn.net.t
        snn.run(50 * ms)
        t_end = snn.net.t
        motor_spikes = snn.get_motor_spikes_in_window(t_start, t_end)

        # ---- Decode predictions and compute free energy ----
        predicted_next, predicted_reward = decoder.decode(motor_spikes)
        fe = compute_free_energy(predicted_next, next_state, predicted_reward, reward)

        # ---- Learning updates ----
        snn.update_weights(fe, t_start, t_end, learning_rate=LEARNING_RATE)
        decoder.update(motor_spikes, next_state, reward, learning_rate=LEARNING_RATE)
        decoder.update_action_weights(motor_spikes, action, fe, learning_rate=LEARNING_RATE)

        # ---- Render the current frame ----
        renderer.draw()
        pygame.display.flip()
        # Capture the pygame surface as an RGB array (WxH -> HxW)
        frame = pygame.surfarray.array3d(renderer.screen)
        frame = frame.swapaxes(0, 1)  # Convert to (height, width, 3)
        frames.append(frame)

        # ---- Telemetry (optional console output) ----
        print(
            f"Step {step+1:02d} | Action: {action_names[action]} | Reward: {reward:+.2f} "
            f"| Free Energy: {fe:.2f} | Spikes: {np.sum(motor_spikes)}"
        )

        state = next_state
        if done:
            print(f"Goal reached in {step+1} steps!")
            break

    # -------------------------------------------------------------------
    # Save video
    # -------------------------------------------------------------------
    print(f"Saving video to {OUTPUT_VIDEO} …")
    imageio.mimsave(OUTPUT_VIDEO, frames, fps=FPS)
    print("Video saved successfully.")

    # Clean up pygame
    pygame.quit()

if __name__ == "__main__":
    # Ensure the required libraries are available – fail fast with a clear message.
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required to record videos. "
            "Run 'pip install imageio' (or add it to requirements.txt) and retry."
        )

    run_and_record()

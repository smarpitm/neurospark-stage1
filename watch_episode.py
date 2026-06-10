"""
watch_episode.py
================
Run a full SNN Active-Inference episode with real-time Pygame visualization.

Usage:
    python watch_episode.py                 # default 300ms per step
    python watch_episode.py --speed 500     # 500ms per step (slower)
    python watch_episode.py --no-verbose    # suppress console output
"""

import os
import sys
import argparse
import numpy as np
from brian2 import ms

# pyrefly: ignore [missing-import]
import pygame

from environment.grid_world import GridWorld
from environment.renderer import Renderer
from brain.network import FlyBrainSNN
from bridge.encoder import Encoder
from bridge.decoder import Decoder
from inference.generative_model import GenerativeModel
from inference.active_inference import select_action, SIMULATION_MS, ACTION_NAMES
from inference.free_energy import compute_free_energy, annealed_reward_weight
from run_episode import load_components_from_weights, _require_weights, TRAINED_WEIGHTS_PATH


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch the SNN agent navigate the grid world in real-time."
    )
    parser.add_argument(
        "--speed", type=int, default=300,
        help="Delay in milliseconds between steps (default: 300)."
    )
    parser.add_argument(
        "--no-verbose", action="store_true",
        help="Suppress console output."
    )
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="Maximum steps per episode (default: 100)."
    )
    parser.add_argument(
        "--episode", type=int, default=1,
        help="Episode number for reward weight annealing (default: 1)."
    )
    return parser.parse_args()


def watch_episode():
    args = parse_args()
    verbose = not args.no_verbose

    # ── Require trained weights ──────────────────────────────────────────
    _require_weights()

    # ── Load all components ──────────────────────────────────────────────
    if verbose:
        print("[watch] Loading trained weights …")
    snn, encoder, decoder, gen_model = load_components_from_weights()

    # ── Environment & Renderer ───────────────────────────────────────────
    env = GridWorld(grid_size=10, start_pos=(0, 0), goal_pos=(9, 9))
    state = env.reset()
    renderer = Renderer(env)

    reward_weight = annealed_reward_weight(args.episode)
    snn.reset_episode_diagnostics()

    if verbose:
        print(f"[watch] Agent @ {env.agent_pos}  Goal @ {env.goal_pos}")
        print(f"[watch] reward_weight={reward_weight:.3f}  speed={args.speed}ms/step")
        print("[watch] Press ESC or close the window to quit.\n")

    # ── Main loop ────────────────────────────────────────────────────────
    running = True
    done = False
    motor_collapse = False

    for step in range(args.max_steps):
        if not running:
            break

        # Handle Pygame events (quit / ESC)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        if not running:
            break

        pos_before = list(env.agent_pos)

        # ── Select action via Active Inference ───────────────────────────
        action, plan_info = select_action(
            snn, state, env.get_actions(), encoder, decoder, gen_model,
            reward_weight=reward_weight, verbose=verbose,
        )

        next_state, reward, done = env.step(action)

        # ── SNN simulation for this step ─────────────────────────────────
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
                print("[watch] Motor collapse detected — aborting episode.")
            break

        # ── Decode & learn ───────────────────────────────────────────────
        predicted_next, predicted_reward = decoder.decode(motor_spikes, current_state=state)
        fe = compute_free_energy(predicted_next, next_state, predicted_reward, reward)

        snn.update_weights(fe, t_start, t_end, learning_rate=0.01, verbose=verbose)
        decoder.update(motor_spikes, next_state, reward, learning_rate=0.01)
        decoder.update_action_weights(motor_spikes, action, fe, learning_rate=0.01)
        decoder.update_forward(state, action, next_state, reward, learning_rate=0.01)

        # ── Extract all-population spikes for heatmap ────────────────────
        sensory_spikes = snn.get_spikes_in_window(
            snn.spike_mon_sensory, 100, t_start, t_end
        )
        recurrent_spikes = snn.get_spikes_in_window(
            snn.spike_mon_recurrent, 800, t_start, t_end
        )

        # ── Feed SNN data to renderer ────────────────────────────────────
        renderer.update_snn_data({
            'sensory_spikes':   sensory_spikes,
            'recurrent_spikes': recurrent_spikes,
            'motor_spikes':     motor_spikes,
            'motor_total':      motor_diag['motor_total'],
            'action_name':      ACTION_NAMES[action],
            'free_energy':      float(fe),
            'motor_firing_pct': motor_diag['motor_firing_pct'],
            'step':             step + 1,
        })

        # ── Update renderer game stats ───────────────────────────────────
        renderer.step_count = step + 1
        renderer.cumulative_reward += reward
        renderer.last_action = ACTION_NAMES[action]
        renderer.last_reward = reward

        # Update trail
        curr_pos = tuple(env.agent_pos)
        if not renderer.trail or renderer.trail[-1] != curr_pos:
            renderer.trail.append(curr_pos)

        # Snap visual position (no smooth lerp needed at 300ms steps)
        renderer.visual_x = float(env.agent_pos[0])
        renderer.visual_y = float(env.agent_pos[1])

        # ── Render frame ─────────────────────────────────────────────────
        renderer.draw()
        pygame.display.flip()

        if verbose:
            pos_after = env.agent_pos
            print(
                f"Step {step+1:02d} | Pos: {pos_before} -> {list(pos_after)} | "
                f"Action: {ACTION_NAMES[action]:5s} | Reward: {reward:5.2f} | "
                f"FE: {fe:6.2f} | Motor: {motor_diag['motor_total']:4d} | "
                f"Firing: {motor_diag['motor_firing_pct']:5.1f}%"
            )

        state = next_state

        if done:
            if verbose:
                print(f"\n[watch] Goal reached in {step+1} steps!")
            # Show the final frame for a couple seconds
            pygame.time.wait(2000)
            break

        # Delay for watchability
        pygame.time.wait(args.speed)

    # ── End of episode ───────────────────────────────────────────────────
    if motor_collapse and verbose:
        print("[watch] Episode aborted: motor collapse.")
    elif not done and running and verbose:
        print("[watch] Episode finished without reaching goal (max steps).")

    # Keep window open until user closes it
    if running:
        if verbose:
            print("\n[watch] Episode finished. Close the window or press ESC to exit.")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    waiting = False
            renderer.clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    watch_episode()

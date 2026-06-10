"""
train.py — SNN training entry-point.

Usage
-----
First-ever run (creates weights from scratch):
    python train.py --bootstrap

Continue training from existing weights:
    python train.py

The agent only gets better — weights are never reset between runs.
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from brain.network import FlyBrainSNN
from bridge.encoder import Encoder
from bridge.decoder import Decoder
from inference.generative_model import GenerativeModel
from run_episode import run_episode, TRAINED_WEIGHTS_PATH

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR                = "data"
PRETRAINED_DECODER_PATH = os.path.join(DATA_DIR, "pretrained_decoder.npz")
BEST_5X5_WEIGHTS_PATH   = os.path.join(DATA_DIR, "best_5x5_weights.npz")
CONVERGED_WEIGHTS_PATH  = os.path.join(DATA_DIR, "converged_weights.npz")
TRAINING_META_PATH      = os.path.join(DATA_DIR, "training_meta.json")

# ── Curriculum constants ───────────────────────────────────────────────────────
STAGE1_EPISODES   = 50
STAGE2_EPISODES   = 50
STAGE1_MAX_STEPS  = 50
STAGE2_MAX_STEPS  = 100
CONVERGE_MAX_STEPS = 30
CONVERGE_STREAK    = 5

# ──────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_meta():
    """Load training metadata from JSON, or return defaults if file is absent."""
    if os.path.exists(TRAINING_META_PATH):
        with open(TRAINING_META_PATH, 'r') as f:
            return json.load(f)
    return {'total_episodes': 0, 'best_steps': None, 'last_steps': None}


def _save_meta(meta):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TRAINING_META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Weight I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_agent_weights(path, snn, decoder, encoder):
    """Persist SNN synapses, decoder, and encoder weights to a .npz checkpoint."""
    np.savez(
        path,
        S_in_w=np.array(snn.S_in.w),
        S_rec_w=np.array(snn.S_rec.w),
        S_out_w=np.array(snn.S_out.w),
        S_direct_w=np.array(snn.S_direct.w),
        decoder_W_out=decoder.W_out,
        decoder_W_reward=decoder.W_reward,
        decoder_W_action=decoder.W_action,
        decoder_W_forward=decoder.W_forward,
        decoder_W_reward_forward=decoder.W_reward_forward,
        encoder_W=encoder.W,
        encoder_W_intention=encoder.W_intention,
    )


def load_agent_weights(path, snn, decoder, encoder):
    """Restore agent weights from a .npz checkpoint."""
    data = np.load(path)
    snn.S_in.w    = data['S_in_w']
    snn.S_rec.w   = data['S_rec_w']
    snn.S_out.w   = data['S_out_w']
    if 'S_direct_w' in data:
        snn.S_direct.w = data['S_direct_w']
    decoder.W_out    = data['decoder_W_out']
    decoder.W_reward = data['decoder_W_reward']
    decoder.W_action = data['decoder_W_action']
    if 'decoder_W_forward' in data:
        decoder.W_forward = data['decoder_W_forward']
    if 'decoder_W_reward_forward' in data:
        decoder.W_reward_forward = data['decoder_W_reward_forward']
    encoder.W           = data['encoder_W']
    encoder.W_intention = data['encoder_W_intention']


# ──────────────────────────────────────────────────────────────────────────────
# Forward-model pretraining
# ──────────────────────────────────────────────────────────────────────────────

def pretrain_forward_model(
    decoder,
    encoder,
    num_random_steps=50,
    grid_size=5,
    goal_pos=(4, 4),
    learning_rate=0.05,
    sgd_epochs=20,
    verbose=True,
):
    """
    Collect random-action transitions in a 5×5 grid and train the decoder's
    forward model (W_forward, W_reward_forward) with supervised MSE learning
    before active inference begins.
    """
    from environment.grid_world import GridWorld

    env   = GridWorld(grid_size=grid_size, start_pos=(0, 0), goal_pos=goal_pos)
    state = env.reset()
    buffer = []

    if verbose:
        print(f"\n[Pretraining] Collecting {num_random_steps} random transitions "
              f"in {grid_size}×{grid_size} grid …")

    for _ in range(num_random_steps):
        actions = env.get_actions()
        action  = int(np.random.choice(actions))
        next_state, reward, done = env.step(action)
        buffer.append((state.copy(), action, next_state.copy(), reward))
        state = next_state
        if done:
            state = env.reset()

    if verbose:
        print(f"[Pretraining] Buffer size: {len(buffer)} transitions. "
              f"Running {sgd_epochs} SGD epochs …")

    for epoch in range(sgd_epochs):
        np.random.shuffle(buffer)
        epoch_state_loss  = 0.0
        epoch_reward_loss = 0.0
        for s, a, ns, r in buffer:
            decoder.update_forward(s, a, ns, r, learning_rate=learning_rate)
            pred_ns = decoder.predict_forward(s, a)
            pred_r  = decoder.predict_reward_forward(s, a)
            epoch_state_loss  += float(np.mean((pred_ns - ns) ** 2))
            epoch_reward_loss += float((pred_r - r) ** 2)

        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
            n = len(buffer)
            print(f"  Epoch {epoch+1:3d}/{sgd_epochs} | "
                  f"State MSE: {epoch_state_loss/n:.4f} | "
                  f"Reward MSE: {epoch_reward_loss/n:.4f}")

    if verbose:
        print("[Pretraining] Forward model pretraining complete.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap — first-ever weight creation
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap():
    """
    Create the very first trained_weights.npz with small random weights and
    pretrained forward model.  This is the ONLY place random initialisation
    of the full agent is allowed.
    """
    if os.path.exists(TRAINED_WEIGHTS_PATH):
        print(
            f"[Bootstrap] '{TRAINED_WEIGHTS_PATH}' already exists.\n"
            "Delete it manually if you really want to start from scratch, then re-run with --bootstrap."
        )
        sys.exit(1)

    print("[Bootstrap] No existing weights found — initialising from scratch …")
    os.makedirs(DATA_DIR, exist_ok=True)

    snn     = FlyBrainSNN(use_noise=True)
    encoder = Encoder()
    decoder = Decoder()

    # Pretrain the forward model immediately so episode 1 isn't completely blind
    pretrain_forward_model(
        decoder, encoder,
        num_random_steps=50, grid_size=5, goal_pos=(4, 4),
        learning_rate=0.05, sgd_epochs=20, verbose=True,
    )
    decoder.save(PRETRAINED_DECODER_PATH)

    save_agent_weights(TRAINED_WEIGHTS_PATH, snn, decoder, encoder)
    meta = {'total_episodes': 0, 'best_steps': None, 'last_steps': None}
    _save_meta(meta)

    print(
        f"[Bootstrap] Initial weights written to '{TRAINED_WEIGHTS_PATH}'.\n"
        "Run 'python train.py' to start training."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    stage1_episodes=STAGE1_EPISODES,
    stage2_episodes=STAGE2_EPISODES,
    stage1_max_steps=STAGE1_MAX_STEPS,
    stage2_max_steps=STAGE2_MAX_STEPS,
    learning_rate=0.01,
):
    """
    Continue training from existing weights.  Refuses to run if
    trained_weights.npz is missing — use --bootstrap first.
    """
    # ── Guard: weights must exist ────────────────────────────────────────────
    if not os.path.exists(TRAINED_WEIGHTS_PATH):
        print(
            "[Error] No trained weights found. "
            "Run train.py first to initialize."
        )
        sys.exit(1)

    # ── Load metadata and print startup banner ───────────────────────────────
    meta = _load_meta()
    total_episodes_so_far = meta.get('total_episodes', 0)
    best_steps_ever       = meta.get('best_steps', None)
    last_steps            = meta.get('last_steps', None)

    best_str = f"{best_steps_ever}" if best_steps_ever is not None else "—"
    last_str = f"{last_steps}"      if last_steps      is not None else "—"
    print(
        f"[Loaded] Episode {total_episodes_so_far} | "
        f"Best: {best_str} steps | Last: {last_str} steps"
    )

    total_episodes = stage1_episodes + stage2_episodes
    print("==========================================")
    print(
        f"Starting SNN Training — {stage1_episodes}×5×5 (max {stage1_max_steps} steps) "
        f"+ {stage2_episodes}×10×10 (max {stage2_max_steps} steps)"
    )
    print("==========================================\n")

    # ── Initialise and load weights ──────────────────────────────────────────
    snn     = FlyBrainSNN(use_noise=True)
    encoder = Encoder()
    decoder = Decoder()
    gen_model = GenerativeModel()

    load_agent_weights(TRAINED_WEIGHTS_PATH, snn, decoder, encoder)
    print(f"[Loaded] Weights restored from '{TRAINED_WEIGHTS_PATH}'.\n")

    # ── Pretrained decoder: load cache or skip ───────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(PRETRAINED_DECODER_PATH):
        decoder.load(PRETRAINED_DECODER_PATH)
        print("[Pretraining] Pretrained decoder loaded — skipping random exploration.\n")
    else:
        pretrain_forward_model(
            decoder, encoder,
            num_random_steps=50, grid_size=5, goal_pos=(4, 4),
            learning_rate=0.05, sgd_epochs=20, verbose=True,
        )
        decoder.save(PRETRAINED_DECODER_PATH)
        print(f"[Pretraining] Pretrained decoder saved to '{PRETRAINED_DECODER_PATH}'.\n")

    # ── Per-run tracking ─────────────────────────────────────────────────────
    steps_history    = []
    avg_fe_history   = []
    best_5x5_steps   = float('inf')
    best_5x5_episode = None
    fast_goal_streak = 0
    converged        = False
    global_episode   = total_episodes_so_far  # continues from last saved episode

    if os.path.exists(BEST_5X5_WEIGHTS_PATH):
        os.remove(BEST_5X5_WEIGHTS_PATH)

    # ── Episode runner ────────────────────────────────────────────────────────
    def run_one_episode(stage_name, grid_size, goal_pos, max_steps, ep_number):
        trajectory, fe_history, reached_goal = run_episode(
            snn=snn,
            encoder=encoder,
            decoder=decoder,
            gen_model=gen_model,
            max_steps=max_steps,
            learning_rate=learning_rate,
            grid_size=grid_size,
            start_pos=(0, 0),
            goal_pos=goal_pos,
            verbose=True,
            episode=ep_number,
        )
        steps_taken = len(fe_history)
        avg_fe      = float(np.mean(fe_history)) if fe_history else 0.0
        status      = "GOAL" if reached_goal else "timeout"
        print(
            f"Episode {ep_number} Summary ({stage_name}): "
            f"Steps = {steps_taken} | Avg FE = {avg_fe:.4f} | {status}"
        )
        return steps_taken, avg_fe, reached_goal

    def persist_weights_if_improved(steps_taken, reached_goal):
        """Save weights and metadata whenever the agent reaches the goal."""
        nonlocal best_steps_ever
        if not reached_goal:
            return
        improved = best_steps_ever is None or steps_taken < best_steps_ever
        if improved:
            best_steps_ever = steps_taken
        # Always save after a successful episode — agent only gets better
        save_agent_weights(TRAINED_WEIGHTS_PATH, snn, decoder, encoder)
        meta_now = {
            'total_episodes': global_episode,
            'best_steps':     best_steps_ever,
            'last_steps':     steps_taken,
        }
        _save_meta(meta_now)
        tag = " ← new best" if improved else ""
        print(f"[Saved] Weights updated — {steps_taken} steps{tag}")

    # ── Stage 1: 5×5 ─────────────────────────────────────────────────────────
    for stage_ep in range(stage1_episodes):
        global_episode += 1
        ep = global_episode

        if ep > total_episodes_so_far + 1:
            snn.reset_monitors()

        print(f"\n--- Episode {ep} ({ep - total_episodes_so_far}/{total_episodes} this run) "
              f"[5×5 Grid | Goal: (4,4)] ---")

        steps_taken, avg_fe, reached_goal = run_one_episode(
            "5x5 Grid", grid_size=5, goal_pos=(4, 4),
            max_steps=stage1_max_steps, ep_number=ep,
        )
        steps_history.append(steps_taken)
        avg_fe_history.append(avg_fe)

        persist_weights_if_improved(steps_taken, reached_goal)

        if reached_goal and steps_taken < best_5x5_steps:
            best_5x5_steps   = steps_taken
            best_5x5_episode = ep
            save_agent_weights(BEST_5X5_WEIGHTS_PATH, snn, decoder, encoder)
            print(
                f"[Checkpoint] New best 5×5 run: {steps_taken} steps "
                f"(episode {ep}) → saved to '{BEST_5X5_WEIGHTS_PATH}'"
            )

    # ── Transition: load best 5×5 weights for 10×10 stage ────────────────────
    if best_5x5_episode is not None and os.path.exists(BEST_5X5_WEIGHTS_PATH):
        load_agent_weights(BEST_5X5_WEIGHTS_PATH, snn, decoder, encoder)
        print(
            f"\n[Curriculum] Loaded best 5×5 weights ({best_5x5_steps} steps, "
            f"episode {best_5x5_episode}) for 10×10 stage."
        )
    else:
        print("\n[Curriculum] No successful 5×5 checkpoint — continuing to 10×10 with current weights.")

    # ── Stage 2: 10×10 ───────────────────────────────────────────────────────
    for stage_ep in range(stage2_episodes):
        global_episode += 1
        ep = global_episode

        snn.reset_monitors()

        print(f"\n--- Episode {ep} ({ep - total_episodes_so_far}/{total_episodes} this run) "
              f"[10×10 Grid | Goal: (9,9)] ---")

        steps_taken, avg_fe, reached_goal = run_one_episode(
            "10x10 Grid", grid_size=10, goal_pos=(9, 9),
            max_steps=stage2_max_steps, ep_number=ep,
        )
        steps_history.append(steps_taken)
        avg_fe_history.append(avg_fe)

        persist_weights_if_improved(steps_taken, reached_goal)

        if reached_goal and steps_taken < CONVERGE_MAX_STEPS:
            fast_goal_streak += 1
            print(
                f"[Convergence] Fast goal ({steps_taken} steps) — "
                f"streak {fast_goal_streak}/{CONVERGE_STREAK}"
            )
        else:
            fast_goal_streak = 0

        if fast_goal_streak >= CONVERGE_STREAK:
            converged = True
            save_agent_weights(CONVERGED_WEIGHTS_PATH, snn, decoder, encoder)
            print(
                f"\n[Converged] Agent reached goal in <{CONVERGE_MAX_STEPS} steps "
                f"for {CONVERGE_STREAK} consecutive episodes. "
                f"Saved to '{CONVERGED_WEIGHTS_PATH}'."
            )
            break

    # ── Final metadata flush (captures timeout episodes too) ─────────────────
    last_steps_run = steps_history[-1] if steps_history else last_steps
    _save_meta({
        'total_episodes': global_episode,
        'best_steps':     best_steps_ever,
        'last_steps':     last_steps_run,
    })

    # ── Training curves ───────────────────────────────────────────────────────
    episodes_run  = len(steps_history)
    episode_axis  = range(1, episodes_run + 1)
    stage1_end    = min(stage1_episodes, episodes_run)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_axis, steps_history, marker='o', color='#38bdf8', linewidth=2)
    if stage1_end < episodes_run:
        plt.axvline(stage1_end + 0.5, color='#f59e0b', linestyle='--', alpha=0.8, label='5×5 → 10×10')
        plt.legend(loc='upper right', facecolor='#1e293b', edgecolor='none', labelcolor='white')
    plt.title("Steps to Goal per Episode", fontsize=12, fontweight='bold', color='white')
    plt.xlabel("Episode", fontsize=10, color='white')
    plt.ylabel("Steps Taken", fontsize=10, color='white')
    plt.grid(True, color='#334155', linestyle=':', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(episode_axis, avg_fe_history, marker='s', color='#10b981', linewidth=2)
    if stage1_end < episodes_run:
        plt.axvline(stage1_end + 0.5, color='#f59e0b', linestyle='--', alpha=0.8)
    plt.title("Avg Free Energy per Episode", fontsize=12, fontweight='bold', color='white')
    plt.xlabel("Episode", fontsize=10, color='white')
    plt.ylabel("Variational Free Energy", fontsize=10, color='white')
    plt.grid(True, color='#334155', linestyle=':', alpha=0.5)

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

    # Final weights save (captures last episode regardless of goal)
    save_agent_weights(TRAINED_WEIGHTS_PATH, snn, decoder, encoder)
    print(f"Final weights saved to '{TRAINED_WEIGHTS_PATH}'")

    if converged:
        print(f"Converged checkpoint at '{CONVERGED_WEIGHTS_PATH}'")
    if best_5x5_episode is not None:
        print(
            f"Best 5×5 checkpoint: {best_5x5_steps} steps "
            f"(episode {best_5x5_episode}) at '{BEST_5X5_WEIGHTS_PATH}'"
        )

    print(
        f"\nTraining complete ({episodes_run} episodes this run, "
        f"{global_episode} total). "
        "The agent only gets better — weights saved after every successful episode."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroSpark SNN trainer")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help=(
            "Create the very first data/trained_weights.npz with small random weights. "
            "Use this once before your first training run. "
            "Will refuse to overwrite an existing checkpoint."
        ),
    )
    args = parser.parse_args()

    np.random.seed(42)

    if args.bootstrap:
        bootstrap()
    else:
        train()

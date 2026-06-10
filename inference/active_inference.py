from brian2 import ms, nA

SIMULATION_MS = 100

ACTION_NAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}

# Epsilon-greedy exploration rate — 20% random actions breaks oscillation loops
EPSILON = 0.20


def select_action(
    snn,
    current_state,
    possible_actions,
    encoder,
    decoder,
    generative_model,
    beta=15.0,
    reward_weight=3.0,
    verbose=True,
    epsilon=EPSILON,
):
    """
    Select action via Expected Free Energy (forward model + SNN imagination).

    Decision is driven purely by −EFE (policy logits are recorded for
    diagnostics but do **not** influence the score — they dominated 100:1
    before training and caused the agent to ignore EFE entirely).

    Epsilon-greedy exploration: with probability *epsilon* a uniformly
    random valid action is chosen instead of the EFE-optimal one.

    Returns (action_index, plan_info dict).
    """
    import numpy as np
    from inference.free_energy import compute_expected_free_energy

    preferred_state = generative_model.get_preferred_state()
    preferred_reward = generative_model.get_preferred_reward()

    snn.net.store('before_imagination')

    has_noise = hasattr(snn, 'noise') and snn.noise is not None
    if has_noise:
        snn.noise.active = False

    efe_list = []
    policy_logits = []

    for action in possible_actions:
        snn.net.restore('before_imagination')
        snn.sensory.I_inj = 0 * nA

        intention = encoder.encode_intention(current_state, action)
        snn.inject_sensory(intention)

        t_start = snn.net.t
        snn.run(SIMULATION_MS * ms)
        motor_spikes = snn.get_motor_spikes_in_window(t_start, snn.net.t)

        predicted_next, predicted_reward = decoder.predict_for_planning(
            current_state, action, motor_spikes
        )

        efe = compute_expected_free_energy(
            predicted_next,
            preferred_state,
            predicted_reward,
            preferred_reward,
            reward_weight=reward_weight,
        )
        efe_list.append(efe)
        # Policy logits are still decoded for diagnostics / future use,
        # but are NOT added to the score.
        policy_logits.append(float(decoder.decode_action(motor_spikes)[action]))

    snn.net.restore('before_imagination')
    if has_noise:
        snn.noise.active = True
    snn.sensory.I_inj = 0 * nA

    efe_arr = np.array(efe_list)
    policy_arr = np.array(policy_logits)

    # Score is purely −EFE (lower EFE = better action)
    scores = -efe_arr

    efe_spread = float(np.max(efe_arr) - np.min(efe_arr))

    # ── Epsilon-greedy exploration ───────────────────────────────────────
    is_random = False
    if np.random.rand() < epsilon:
        selected_action = int(np.random.choice(possible_actions))
        is_random = True
    elif efe_spread < 0.02:
        # EFE is flat — pick uniformly at random (no signal to exploit)
        selected_action = int(np.random.choice(possible_actions))
        is_random = True
    else:
        # Softmax over −EFE with temperature 1/beta
        shifted = scores - np.max(scores)
        exp_scores = np.exp(beta * shifted)
        probs = exp_scores / np.sum(exp_scores)
        selected_action = int(np.random.choice(possible_actions, p=probs))

    plan_info = {
        'actions': [int(a) for a in possible_actions],
        'action_names': [ACTION_NAMES[a] for a in possible_actions],
        'efe': [float(v) for v in efe_arr],
        'scores': [float(v) for v in scores],
        'policy_logits': [float(v) for v in policy_arr],
        'random': is_random,
        'efe_spread': efe_spread,
        'selected': selected_action,
        'selected_name': ACTION_NAMES[selected_action],
    }

    if verbose:
        efe_dict = dict(zip(plan_info['action_names'], [round(v, 4) for v in plan_info['efe']]))
        scores_dict = dict(zip(plan_info['action_names'], [round(v, 4) for v in plan_info['scores']]))
        tag = " (ε-random)" if is_random else ""
        print(
            f"  [Plan] EFE: {list(efe_dict.values())} | "
            f"Score(−EFE): {list(scores_dict.values())} | "
            f"Pick: {ACTION_NAMES[selected_action]}{tag}"
        )

    return selected_action, plan_info

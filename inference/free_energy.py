import numpy as np

def compute_prediction_error(predicted, actual):
    """
    Computes the prediction error (Mean Squared Error) between predicted and actual states.
    """
    return np.mean((predicted - actual) ** 2)

def compute_free_energy(predicted_state, actual_state, predicted_reward, actual_reward):
    """
    Computes the Variational Free Energy (VFE) which represents the mismatch
    between the SNN's predictions and the actual state transition that occurred.
    VFE = state_prediction_error + reward_prediction_error
    """
    state_error = compute_prediction_error(predicted_state, actual_state)
    reward_error = (predicted_reward - actual_reward) ** 2
    return state_error + reward_error

def compute_expected_free_energy(predicted_state, preferred_state, predicted_reward, preferred_reward, reward_weight=20.0):
    """
    Computes the Expected Free Energy (EFE) for an imagined action outcome.
    EFE measures how surprising the imagined state is compared to the agent's prior preferences.
    EFE = pragmatic_divergence (distance from goal) + reward_weight * reward_divergence
    """
    pragmatic_error = compute_prediction_error(predicted_state, preferred_state)
    reward_error = (predicted_reward - preferred_reward) ** 2
    return pragmatic_error + reward_weight * reward_error

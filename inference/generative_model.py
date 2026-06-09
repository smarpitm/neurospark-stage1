import numpy as np

class GenerativeModel:
    def __init__(self, preferred_state=None, preferred_reward=1.0):
        """
        Maintains the agent's generative expectations and prior preferences (goals).
        """
        # Default preferred observation: at the goal, no collisions.
        # 9 local patch values = 0.0, 2 relative goal direction values = [0.0, 0.0]
        if preferred_state is None:
            self.preferred_state = np.zeros(11, dtype=np.float32)
        else:
            self.preferred_state = np.array(preferred_state, dtype=np.float32)
            
        self.preferred_reward = preferred_reward

    def get_preferred_state(self):
        return self.preferred_state

    def get_preferred_reward(self):
        return self.preferred_reward

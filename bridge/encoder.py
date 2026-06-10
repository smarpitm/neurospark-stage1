import numpy as np
from brian2 import nA

class Encoder:
    def __init__(self, input_dim=11, sensory_neurons=100):
        """
        Initializes the encoder with random weights.
        :param input_dim: Dimensions of the input state (default 11)
        :param sensory_neurons: Number of sensory neurons (default 100)
        """
        self.input_dim = input_dim
        self.sensory_neurons = sensory_neurons
        # Initialize weights
        self.W = np.random.randn(sensory_neurons, input_dim) * 0.3
        self.W_intention = np.random.randn(sensory_neurons, input_dim + 4) * 0.3

    def encode(self, state):
        """
        State vector -> current injection pattern
        :param state: 11-dimensional observation state vector
        :return: Current injection pattern (100-dimensional Brian2 Quantity in nA)
        """
        return np.dot(self.W, state) * 1.5 * nA

    def encode_intention(self, state, action):
        """
        State + intended action -> sensory input
        :param state: 11-dimensional observation state vector
        :param action: Action index (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT)
        :return: Current injection pattern (100-dimensional Brian2 Quantity in nA)
        """
        intention_vector = np.zeros(self.input_dim + 4)
        intention_vector[:self.input_dim] = state
        intention_vector[self.input_dim + action] = 1.0
        return np.dot(self.W_intention, intention_vector) * 1.5 * nA

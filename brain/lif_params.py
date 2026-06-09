from brian2 import ms, mV, pF, Mohm

# Leaky Integrate-and-Fire (LIF) parameters
tau = 20 * ms
v_rest = -70 * mV
v_threshold = -50 * mV
v_reset = -70 * mV
refractory_period = 5 * ms

# Capacitance and resistance
C = 250 * pF
R = 80 * Mohm  # R * C = 20 ms

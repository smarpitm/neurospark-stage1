


import os
import numpy as np

def generate_watts_strogatz(n, k, p):
    """
    Generates a Watts-Strogatz small-world network.
    :param n: Number of nodes
    :param k: Average degree (must be even)
    :param p: Rewiring probability
    :return: (sources, targets) arrays of connections
    """
    assert k % 2 == 0, "Average degree k must be even"
    
    # Initialize adjacency sets
    adj = {i: set() for i in range(n)}
    
    # 1. Ring lattice connection
    for i in range(n):
        for dist in range(1, k // 2 + 1):
            neighbor = (i + dist) % n
            adj[i].add(neighbor)
            adj[neighbor].add(i)
            
    # 2. Rewire edges with probability p
    for i in range(n):
        for dist in range(1, k // 2 + 1):
            old_neighbor = (i + dist) % n
            if np.random.rand() < p:
                # Find valid new neighbors (no self-loops, no existing connections)
                possible_targets = set(range(n)) - {i} - adj[i]
                if possible_targets:
                    new_neighbor = np.random.choice(list(possible_targets))
                    
                    # Remove old undirected edge
                    if old_neighbor in adj[i]:
                        adj[i].remove(old_neighbor)
                    if i in adj[old_neighbor]:
                        adj[old_neighbor].remove(i)
                        
                    # Add new undirected edge
                    adj[i].add(new_neighbor)
                    adj[new_neighbor].add(i)
                    
    # 3. Build directed source/target arrays
    sources = []
    targets = []
    for u in range(n):
        for v in adj[u]:
            sources.append(u)
            targets.append(v)
            
    return np.array(sources), np.array(targets)

def load_or_generate_connectome(data_dir="data", filename="flywire_subset_1k.npz"):
    """
    Loads the connectome file if it exists, otherwise generates a synthetic small-world
    recurrent connectome (800 nodes, average degree 80, rewiring probability 0.1) and caches it.
    """
    # Go up one level if we are inside the brain folder to place data in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(project_root, data_dir)
    filepath = os.path.join(target_dir, filename)
    
    if os.path.exists(filepath):
        data = np.load(filepath)
        return data['sources'], data['targets'], data['weights']
        
    # Generate synthetic small-world connectivity for 800 recurrent neurons
    os.makedirs(target_dir, exist_ok=True)
    n_recurrent = 800
    k_degree = 80
    p_rewire = 0.1
    
    # Set seed for reproducibility of the connectome initialization
    np.random.seed(42)
    sources, targets = generate_watts_strogatz(n_recurrent, k_degree, p_rewire)
    
    # Draw synaptic weights from log-normal distribution (biologically realistic)
    n_edges = len(sources)
    raw_weights = np.random.lognormal(mean=0.0, sigma=0.5, size=n_edges)
    
    # Scale weights so they represent reasonable synaptic conductances (around 0.02 to 0.5 mV change post-synaptically)
    # The average sum of input weights per neuron is normalized to ~5.0 (dimensionless weight) to prevent runaway recurrent excitation
    weights = raw_weights * (5.0 / k_degree)
    
    np.savez(filepath, sources=sources, targets=targets, weights=weights)
    print(f"Generated synthetic small-world connectome and cached to: {filepath}")
    return sources, targets, weights

if __name__ == "__main__":
    srcs, tgts, wts = load_or_generate_connectome()
    print(f"Loaded connectome. Neurons: 800, Connections: {len(srcs)}, Avg weight: {wts.mean():.4f}")

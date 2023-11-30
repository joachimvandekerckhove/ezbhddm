import numpy as np
import importlib
import simulation
import concurrent.futures

P = np.array([8,16,32,64])
T = np.array([8,16,32,64])

s = np.empty((len(P), len(T)), dtype=object)

# Initialize Hddm_Design objects
for r, p in enumerate(P):
    for c, t in enumerate(T):
        s[r, c] = simulation.Hddm_Design(p, t, np.arange(0, p) % 2, 'drift')

# Function to run the simulation and return the updated object
def run_simulation(simulation_object):
    simulation_object.run(1000)
    return simulation_object

# Distribute tasks across processors
with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
    # Create a list of futures for all simulation tasks
    futures = [executor.submit(run_simulation, s[r, c]) for r in range(len(P)) for c in range(len(T))]

    # Update s with the results from the futures
    for future, (r, c) in zip(futures, [(r, c) for r in range(len(P)) for c in range(len(T))]):
        s[r, c] = future.result()

# s should now be updated


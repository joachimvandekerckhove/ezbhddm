# Runs the extensive EZBHDDM simulation studies

#   Takes a while to run.  Check out main.py first.
#   There are much better ways to parallelize this if needed.

import numpy as np
import importlib
import simulation
import concurrent.futures
import pickle
import os, datetime


P = np.array([20, 40, 80, 160, 320])
T = np.array([20, 40, 80, 160, 320])


# Run ttest design

for criterion in ['drift', 'nondt', 'bound']:

    print(f"[{datetime.datetime.now().isoformat()}]  Design: ttest     Criterion: {criterion}")

    s = np.empty((len(P), len(T)), dtype=object)

    # Initialize Hddm_Design objects
    for r, p in enumerate(P):
        for c, t in enumerate(T):
            s[r, c] = simulation.Hddm_Design(p, t, np.arange(0, p) % 2, criterion)

    # Function to save the matrix to disk
    def save_to_disk(matrix, filename=f'../cache/{criterion}_ttest.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(matrix, f)

    # Function to run the simulation and return the updated object
    def run_simulation(simulation_object):
        try:
            simulation_object.run(1000)
        except Exception as exc:
            print(f'Simulation run encountered an exception: {exc}')
        return simulation_object

    # Distribute tasks across processors
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(run_simulation, s[r, c]) for r in range(len(P)) for c in range(len(T))]

        for i, (future, (r, c)) in enumerate(zip(futures, [(r, c) for r in range(len(P)) for c in range(len(T))])):
            try:
                s[r, c] = future.result()
            except Exception as exc:
                print(f'Worker thread for s[{r},{c}] encountered an exception: {exc}')

    # Final save at the end
    save_to_disk(s)


# Run linreg design

for criterion in ['nondt', 'drift', 'bound']:

    print(f"[{datetime.datetime.now().isoformat()}]  Design: linreg     Criterion: {criterion}")

    s = np.empty((len(P), len(T)), dtype=object)

    # Initialize Hddm_Design objects
    for r, p in enumerate(P):
        for c, t in enumerate(T):
            s[r, c] = simulation.Hddm_Design(p, t, np.arange(0, p)/p, criterion)

    # Function to save the matrix to disk
    def save_to_disk(matrix, filename=f'../cache/{criterion}_linreg.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(matrix, f)

    # Function to run the simulation and return the updated object
    def run_simulation(simulation_object):
        try:
            simulation_object.run(1000)
        except Exception as exc:
            print(f'Simulation run encountered an exception: {exc}')
        return simulation_object

    # Distribute tasks across processors (note each run takes 4 threads)
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(run_simulation, s[r, c]) for r in range(len(P)) for c in range(len(T))]

        for i, (future, (r, c)) in enumerate(zip(futures, [(r, c) for r in range(len(P)) for c in range(len(T))])):
            try:
                s[r, c] = future.result()
            except Exception as exc:
                print(f'Worker thread for s[{r},{c}] encountered an exception: {exc}')

    # Final save at the end
    save_to_disk(s)


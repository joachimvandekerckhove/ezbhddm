# Runs a small part of the EZBHDDM simulations

import numpy as np
import pandas as pd
import pickle

import data_set
import parameter_set
import prior
import simulation
import ezhbddm

# Set the number of participants and trials per participants
P = np.array([20, 40])
T = np.array([20, 40])

s = np.empty((len(P), len(T)), dtype=object)

# Select the criterion from ['drift', 'bound', 'nondt']
criterion = 'drift'

# Select the design from ['ttest', 'linreg']
design = 'ttest'

# Set up the simulation
for r, p in enumerate(P):
    for c, t in enumerate(T):
        if design == 'ttest':
            s[r, c] = simulation.Hddm_Design(p, t, np.arange(0, p) % 2, criterion)
        if design == 'linreg':
            s[r, c] = simulation.Hddm_Design(p, t, np.arange(0, p) % 2, criterion)

# Run the simulations
for r, p in enumerate(P):
    for c, t in enumerate(T):
        s[r, c].run(100)  # Only 100 runs

# Save the result
with open(f'../cache/{criterion}_{design}_small.pkl', 'wb') as f:
    pickle.dump(s, f)

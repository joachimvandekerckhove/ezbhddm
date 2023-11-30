import importlib

import numpy as np
import pandas as pd
import pickle

import data_set
import parameter_set
import prior
import simulation
import ezhbddm

data_set      = importlib.reload(data_set)
parameter_set = importlib.reload(parameter_set)
prior         = importlib.reload(prior)
simulation    = importlib.reload(simulation)
ezhbddm       = importlib.reload(ezhbddm)

P = np.array([10, 20, 40, 80])
T = np.array([10, 20, 40, 80])

s = np.empty((len(P), len(T)), dtype=object)

for r, p in enumerate(P):
    for c, t in enumerate(T):
        s[r, c] = simulation.Hddm_Design(p, t, np.arange(0, p) % 2, 'drift')


for x in range(20):
    for c, t in enumerate(T):
        for r, p in enumerate(P):
            #print(f"{(x+1):>2}: Sim(P={p}, T={t})")
            s[r, c].run(100).report('short')
            with open('ezbhddm.pkl', 'wb') as f:
                pickle.dump(s, f)

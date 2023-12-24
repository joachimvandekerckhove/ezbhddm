# Functions to create figures for the EZBHDDM simulation studies

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

os.chdir('/srv/host/src/')

import simulation

def biasplot(s, parameter, i, j, ax):
    x = [getattr(s[i,j].results[k][0], parameter)
         for k in range(len(s[i,j].results))
         if (s[i,j].results[k][0] - s[i,j].results[k][1]) is not None]
    y = [getattr(s[i,j].results[k][1], parameter)
         for k in range(len(s[i,j].results))
         if (s[i,j].results[k][0] - s[i,j].results[k][1]) is not None]

    ax.scatter(x, y, s=4)

    #ax.set_xlabel('simulated value')
    if j==0:
        ax.set_ylabel(f'P={s[i,j].n_Participants}', fontsize=10)
    if i==0:
        ax.set_title(f'T={s[i,j].n_TrialsPerPerson}', fontsize=10)
    if j!=(s.shape[1]-1):
        ax.set_yticklabels([])
    if i!=(s.shape[0]-1):
        ax.set_xticklabels([])
    ax.yaxis.tick_right()

    ax.axis('square')

    if parameter == "betaweight":
        mn, mx = 0, 1
    if parameter == "bound_mean":
        mn, mx = 1, 2
    if parameter == "drift_mean":
        mn, mx = -2, 2
    if parameter == "nondt_mean":
        mn, mx = .15, .45

    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)

    ax.plot([mn, mx], [mn, mx], color='k', linestyle='--')
    ax.grid()

def rmseplot(s, parameter, ax, title, axlbl):
    v = np.zeros(s.shape)
    xticklabels = [s[0,i].n_TrialsPerPerson for i in range(s.shape[1])]
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            q = [getattr(s[i,j].results[k][1], parameter) - getattr(s[i,j].results[k][0], parameter)
                 for k in range(len(s[i,j].results))
                 if (s[i,j].results[k][0] - s[i,j].results[k][1]) is not None]
            v[i,j] = np.sqrt(np.mean(np.array(q)**2))

    for i in range(v.shape[0]):
        ax.plot(xticklabels, v[i, :], label=f'P = {s[i,0].n_Participants}', marker='o', linestyle='--')

    ax.grid()
    ax.set_xlabel('Trials per participant')
    if not axlbl:
        ax.set_ylabel('RMSE')
    else:
        ax.set_yticklabels([])
    ax.set_title(f'Error: {title}')
    ax.set_xscale('log')
    ax.set_ylim((0,np.maximum(0.25, np.max(v))))
    ax.set_xticks(xticklabels)
    ax.set_xticklabels(xticklabels)


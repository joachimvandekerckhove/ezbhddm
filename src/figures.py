# This script creates production figures for the EZBHDDM simulation studies

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

os.chdir('/srv/host/src/')

import simulation
from plots import *

parameter = "betaweight"

for design in ['ttest','linreg']:
    for criterion in ['nondt','drift','bound']:

        print(f'{criterion}_{design}.pkl ...')

        s = pickle.load(open(f'../cache/{criterion}_{design}.pkl', 'rb'))

        # Bias plots
        for i, par in enumerate([("betaweight",r"$\beta$"),
                                 ("bound_mean",r"$\mu_\alpha$"),
                                 ("drift_mean",r"$\mu_\nu$"),
                                 ("nondt_mean",r"$\mu_\tau$")]):
            fig, axes = plt.subplots(s.shape[0], s.shape[1], figsize=(5,5))

            for i in range(s.shape[0]):
                for j in range(s.shape[1]):
                    print(f'Bias plot: {par[0]}[{i},{j}]')
                    biasplot(s, par[0], i, j, axes[i,j])

            plt.savefig(f'../figures/{design}_{criterion}_{par[0]}.eps', format='eps', bbox_inches='tight', transparent=True)
            plt.close()

        # RMSE calibration plots
        
        fig, axes = plt.subplots(2, 2, figsize=(5, 5))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

        for i, par in enumerate([("betaweight",r"$\beta$"),
                                 ("bound_mean",r"$\mu_\alpha$"),
                                 ("drift_mean",r"$\mu_\nu$"),
                                 ("nondt_mean",r"$\mu_\tau$")]):
            print(f'RMSE plot {par}')
            rmseplot(s, par[0], axes[i // 2, i % 2], title=par[1], axlbl=i%2)

        axes[1,1].legend(loc="upper right", prop={'size': 8})

        plt.savefig(f'../figures/{design}_{criterion}_rmse.eps', format='eps', bbox_inches='tight', transparent=True)
        plt.close()

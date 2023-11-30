# Classes to do simulations

import numpy as np
import pandas as pd
import pyjags
import copy
import sys
import time
import random

import parameter_set
import prior
import data_set
import ezhbddm

class Hddm_Design:
    def __init__(self, participants, trials, predictor, criterion = None, prior = prior.Hddm_Prior()):
        self.n_Participants    = int(participants)
        self.n_TrialsPerPerson = int(trials)
        self.prior             = prior
        self.parameter_set     = None
        self.data              = None
        self.estimate          = None
        self.predictor         = predictor
        self.criterion         = criterion
        self.results           = []
        self.statistics        = {}
        self.walltime          = []

    def run(self, iterations = 1, showProgress = True):
        start = len(self.results) + 1
        stop  = start + iterations - 1
        for i in range(iterations):
            self.sample_parameters()
            self.sample_data()
            start_time = time.time()
            self.estimate_parameters(silent = True)
            self.walltime.append(time.time() - start_time)
            self.results.append((copy.deepcopy(self.parameter_set), copy.deepcopy(self.estimate)))
            if showProgress:
                percent = ((start+i) / stop) * 100
                cplt = int(np.fix(percent/2))
                sys.stdout.write(f"\rProgress [{'='*cplt}{' '*(50-cplt)}] {percent:6.2f}%")
                sys.stdout.flush()
        sys.stdout.write(f"\n")
        sys.stdout.flush()
        self.compute_statistics()
        return self

    def compute_statistics(self):
        error = [(a-b).betaweight for a, b in self.results if a is not None and b is not None]
        self.statistics['mean_error'] = np.mean(error)
        self.statistics['mse']        = np.mean(np.array(error)**2)
        self.statistics['rmse']       = np.sqrt(self.statistics['mse'])
        self.statistics['mae']        = np.mean(np.abs(np.array(error)))
        return

    def report(self, style = 'long'):
        me    = self.statistics['mean_error']
        mse   = self.statistics['mse']
        rmse  = self.statistics['rmse']
        mae   = self.statistics['mae']

        if style == 'short':
            print(f"Sim(P={self.n_Participants}, T={self.n_TrialsPerPerson}, {np.median(self.walltime):.2f}s): ME = {me:.6f}, MAE = {mae:.6f}")
        if style == 'long':
            print("  Simulation results")
            print("  ------------------")
            print(f"  Participants{self.n_Participants:>6}")
            print(f"  Trials      {self.n_TrialsPerPerson:>6}")
            print(f"  Criterion   '{self.criterion}'")
            print(f"  Walltime    {np.median(self.walltime):6.2f}s\n")

            print(f"  ME   = {me:.6f}")
            print(f"  MSE  = {mse:.6f}")
            print(f"  RMSE = {rmse:.6f}")
            print(f"  MAE  = {mae:.6f}")

        return

    def sample_parameters(self):
        self.parameter_set = parameter_set.Hddm_Parameter_Set.random(self.prior, self.n_Participants, self.criterion, self.predictor)
        return

    def sample_data(self):
        if not self.parameter_set:
            raise ValueError("You must set or draw a parameter set before sampling data.")
        self.data = data_set.Hddm_Data().sample(self)
        return

    def estimate_parameters(self, silent = False):
        self.estimate = ezhbddm.estimate(self.data, self.prior, self.criterion, silent)
        return

    def __str__(self):
        output = [
            "Hddm_Design Parameters:",
            f"Number of Participants: {self.n_Participants}",
            f"Trials Per Person:      {self.n_TrialsPerPerson}",
            f"Prior:                  {self.prior}",
            f"Parameter Set:          {self.parameter_set}",
            f"Data:                   {self.data}",
            f"Criterion:              {self.criterion}"
        ]
        return '\n'.join(output)


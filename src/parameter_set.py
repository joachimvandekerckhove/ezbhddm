# Classes to do simulations

import numpy as np
import pandas as pd
import pyjags
import copy

class Hddm_Parameter_Set:
    def __init__(self,
                 bound_mean = None, bound_sdev = None, bound = None,
                 drift_mean = None, drift_sdev = None, drift = None,
                 nondt_mean = None, nondt_sdev = None, nondt = None,
                 betaweight = None):
        self.betaweight = betaweight

        self.bound_mean = bound_mean
        self.bound_sdev = bound_sdev
        self.drift_mean = drift_mean
        self.drift_sdev = drift_sdev
        self.nondt_mean = nondt_mean
        self.nondt_sdev = nondt_sdev

        self.nondt      = nondt
        self.bound      = bound
        self.drift      = drift

    @staticmethod
    def random(prior, n_Participants = None, criterion = None, predictor = 0):
        parameter_set = Hddm_Parameter_Set()
        parameter_set.betaweight = np.random.normal (prior.betaweight_mean , prior.betaweight_sdev)
        parameter_set.bound_mean = np.random.normal (prior.bound_mean_mean , prior.bound_mean_sdev)
        parameter_set.drift_mean = np.random.normal (prior.drift_mean_mean , prior.drift_mean_sdev)
        parameter_set.nondt_mean = np.random.normal (prior.nondt_mean_mean , prior.nondt_mean_sdev)
        parameter_set.bound_sdev = np.random.uniform(prior.bound_sdev_lower, prior.bound_sdev_upper)
        parameter_set.drift_sdev = np.random.uniform(prior.drift_sdev_lower, prior.drift_sdev_upper)
        parameter_set.nondt_sdev = np.random.uniform(prior.nondt_sdev_lower, prior.nondt_sdev_upper)

        Xb = predictor if criterion == 'bound' else 0
        Xd = predictor if criterion == 'drift' else 0
        Xn = predictor if criterion == 'nondt' else 0

        parameter_set.bound = np.random.normal(parameter_set.bound_mean + (parameter_set.betaweight * Xb),
                                               parameter_set.bound_sdev,
                                               n_Participants)
        parameter_set.drift = np.random.normal(parameter_set.drift_mean + (parameter_set.betaweight * Xd),
                                               parameter_set.drift_sdev,
                                               n_Participants)
        parameter_set.nondt = np.random.normal(parameter_set.nondt_mean + (parameter_set.betaweight * Xn),
                                               parameter_set.nondt_sdev,
                                               n_Participants)
        return parameter_set

    def __sub__(self, other):
        if other is None:
            return None

        if not isinstance(other, Hddm_Parameter_Set):
            raise ValueError("Can only take a difference between two Hddm_Parameter_Set objects.")

        return Hddm_Parameter_Set(
            betaweight = self.betaweight - other.betaweight,
            bound_mean = self.bound_mean - other.bound_mean,
            bound_sdev = self.bound_sdev - other.bound_sdev,
            drift_mean = self.drift_mean - other.drift_mean,
            drift_sdev = self.drift_sdev - other.drift_sdev,
            nondt_mean = self.nondt_mean - other.nondt_mean,
            nondt_sdev = self.nondt_sdev - other.nondt_sdev,
            bound      = self.bound - other.bound if self.bound is not None and other.bound is not None else None,
            drift      = self.drift - other.drift if self.drift is not None and other.drift is not None else None,
            nondt      = self.nondt - other.nondt if self.nondt is not None and other.nondt is not None else None
        )

    def __eq__(self, other):
        if not isinstance(other, Hddm_Parameter_Set):
            return False

        return (
            self.betaweight == other.betaweight and
            self.bound_mean == other.bound_mean and
            self.bound_sdev == other.bound_sdev and
            self.drift_mean == other.drift_mean and
            self.drift_sdev == other.drift_sdev and
            self.nondt_mean == other.nondt_mean and
            self.nondt_sdev == other.nondt_sdev and
            np.array_equal(self.bound, other.bound) and
            np.array_equal(self.drift, other.drift) and
            np.array_equal(self.nondt, other.nondt)
        )

    def __str__(self):
        output = [
            "Hddm_Parameter_Set Details:",
            f"Betaweight:              {self.betaweight}",
            f"Bound Mean:              {self.bound_mean}",
            f"Bound Std Dev:           {self.bound_sdev}",
            f"Drift Mean:              {self.drift_mean}",
            f"Drift Std Dev:           {self.drift_sdev}",
            f"Non-decision Time Mean:  {self.nondt_mean}",
            f"Non-decision Time Std:   {self.nondt_sdev}",
            f"Bound:                   {self.bound}",
            f"Drift:                   {self.drift}",
            f"Non-decision Time:       {self.nondt}"
        ]
        return '\n'.join(output)


# Classes to do simulations

import numpy as np
import pandas as pd
import pyjags
import copy

class Hddm_Prior:
    def __init__(self):
        self.betaweight_mean = 0.00
        self.betaweight_sdev = 0.50
        self.bound_mean_mean = 1.50
        self.bound_mean_sdev = 0.20
        self.drift_mean_mean = 0.00
        self.drift_mean_sdev = 0.50
        self.nondt_mean_mean = 0.30
        self.nondt_mean_sdev = 0.06
        self.bound_sdev_lower = 0.10
        self.bound_sdev_upper = 0.40
        self.drift_sdev_lower = 0.20
        self.drift_sdev_upper = 0.40
        self.nondt_sdev_lower = 0.05
        self.nondt_sdev_upper = 0.25

    def __str__(self):
        output = [
            "Hddm_Prior Details:",
            f"Betaweight Mean:             {self.betaweight_mean}",
            f"Betaweight Std Dev:          {self.betaweight_sdev}",
            f"Bound Mean Mean:             {self.bound_mean_mean}",
            f"Bound Mean Std Dev:          {self.bound_mean_sdev}",
            f"Drift Mean Mean:             {self.drift_mean_mean}",
            f"Drift Mean Std Dev:          {self.drift_mean_sdev}",
            f"Non-decision Time Mean Mean: {self.nondt_mean_mean}",
            f"Non-decision Time Mean Std:  {self.nondt_mean_sdev}",
            f"Bound Std Dev Shape:         {self.bound_sdev_lower}",
            f"Bound Std Dev Scale:         {self.bound_sdev_upper}",
            f"Drift Std Dev Shape:         {self.drift_sdev_lower}",
            f"Drift Std Dev Scale:         {self.drift_sdev_upper}",
            f"Non-decision Time Shape:     {self.nondt_sdev_lower}",
            f"Non-decision Time Scale:     {self.nondt_sdev_upper}"
        ]
        return '\n'.join(output)

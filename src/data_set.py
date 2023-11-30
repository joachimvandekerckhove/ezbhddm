# Classes to do simulations

import numpy as np
import pandas as pd
import pyjags
import copy

from wdm import wdmrnd

class Hddm_Data():
    def __init__(self, person = None, rt = None, accuracy = None, n_TrialsPerPerson = None, X = None):
        self.person            = person
        self.rt                = rt
        self.accuracy          = accuracy
        self.n_TrialsPerPerson = n_TrialsPerPerson
        self.X                 = X

    @staticmethod
    def read_rr_data():
        url = "https://osf.io/download/28ahk/"

    @staticmethod
    def sample(design):
        T = design.n_TrialsPerPerson
        P = design.n_Participants
        parameters = design.parameter_set

        person_list   = []
        rt_list       = []
        accuracy_list = []

        for p in range(P):
            accuracy = 0
            while np.sum(accuracy) == 0:
                rt, accuracy = wdmrnd(parameters.bound[p], parameters.drift[p], parameters.nondt[p], T)
            person_list.extend([p] * T)
            rt_list.extend(rt)
            accuracy_list.extend(accuracy)

        person   = np.array(person_list)
        rt       = np.array(rt_list)
        accuracy = np.array(accuracy_list)

        return Hddm_Data(person, rt, accuracy, T, design.predictor)


    def summary(self):
        if self.person is None or self.rt is None or self.accuracy is None:
            print("Data not available.")
            return

        unique_persons = np.unique(np.array(self.person))
        print("{:<10} {:<20} {:<20} {:<20}".format("Person", "Mean Accuracy", "Mean RT (Correct)", "Variance RT (Correct)"))

        for person_id in unique_persons:
            # Filter data for current person
            person_indices  = np.where(self.person == person_id)
            person_rts      = np.array(self.rt)[person_indices]
            person_accuracy = np.array(self.accuracy)[person_indices]

            # Compute the metrics
            mean_accuracy       = np.mean(person_accuracy)
            correct_rts         = person_rts[person_accuracy == 1]  # only accurate responses
            mean_rt_correct     = np.mean(correct_rts) if len(correct_rts) > 0 else np.nan
            variance_rt_correct = np.var(correct_rts) if len(correct_rts) > 0 else np.nan

            print("{:<10} {:<20.3f} {:<20.3f} {:<20.3f}".format(person_id, mean_accuracy, mean_rt_correct, variance_rt_correct))

    def to_jags(self):
        if self.person is None or self.rt is None or self.accuracy is None:
            return None

        unique_persons = np.unique(np.array(self.person)).astype(int)
        nParticipants  = len(unique_persons)
        nTrials        = np.repeat(int(self.n_TrialsPerPerson), nParticipants)

        # Initialize arrays to NaN for storing metrics
        sum_accuracy        = np.zeros(nParticipants, dtype=int)
        mean_rt_correct     = np.full(nParticipants, np.nan)
        variance_rt_correct = np.full(nParticipants, np.nan)

        # Loop over unique persons and compute metrics
        for person_id in unique_persons:
            # Filter data for the current person
            person_indices  = self.person == person_id
            person_rts      = self.rt[person_indices]
            person_accuracy = self.accuracy[person_indices]

            # Update metrics
            sum_accuracy[person_id] = np.sum(person_accuracy)
            correct_rts = person_rts[person_accuracy == 1]  # only accurate responses

            if correct_rts.size > 1:
                mean_rt_correct[person_id]     = np.mean(correct_rts)
                variance_rt_correct[person_id] = np.var(correct_rts)

        # Filter out participants with NaN values in any metric
        valid_indices = ~(
            np.isnan(mean_rt_correct) |
            np.isnan(variance_rt_correct)
        )

        # Extract valid metrics
        nTrials             = nTrials[valid_indices].tolist()
        sum_accuracy        = sum_accuracy[valid_indices].tolist()
        mean_rt_correct     = mean_rt_correct[valid_indices].tolist()
        variance_rt_correct = variance_rt_correct[valid_indices].tolist()
        X                   = self.X[valid_indices].tolist()

        return {
            "nTrials": nTrials,
            "meanRT":  mean_rt_correct,
            "varRT":   variance_rt_correct,
            "correct": sum_accuracy,
            "X":       X,
        }, unique_persons[valid_indices]


    def __str__(self):
        output = [
            "Hddm_Data Details:",
            f"Person:    {self.person}",
            f"RT:        {self.rt}",
            f"Accuracy:  {self.accuracy}"
        ]
        return '\n'.join(output)



# https://osf.io/download/28ahk/

# Function to read and process the metastudy data
def process_data(file_path):
    # Reading the CSV file
    data = pd.read_csv(file_path)

    # Selecting only the relevant columns
    selected_columns = ['phase', 'acc', 'rt', 'cond', 'session', 'respX', 'respY']
    data = data[selected_columns]

    # Applying the censor/filter conditions
    censor_conditions = (data['rt'] > 2.5) | (data['rt'] < 0.15) | ((data['respX'] == 0) & (data['respY'] == 0)) | (data['phase'] != 1)
    filtered_data = data[~censor_conditions]

    # Selecting the independent variable (iv), dependent variable (dv), and facet
    iv = filtered_data['cond']
    dv = filtered_data['acc']
    facet = filtered_data['session']

    # Printing the results
    print("Independent Variable (iv):", iv.head())
    print("Dependent Variable (dv):", dv.head())
    print("Facet:", facet.head())
    print('done.')

#url = "https://osf.io/download/28ahk/"

#process_data(file_path)





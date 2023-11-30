# Classes to do simulations

import numpy as np
import pandas as pd
import pyjags
import copy

import parameter_set

def ez_jags_code(prior, criterion, version = 'base'):
    if version == 'base':
        code = f"""
        model {{
            # Priors for the hierarchical diffusion model parameters
            betaweight ~ dnorm({prior.betaweight_mean},  {prior.betaweight_sdev**-2})
            bound_mean ~ dnorm({prior.bound_mean_mean},  {prior.bound_mean_sdev**-2}) T( 0.10, 3.00)
            drift_mean ~ dnorm({prior.drift_mean_mean},  {prior.drift_mean_sdev**-2}) T(-3.00, 3.00)
            nondt_mean ~ dnorm({prior.nondt_mean_mean},  {prior.nondt_mean_sdev**-2}) T( 0.05,)
            bound_sdev ~ dunif({prior.bound_sdev_lower}, {prior.bound_sdev_upper})
            drift_sdev ~ dunif({prior.drift_sdev_lower}, {prior.drift_sdev_upper})
            nondt_sdev ~ dunif({prior.nondt_sdev_lower}, {prior.nondt_sdev_upper})

            for (p in 1:length(meanRT)) {{
                bound[p] ~ dnorm(bound_mean{' + betaweight * X[p]' if criterion == 'bound' else ''}, pow(bound_sdev, -2)) T( 0.10, 3.00)
                drift[p] ~ dnorm(drift_mean{' + betaweight * X[p]' if criterion == 'drift' else ''}, pow(drift_sdev, -2)) T(-3.00, 3.00)
                nondt[p] ~ dnorm(nondt_mean{' + betaweight * X[p]' if criterion == 'nondt' else ''}, pow(nondt_sdev, -2)) T( 0.05,)

                # Forward equations from EZ Diffusion
                ey[p]  = exp(-bound[p] * drift[p])
                Pc[p]  = 1 / (1 + ey[p])
                PRT[p] = 2 * pow(drift[p], 3) / bound[p] * pow(ey[p] + 1, 2) / (2 * -bound[p] * drift[p] * ey[p] - ey[p]*ey[p] + 1)
                MDT[p] = (bound[p] / (2 * drift[p])) * (1 - ey[p]) / (1 + ey[p])
                MRT[p] = MDT[p] + nondt[p]

                # Loss functions using MRT, PRT, and Pc
                correct[p] ~ dbin(Pc[p], nTrials[p])
                varRT[p]   ~ dnorm(1/PRT[p], 0.5 * (correct[p]-1) * PRT[p] * PRT[p])
                meanRT[p]  ~ dnorm(MRT[p], PRT[p] * correct[p])
                #meanRT[p]  ~ dnorm(MRT[p], correct[p]/varRT[p])
                #varRT[p]   ~ dnorm(1/PRT[p], 0.5 * correct[p] * PRT[p] * PRT[p])
                #varRT[p]   ~ dgamma(2/(PRT[p]*(correct[p]-1)), 2/(correct[p]-1))
            }}
        }}
        """
    if version == 'bBDN':
        code = f"""
        model {{
            # Priors for the hierarchical diffusion model parameters
            for (b in 1:B) {{
                betaweightB[1,b] ~ dnorm({prior.betaweight_mean},  {prior.betaweight_sdev**-2})
                betaweightD[1,b] ~ dnorm({prior.betaweight_mean},  {prior.betaweight_sdev**-2})
                betaweightN[1,b] ~ dnorm({prior.betaweight_mean},  {prior.betaweight_sdev**-2})
            }}
            bound_mean ~ dnorm({prior.bound_mean_mean},  {prior.bound_mean_sdev**-2}) T( 0.10, 3.00)
            drift_mean ~ dnorm({prior.drift_mean_mean},  {prior.drift_mean_sdev**-2}) T(-3.00, 3.00)
            nondt_mean ~ dnorm({prior.nondt_mean_mean},  {prior.nondt_mean_sdev**-2}) T( 0.05,)
            bound_sdev ~ dunif({prior.bound_sdev_lower}, {prior.bound_sdev_upper})
            drift_sdev ~ dunif({prior.drift_sdev_lower}, {prior.drift_sdev_upper})
            nondt_sdev ~ dunif({prior.nondt_sdev_lower}, {prior.nondt_sdev_upper})

            for (p in 1:length(meanRT)) {{
                bound[p] ~ dnorm(bound_mean + betaweightB[1,] %*% XB[,p], pow(bound_sdev, -2)) T( 0.10, 3.00)
                drift[p] ~ dnorm(drift_mean + betaweightD[1,] %*% XD[,p], pow(drift_sdev, -2)) T(-3.00, 3.00)
                nondt[p] ~ dnorm(nondt_mean + betaweightN[1,] %*% XN[,p], pow(nondt_sdev, -2)) T( 0.05,)

                # Forward equations from EZ Diffusion
                ey[p]  = exp(-bound[p] * drift[p])
                Pc[p]  = 1 / (1 + ey[p])
                PRT[p] = 2 * pow(drift[p], 3) / bound[p] * pow(ey[p] + 1, 2) / (2 * -bound[p] * drift[p] * ey[p] - ey[p]*ey[p] + 1)
                MRT[p] = (bound[p] / (2 * drift[p])) * (1 - ey[p]) / (1 + ey[p]) + nondt[p]

                # Loss functions using MRT, PRT, and Pc
                correct[p] ~ dbin(Pc[p], nTrials[p])
                varRT[p]   ~ dnorm(1/PRT[p], 0.5 * (correct[p]-1) * PRT[p] * PRT[p])
                meanRT[p]  ~ dnorm(MRT[p], PRT[p] * correct[p])
            }}
        }}
        """
    return code

def estimate(dataObject, priorObject, criterion = 'drift', silent = False):

    code = ez_jags_code(priorObject, criterion, 'base')

    data, valid_indices = dataObject.to_jags()

    n_Original_Participants = len(np.unique(dataObject.person))
    n_Participants = len(data['nTrials'])
    #print(f"{n_Original_Participants} participants originally. {len(valid_indices)} valid indices. {n_Participants} participants remain.")

    # Initial values
    init = { "drift" : np.random.normal(0, 0.1, n_Participants) }

    try:
        model = pyjags.Model(
                progress_bar = False,
                code    = code,
                data    = data,
                init    = init,
                adapt   = 100,
                chains  = 4,
                threads = 4)
    except Exception as e:
        if silent:
            print('e', end='')
        else:
            error_message = str(e)
            print(type(error_message))
            print(error_message)
            data.summary()
            print(data.to_jags())
        return

    samples = model.sample(400,
                           vars = ['bound_mean', 'drift_mean', 'nondt_mean',
                                   'bound_sdev', 'drift_sdev', 'nondt_sdev',  'betaweight',
                                   'bound',      'drift',      'nondt'])

    # Annoying management of sample object...  First move individual parameters to their own fields
    for i in range(n_Participants):
        samples.update({'bound_'+str(valid_indices[i]): samples['bound'][i,:,:],
                        'drift_'+str(valid_indices[i]): samples['drift'][i,:,:],
                        'nondt_'+str(valid_indices[i]): samples['nondt'][i,:,:], })

    # ... remove the old unwieldy matrices
    for s in ["bound", "drift", "nondt"]:
        samples.pop(s)

    # Start a new dict with estimates only
    estimate = { "bound": [np.nan] * n_Original_Participants,
                 "drift": [np.nan] * n_Original_Participants,
                 "nondt": [np.nan] * n_Original_Participants
               }

    for varname in ['bound_mean', 'drift_mean', 'nondt_mean', 'betaweight',
                    'bound_sdev', 'drift_sdev', 'nondt_sdev']:
        estimate.update({varname: np.mean(samples[varname])})

    # ... make new, wieldy matrices
    for i in valid_indices:
        estimate['bound'][i] = np.mean(samples['bound_'+str(i)])
        estimate['drift'][i] = np.mean(samples['drift_'+str(i)])
        estimate['nondt'][i] = np.mean(samples['nondt_'+str(i)])

    # Copy estimate to design object
    est = parameter_set.Hddm_Parameter_Set()
    est.bound_mean = estimate['bound_mean']
    est.drift_mean = estimate['drift_mean']
    est.nondt_mean = estimate['nondt_mean']
    est.bound_sdev = estimate['bound_sdev']
    est.drift_sdev = estimate['drift_sdev']
    est.nondt_sdev = estimate['nondt_sdev']
    est.bound      = estimate['bound']
    est.drift      = estimate['drift']
    est.nondt      = estimate['nondt']
    est.betaweight = estimate['betaweight']

    return est


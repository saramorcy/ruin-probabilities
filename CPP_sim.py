import numba
import numpy as np
from numba import njit
from numpy import random

# Simulate exponential rv
@njit
def sim_exp(mu):
    u = random.uniform(0.0, 1.0)
    B = -np.log(u) * (1 / mu)
    return B

# Simulate hyper exponential
@njit
def sim_hyp_exp(vMu, vPhaseProb):
    fU = random.uniform(0.0, 1.0)
    vProbBounds = np.cumsum(vPhaseProb)
    B = sim_exp(vMu[np.argmin((fU >= vProbBounds))])
    return B

# Simulate Poisson rv
@njit
def sim_poisson(w):
    u_0 = random.uniform(0.0,1.0)
    n = 0
    while u_0 >= np.exp(-w):
        u_new = random.uniform()
        u_0 = u_0 * u_new
        n += 1
    return n

# Simulate Hyper Erlang
@njit
def sim_hyp_Erlang(fMu, iNPhases, fPhaseProb):
    """
    fMu = Rate of exponentials
    iNPhases = Number of exponentials
    fPhaseProb = p-value s.t. Erlang(k-1), p-1 s.t Erlang(k)
    """
    fU = random.uniform(0.0, 1.0)
    if fU < fPhaseProb:
        vExponentials = np.array([sim_exp(fMu) for i in range(iNPhases-1)])
        fErlang = np.sum(vExponentials)
    else:
        vExponentials = np.array([sim_exp(fMu) for i in range(iNPhases)])
        fErlang = np.sum(vExponentials)
    return fErlang

# Simulate Log-Nornal
@njit
def sim_lognormal(fMu,fSigma2):
    Z = np.sqrt(-2*np.log(random.uniform(0.0, 1.0)))*np.cos(2*np.pi*random.uniform(0.0, 1.0))
    return np.exp(fMu + np.sqrt(fSigma2)*Z)


""" 
The following functions are for finding the parameters of Hyperexp-2, Hypererlang and lognormal 
such that the variance is is var_x and expectation is 1. (first two moments fit)
"""
# Find parameters for hyper Exponential 2 distribution with variance var_x and expectation 1
@njit
def search_par_hypExp_2(var_x):
    p_1 = 0.5*(1 + np.sqrt((var_x - 1)/(var_x +1)))
    p_2 = 1 - p_1
    mu_1 = 2*p_1
    mu_2 = 2*p_2
    return [np.array([mu_1,mu_2]),np.array([p_1,p_2])]

# Find parameters for hyper Erlang distribution with variance var_x and expectation 1
@njit
def search_par_hypErlang(fCx2):
    iK = np.ceil(1/fCx2)
    fP = 1 /(1 + fCx2) * (iK*fCx2 - (iK*(1 + fCx2) - (iK**2)*fCx2)**(1/2))
    fMu = iK - fP
    return [fMu, iK, fP]

# Find Parameters log-normal
@njit
def search_par_lognormal(fCx2):
    fSigma2 = np.log(fCx2 + 1)
    fMu = -0.5*fSigma2
    return [fMu,fSigma2]

# Make an array of u-values
@njit
def get_Uarray():
    iU = 10 #The highest u desired
    iU_numbers = 10 # The number of different u values desired
    vUarray = np.linspace(1,iU,iU_numbers)

    # The following loop makes is into integers
    for u_index in range(iU_numbers):
        vUarray[u_index] = int(vUarray[u_index])
    return vUarray

"""
The following functions simulate the ruin probabilities for the u-values. 
"""
# Simulate p(u) for HyperErlang 
@njit
def simulate_hypErlang_q(vModel, fOmega, iNPhases, fPhaseProb):
    """
    vModel = [iU,fR,iLambdaP,fMu]
    This describes the compound poisson process where ruin happens when the Y-process is over u at an inspection time
    where premiums come with rate iR and claims come according to a poisson process with rate iLambdaP with the claims
    exponentially distributed with expectation 1/fMu
    The interinspection times are HyperErlang(fOmega,iNPhases,fPhaseProb) distributed where they represent the rate, number
    of phases and the the phase transition probability respectively.
    """
    iNruns = 100*1000
    vUarray = get_Uarray()
    iNumberValU = len(vUarray)
    fR = vModel[0]
    iLambdaP = vModel[1]
    fMu = vModel[2]
    
    # Getting the characterizing values under measure Q
    fThetaStar = fMu - (iLambdaP / fR)
    fLambdaQ = iLambdaP * fMu / (fMu - fThetaStar)
    fMuQ = fMu - fThetaStar

    mYt = np.zeros((iNumberValU, iNruns), dtype= numba.float64)
    # mYt = np.zeros((iNumberValU, iNruns))
    for i in range(iNruns):
        fYt = 0.0
        iCounterU = -1
        #vUarray = np.linspace(1,iU,iNumberValU)
        vUarray = get_Uarray()
        while fYt < vUarray[-1]:
            # Get length interinspection time
            fOmega_i = sim_hyp_Erlang(fOmega, iNPhases, fPhaseProb)
            
            # Get number of claims in timestep omega_i
            fMeanPoissonQ = fLambdaQ * fOmega_i
            iN_i = random.poisson(fMeanPoissonQ)
            #N_i = sim_poisson(lambda_q*omega_i)

            # Get all claims counted up made over timestep omega_i
            fClaims = 0.0
            for j in range(iN_i):
                B_ij = random.exponential(1/fMuQ)
                fClaims = fClaims + B_ij
            
            # Calculate Y on t + omega_i
            fYt = fYt + fClaims - (fR * fOmega_i)
            
            # Check if u_i is surpassed
            while fYt >= vUarray[0] and len(vUarray) > 1:
                iCounterU += 1
                mYt[iCounterU, i] = np.exp(-fThetaStar * fYt)
                vUarray = np.delete(vUarray, 0)
        
        # Add the yT at ruin for biggest u Value
        iCounterU += 1
        mYt[iCounterU,i] = np.exp(-fThetaStar * fYt)

    # Calculate the mean of all the probabilities per row (u value)
    vPu = np.zeros(iNumberValU, dtype= numba.float64)
    for iRowIndex, vRow in enumerate(mYt):
        fRowMean = np.mean(vRow)
        vPu[iRowIndex] = fRowMean
    return vPu

# Simulate p(u) for HyperExp 
@njit
def simulate_hypExp_q(vModel, vRates, vPhaseProb):
    """
    vModel = [iU,fR,iLambdaP,fMu]
    This describes the compound poisson process where ruin happens when the Y-process is over u at an inspection time
    where premiums come with rate iR and claims come according to a poisson process with rate iLambdaP with the claims
    exponentially distributed with expectation 1/fMu
    The interinspection times are HyperExp_2(vRates,vPhaseProb) distributed where they represent the rates and 
    phase transition probabilities respectively.
    """
    iNruns = 100*1000
    vUarray = get_Uarray()
    iNumberValU = len(vUarray)
    fR = vModel[0]
    iLambdaP = vModel[1]
    fMu = vModel[2]

    fThetaStar = fMu - (iLambdaP / fR)
    fLambdaQ = iLambdaP * fMu / (fMu - fThetaStar)
    fMuQ = fMu - fThetaStar

    mYt = np.zeros((iNumberValU, iNruns), dtype= numba.float64)
    for i in range(iNruns):
        fYt = 0.0
        iCounterU = -1
        vUarray = get_Uarray()
        while fYt < vUarray[-1]:
            # Get length interinspection time
            fOmega_i = sim_hyp_exp(vRates, vPhaseProb)
            
            # Get number of claims in timestep omega_i
            fMeanPoissonQ = fLambdaQ * fOmega_i
            iN_i = random.poisson(fMeanPoissonQ)
            #N_i = sim_poisson(lambda_q*omega_i)

            # Get all claims counted up made over timestep omega_i
            fClaims = 0.0
            for j in range(iN_i):
                B_ij = random.exponential(1/fMuQ)
                fClaims = fClaims + B_ij
            
            # Calculate Y on t + omega_i
            fYt = fYt + fClaims - (fR * fOmega_i)
            
            # Check if u_i is surpassed
            while fYt >= vUarray[0] and len(vUarray) > 1:
                iCounterU += 1
                mYt[iCounterU, i] = np.exp(-fThetaStar * fYt)
                vUarray = np.delete(vUarray, 0)
        
        # Add the yT at ruin for biggest u Value
        iCounterU += 1
        mYt[iCounterU,i] = np.exp(-fThetaStar * fYt)
        
    # Calculate the mean of all the probabilities per row (u value)
    vPu = np.zeros(iNumberValU, dtype= numba.float64)
    for iRowIndex, vRow in enumerate(mYt):
        fRowMean = np.mean(vRow)
        vPu[iRowIndex] = fRowMean
    return vPu
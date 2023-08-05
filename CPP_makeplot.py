from CPP_sim import *
import matplotlib.pyplot as plt
import time

"""
Gives for every value of u an array of p(u) values per variance 
Gives back a matrix with all ruin probabilities where every row corresponds 
to a value u and every column corresponds to a variance
Assuming the VModel = [iU, 1.2, 2, 2]
"""

@njit
def get_results_hypErlang(iNumberVar = 50):
    vUarray = get_Uarray()
    vModelVal = np.array([1.2,2,2]) 
    vVar = np.linspace(0.001,0.999,iNumberVar)
    mU_Var_hypErlang = np.zeros((len(vUarray), iNumberVar), dtype= numba.float64)
    for iIndex, fVar in enumerate(vVar):
        parameters = search_par_hypErlang(fVar)
        vPu_hypErlang = simulate_hypErlang_q(vModelVal,parameters[0],int(parameters[1]),parameters[2])
        mU_Var_hypErlang[:,iIndex] = vPu_hypErlang
    return mU_Var_hypErlang

@njit
def get_results_hypExp(iNumberVar = 100):
    vUarray = get_Uarray()
    vModelVal = np.array([1.2,2,2])
    vVar = np.linspace(1,3,iNumberVar)
    mU_Var_hypExp = np.zeros((len(vUarray), iNumberVar), dtype= numba.float64)
    for iIndex, fV in enumerate(vVar):
        parameters = search_par_hypExp_2(fV)
        vPu_hypExp = simulate_hypExp_q(vModelVal,parameters[0],parameters[1])
        mU_Var_hypExp[:,iIndex] = vPu_hypExp
    return mU_Var_hypExp

def make_plot(iX):
    # Create y values for plot
    mPlots_Erlang= get_results_hypErlang(iNumberVar = iX)
    mPlots_hypExp = get_results_hypExp(iNumberVar = 2*iX)
    mPlots = np.concatenate([mPlots_Erlang,mPlots_hypExp], axis=1)

    # Make array of x values for plot
    vVar_1 = np.linspace(0.001,0.999,iX)
    vVar_2 = np.linspace(1,3,2*iX)
    vVar = np.concatenate([vVar_1,vVar_2])
    return mPlots, vVar

def cal_gamma(mPlots):
    vUarray = get_Uarray()
    fThetaStar = 2 - (2 / 1.2)
    mGammaPlots = []
    for iIndex, vPlot in enumerate(mPlots):
        iUval = vUarray[iIndex]
        vGPlot = vPlot * np.exp(fThetaStar * iUval)
        mGammaPlots.append(vGPlot)
    return mGammaPlots

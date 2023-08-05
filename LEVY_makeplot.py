from LEVY_sim import *
import matplotlib.pyplot as plt

@njit
def get_results_lognormal(iNumberVar = 150):
    """
    Gives for every value of u an array of p(u) values per variance:
    Gives back a matrix with all ruin probabilities where every row corresponds 
    to a value u and every column corresponds to a variance
    Assuming the VModel = [1.2, 2, 2, fSigma_2]
    In this case fSigma_2 = 0.02
    """
    vModelVal = np.array([1.2,2,2,0.02])
    vUarray = get_Uarray()
    vVar = np.linspace(0.001,3,iNumberVar)
    mU_Var_lognormal = np.zeros((len(vUarray), iNumberVar), dtype= numba.float64)
    for iIndex, fVar in enumerate(vVar):
        parameters = search_par_lognormal(fVar)
        vPu_lognormal = simulate_lognorm_q(vModelVal,parameters[0],parameters[1])
        mU_Var_lognormal[:,iIndex] = vPu_lognormal
    return mU_Var_lognormal
@njit
def get_results_hypErlang(iNumberVar = 50):
    """
    Gives for every value of u an array of p(u) values per variance:
    Gives back a matrix with all ruin probabilities where every row corresponds 
    to a value u and every column corresponds to a variance
    Assuming the VModel = [1.2, 2, 2, fSigma_2]
    In this case fSigma_2 = 0.02
    """
    vModelVal = np.array([1.2,2,2,0.02])
    vUarray = get_Uarray()
    vVar = np.linspace(0.001,0.999,iNumberVar)
    mU_Var_hypErlang = np.zeros((len(vUarray), iNumberVar), dtype= numba.float64)
    for iIndex, fVar in enumerate(vVar):
        parameters = search_par_hypErlang(fVar)
        vPu_hypErlang = simulate_hypErlang_q(vModelVal,parameters[0],int(parameters[1]),parameters[2])
        mU_Var_hypErlang[:,iIndex] = vPu_hypErlang
    return mU_Var_hypErlang

@njit
def get_results_hypExp(iNumberVar = 100):
    """
    Gives for every value of u an array of p(u) values per variance:
    Gives back a matrix with all ruin probabilities where every row corresponds 
    to a value u and every column corresponds to a variance
    Assuming the VModel = [1.2, 2, 2, fSigma_2]
    In this case fSigma_2 = 0.02
    """
    vModelVal = np.array([1.2,2,2,0.02])
    vUarray = get_Uarray()
    vVar = np.linspace(1,3,iNumberVar)
    mU_Var_hypExp = np.zeros((len(vUarray), iNumberVar), dtype= numba.float64)
    for iIndex, fV in enumerate(vVar):
        parameters = search_par_hypExp_2(fV)
        vPu_hypExp = simulate_hypExp_q(vModelVal,parameters[0],parameters[1])
        mU_Var_hypExp[:,iIndex] = vPu_hypExp
    return mU_Var_hypExp

def make_plot_lognorm(iX):
    """
    This function creates the smoothed plot values:
    y values = ruin probabilities per u depending on the variance of the interinspection times
    x values = the variances of the interinspection times (going from 0 up to 3)

    Every row in mPlots corresponds to one graph for u_i
    """
    # Create y values for plot
    mPlots_Lognorm = get_results_lognormal(iNumberVar = 3*iX)
    mPlots = mPlots_Lognorm

    # Make array of x values for plot
    vVar = np.linspace(0.001,3,3*iX)
    return mPlots, vVar

def make_plot_phasetype(iX):
    """
    This function creates the smoothed plot values:
    y values = ruin probabilities per u depending on the variance of the interinspection times
    x values = the variances of the interinspection times (going from 0 up to 3)
    Every row in mPlots corresponds to one graph for u_i
    """
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
    """
    This function returns a scaled plot of the ruin probabilities where the y-values have been altered:
    y values = (ruin probabilities per u_i depending on the variance of the interinspection times) * np.exp(fThetastar * u_i)
    x values = the variances of the interinspection times (going from 0 up to 3)

    Every row in mGammaPlots corresponds to one graph for u_i
    """
    vModelVal = np.array([1.2,2,2,0.02])
    fR, iLambdaP, fMu, fSigma_2 = vModelVal
    fThetaStar = (fMu*fSigma_2 + 2*fR)/(2*fSigma_2) - 1/(2*fSigma_2) * np.sqrt((fMu*fSigma_2 + 2*fR)**2 + 4*fSigma_2*(2*iLambdaP - 2*fR*fMu))
    vUarray = get_Uarray()
    mGammaPlots = []
    for iIndex, vPlot in enumerate(mPlots):
        iUval = vUarray[iIndex]
        vGPlot = vPlot * np.exp(fThetaStar * iUval)
        mGammaPlots.append(vGPlot)
    return mGammaPlots

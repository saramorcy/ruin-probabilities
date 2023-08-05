import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from math import comb, factorial
from LEVY_sim import search_par_hypExp_2, search_par_hypErlang

# Define Initial and Global values
vModel = np.array([1.2,2,2,0.02])
fR, iLambdaP, fMu, sigma_2 = vModel
theta_star = (fMu*sigma_2 + 2*fR)/(2*sigma_2) - 1/(2*sigma_2) * np.sqrt((fMu*sigma_2 + 2*fR)**2 + 8*sigma_2*(iLambdaP - fR*fMu))
def b(x):
    return fMu / (x + fMu)  
def phi(alpha):
    return 0.5 * alpha**2 * sigma_2 + fR*alpha - iLambdaP*(1 - b(alpha))
def phi_q(alpha):
    return 0.5 * alpha**2 * sigma_2 + (fR - theta_star*sigma_2)*alpha - iLambdaP * b(-theta_star)*(1 - b(alpha - theta_star)/b(-theta_star)) 
def phi_tilde(alpha):
     return (0.5 * alpha**2 * sigma_2 + (fR - theta_star*sigma_2)*alpha)*(alpha - theta_star + fMu) - \
        iLambdaP * b(-theta_star)*(alpha - theta_star + fMu - fMu/b(-theta_star))
def fDer_phi(fAlpha):
    alpha = sp.Symbol('alpha')
    expr = phi_q(alpha)
    der_res = expr.diff(alpha, 1)
    return der_res.subs(alpha, fAlpha)
fDer_Phi_min_thetastar = fDer_phi(0)

# Defining \psi(\beta)
def Psi_S(fBeta):
    """
    This function defines the right inverse function "psi" of phi 
    """
    alpha = sp.Symbol('alpha')

    # Solve the equation to obtain inverse
    vSol_psi_beta = sp.solve(fBeta - phi(alpha), alpha)

    # Obtain the relevant (real valued positive) solution of the equation
    for solution in vSol_psi_beta:
        if solution.as_real_imag()[0] > 0:
            fPsiBeta = solution.as_real_imag()[0]
    return fPsiBeta

# Defining \psi_Q(\beta)
def Psi_q_S(fBeta):
    """
    This function defines the right inverse function "psi" of phi under measure Q
    """
    alpha = sp.Symbol('alpha')

    # Solve the equation to obtain inverse
    vSol_psi_beta = sp.solve(fBeta - phi_q(alpha), alpha)

    # Obtain the relevant (real valued positive) solution of the equation
    for solution in vSol_psi_beta:
        if solution.as_real_imag()[0] > 0:
            fPsiBeta = solution.as_real_imag()[0]
    return fPsiBeta


def Zeta_Q(fAlpha, fBeta):
    """
    This function defines xi_Q(alpha,beta)
    This function is called Zeta in this program, since xi is a reserved variable
    """
    fPsi_Beta = Psi_S(fBeta)

    # L'hopital for fBeta=phi_q(alpha)
    if - (10**(-14))< fBeta - phi(fAlpha - theta_star) < (10**(-14)):
        # Seperate terms
        fVal_Zeta_Q_part1 = 1/(fPsi_Beta*sigma_2 + fR - iLambdaP*fMu/(fMu + fPsi_Beta)**2)
        fVal_Zeta_Q_part2 = fBeta/(fPsi_Beta + theta_star)
        # Collect result
        fVal_Zeta_Q = fVal_Zeta_Q_part1 * fVal_Zeta_Q_part2 
    
    # Direct computation where it is not needed
    else:
        # Seperate terms
        fVal_Zeta_Q_part1 = (fPsi_Beta + theta_star - fAlpha)/(fPsi_Beta + theta_star) 
        fVal_Zeta_Q_part2 = fBeta/(fBeta - phi(fAlpha - theta_star))
        # Collect result
        fVal_Zeta_Q = fVal_Zeta_Q_part1 * fVal_Zeta_Q_part2
        
    return fVal_Zeta_Q



def F_circle(fAlpha,fBeta,fTheta_i):
    """
    This function calculates F^circ_i(alpha,beta), where theta_i corresponds to which F_i
    """
    # Direct computation big divisions
    fDiv_2 = (Zeta_Q(theta_star, fBeta) - Zeta_Q(fAlpha, fBeta))/ (fAlpha - theta_star)
    fDiv_1 = (Zeta_Q(fTheta_i,fBeta) - Zeta_Q(fAlpha,fBeta))/(fAlpha - fTheta_i)

    return fTheta_i / (theta_star - fTheta_i) * (fDiv_1 - fDiv_2)

def F_circle_alpha1(fW_i, fTheta_i):
    """
    Special case for alpha = theta^star (Use l'hopital)
    This function calculates F^circ_i(alpha_1,w_i), where alpha_1 = theta^star
    """
    fAlpha_1 = theta_star
    # Direct computation big division 1
    fDiv_1 = (Zeta_Q(fTheta_i,fW_i) - Zeta_Q(fAlpha_1,fW_i)) / (fAlpha_1 - fTheta_i)

    # l'hopital for big division 2
    fDiv_2 = (fW_i - (fTheta_i - theta_star)*(fR - iLambdaP/fMu)) / (fTheta_i*fW_i)

    return fTheta_i / (theta_star - fTheta_i) * (fDiv_1 - fDiv_2)



def Calculate_theta_12(parameters):
    """
    For: Hyp-Exp distributed inter inspection times.
    This function calculates theta_1 = Psi(w_1) and theta_2 = Psi(w_2).
    """
    alpha = sp.Symbol('alpha')
    vParameters_w = parameters[0]
    fW_1, fW_2 = vParameters_w

    # Solve equations to find inverses theta_1 and theta_2
    vSol_theta_1 = sp.solve(fW_1 - phi_q(alpha), alpha)
    vSol_theta_2 = sp.solve(fW_2 - phi_q(alpha), alpha)

     # Obtain the relevant (real valued positive) solution of the equations
    for solution in vSol_theta_1:
        if solution.as_real_imag()[0] > 0:
            fTheta_1 = solution.as_real_imag()[0]
    for solution in vSol_theta_2:
        if solution.as_real_imag()[0] > 0:
            fTheta_2 = solution.as_real_imag()[0]
            
    return [fTheta_1, fTheta_2]

# Make equation for finding unknowns for alpha2
def Get_Left_EQ(fAlpha, parameters):
    """
    Parameters = [[w_1,w_2],[p_1,p_2]] for Hyper-Exponential distribution.
    This function gets left side of the equation of numerator = 0 for alpha=alpha_2 (root of denominator)
    """
    # Define unknowns
    x_1 = sp.Symbol('x_1')
    x_2 = sp.Symbol('x_2')

    # Get all needed parameters
    fTheta_1, fTheta_2 = Calculate_theta_12(parameters)
    vParameters_w, vParameters_p = parameters
    fP1, fP2 = vParameters_p
    fW_1, fW_2 = vParameters_w

    # ## This if statement is not needed 
    # # To make sure we never divide by zero
    # if fTheta_1 == fTheta_2 == fAlpha:
    #     fAlpha = fAlpha - 1**(-14)

    # Calculate first and second part of the equation (p_1,p_2)
    fPart_1 = fP1 *(F_circle(fAlpha, fW_1, fTheta_1) - fTheta_1/(fTheta_1 - fAlpha) * Zeta_Q(fTheta_1, fW_1)*x_1)
    fPart_2 = fP2 *(F_circle(fAlpha, fW_2, fTheta_2) - fTheta_2/(fTheta_2 - fAlpha) * Zeta_Q(fTheta_2, fW_2)*x_2)

    return fPart_1 + fPart_2

# Make equation for finding unknowns for alpha1 = \theta^\star
def Get_Left_EQ_alpha1(parameters):
    """
    Parameters = [[w_1,w_2],[p_1,p_2]] for Hyper-Exponential distribution.
    This function gets left side of the equation of numerator = 0 for alpha=alpha_1 (root of denominator)
    """
    # Define unknowns
    x_1 = sp.Symbol('x_1')
    x_2 = sp.Symbol('x_2')
    # Get all needed parameters
    fAlpha_1 = theta_star
    fTheta_1, fTheta_2 = Calculate_theta_12(parameters)
    vParameters_w, vParameters_p = parameters
    fP1, fP2 = vParameters_p
    fW_1, fW_2 = vParameters_w

    # Make equations
    fPart_1 = fP1 *(F_circle_alpha1(fW_1, fTheta_1) - fTheta_1/(fTheta_1 - fAlpha_1) * Zeta_Q(fTheta_1, fW_1)*x_1)
    fPart_2 = fP2 *(F_circle_alpha1(fW_2, fTheta_2) - fTheta_2/(fTheta_2 - fAlpha_1) * Zeta_Q(fTheta_2, fW_2)*x_2)
    return fPart_1 + fPart_2

# Note that we only need to find one root, as we know that the other is alpha_1 = \theta^\star
def Calculate_alpha2(parameters):
    """
    For: Hyp-Exp distributed inter inspection times.
    This function finds the root alpha_2 of the denominatior of pi(alpha|HypExp)
    The Hyp-Exp distribution has omega = (W_1,W_2) and p = (p_1,p_2)
    """
    alpha = sp.Symbol('alpha')
    vParameters_w, vParameters_p = parameters
    fP1, fP2 = vParameters_p
    fW_1, fW_2 = vParameters_w

    # Solve the equation to obtain solution for alpha_2
    vSol_alpha_2 = sp.solve(fP2*fW_1 + fP1*fW_2 - phi_q(alpha), alpha)

    # Obtain the relevant (real valued positive) solution of the equation
    for solution in vSol_alpha_2:
        if solution.as_real_imag()[0] > 0:
            fAlpha_2 = solution.as_real_imag()[0]
    return fAlpha_2

# Calculate unknowns 
def Calculate_z12(parameters):
    """
    This function finds the unknown constants z_0 and z_1 (which were pi(theta)).
    """
    # Get roots of denominator
    fAlpha_2 = Calculate_alpha2(parameters)
    # Define unknowns
    x_1 = sp.Symbol('x_1')
    x_2 = sp.Symbol('x_2')
    # Get the two equations
    Left_EQ_alpha1 = Get_Left_EQ_alpha1(parameters)
    Left_EQ_alpha2 = Get_Left_EQ(fAlpha_2, parameters)
    # Find solutions for unknowns
    dSol_z = sp.solve([Left_EQ_alpha1, Left_EQ_alpha2], [x_1,x_2])
    
    # Assign unknowns
    fZ_1 = dSol_z[x_1]
    fZ_2 = dSol_z[x_2]
    return [fZ_1, fZ_2]

# Functions needed for the gamma expression in the denominator
def Get_Phi_functions(parameters):
    """
    For: Hyp-Exp distributed inter inspection times.
    """
    vParameters_w = parameters[0]
    fW_1, fW_2 = vParameters_w
    #fDer_Phi_min_thetastar = -sigma_2 * theta_star + fR - iLambdaP*fMu/ (fMu - theta_star)**2
    fBig_Phi_0 = fW_1*fW_2
    fDer_Big_Phi_0 = -(fW_1 + fW_2)*fDer_Phi_min_thetastar
    return [fDer_Phi_min_thetastar, fBig_Phi_0, fDer_Big_Phi_0]

# Note that the index i is given using the argument fW_i which is \omega_i
def Der_Big_Phi_i_0(fW_i, parameters):
    fDer_Phi_min_thetastar, fBig_Phi_0, fDer_Big_Phi_0 = Get_Phi_functions(parameters)
    return (fW_i * fDer_Big_Phi_0 + fBig_Phi_0*fDer_Phi_min_thetastar) / (fW_i**2)

# Calculate gamma with no unknowns
def Theoretic_Gamma_Hexp(parameters):
    # Get all needed parameters
    vParameters_w, vParameters_p = parameters
    fP1, fP2 = vParameters_p
    fW_1, fW_2 = vParameters_w
    fTheta_1, fTheta_2 = Calculate_theta_12(parameters)
    fZ_1, fZ_2 = Calculate_z12(parameters)

    # Define values for denominator part of gamma
    fBig_Phi_0, fDer_Big_Phi_0 = [Get_Phi_functions(parameters)[1], Get_Phi_functions(parameters)[2]]

    # Calculate numerator
    upper_left = fBig_Phi_0 * fP1*(F_circle(0,fW_1, fTheta_1) - Zeta_Q(fTheta_1, fW_1)*fZ_1)
    upper_right = fBig_Phi_0 * fP2*(F_circle(0, fW_2, fTheta_2) - Zeta_Q(fTheta_2, fW_2)*fZ_2)

    # Calculate denominator
    under = fDer_Big_Phi_0 - (fP1*Der_Big_Phi_i_0(fW_1, parameters)*fW_1 + (1 - fP1)*Der_Big_Phi_i_0(fW_2, parameters)*fW_2)

    return (upper_left + upper_right) / under 

# Below follow functions only needed for Hyp-Erlang distributed interinspection times 

# Define Zeta_Q_k function
def Zeta_Q_k(fAlpha, fBeta, iK):
    """
    For: Hyp-Erlang distributed inter inspection times.
    This function defines xi_Q(alpha,beta)^k
    This function is called Zeta in this program, since xi is a reserved variable
    """
    fVal_Zeta_Q_k = (Zeta_Q(fAlpha, fBeta))**iK
    return fVal_Zeta_Q_k

# Find theta= psi_q(fW)
def Cal_Theta(parameters):
    """
    For: Hyp-Erlang distributed inter inspection times.
    This function calculates theta = Psi(w).
    """
    alpha = sp.Symbol('alpha')
    fW = parameters[0]
    vSol_theta = sp.solve(fW - phi_q(alpha), alpha)
    for solution in vSol_theta:
        if solution.as_real_imag()[0] > 0:
            fTheta = solution.as_real_imag()[0]
    return fTheta

# Define nth derivative Zeta_Q_k function
def D_Zeta_Q_k(parameters,iK):
    """
    For: Hyp-Erlang distributed inter inspection times.
    This function returns the zeroth up to the kth derivative in a vector of xi_Q,k evaluated at theta.
    """
    alpha = sp.Symbol('alpha')
    fW = parameters[0]
    iK = int(iK)
    fTheta = Cal_Theta(parameters)
    fBeta = fW
    fPsi_Beta_q = Psi_q_S(fBeta)
    Teller_1 = (alpha - fPsi_Beta_q)*fBeta
    Noemer_1 = (phi_q(alpha) - fBeta)*(fPsi_Beta_q)
    Teller = sp.Pow(Teller_1, iK)
    Noemer = sp.Pow(Noemer_1, iK)
    fVal_Zeta_Q_k = Teller/Noemer    
    
    # Make list of N and T values
    lN = [Noemer]
    lT = [Teller]
    for n in range(1,iK+1):
        lN.append(sp.diff(lN[-1], alpha))
        lT.append(sp.diff(lT[-1], alpha))
        
    # Make list of Zeta values
    lDZeta = [Teller/Noemer, (lT[1] - lN[1]*fVal_Zeta_Q_k)/lN[0]]
    for n in range(2,iK+1):
        som = 0
        for i in range(1,n+1):
            som += comb(n,i)*lN[i]*lDZeta[n-i]
        Simplified_DZeta = (lT[n] - som)/lN[0]
        lDZeta.append(Simplified_DZeta)
    
    lDZeta_Theta = []
    for itemDZeta in lDZeta:
        # Interpolate
        plus = itemDZeta.subs(alpha, fTheta + 0.01)
        min  = itemDZeta.subs(alpha, fTheta - 0.01)
        meanDZeta = (plus + min)/2
        lDZeta_Theta.append(meanDZeta)
    return lDZeta_Theta

def I_circle(fAlpha, i, parameters, lDZeta_Theta):
    fW, iK, _ = parameters
    fTheta = Cal_Theta(parameters)
    #i can only be 1 or 2
    if i == 1:
        iK_i = int(iK -1)
    else:
        iK_i = int(iK)
    
    # Calculation part_1 of I^circ
    if fAlpha == fTheta:
        fAlpha = fAlpha - 10**(-10)
    if theta_star - 10**(-10) < fAlpha < theta_star + 10**(-10):
        fAlpha_plus = fAlpha + 10**(-10)
        fAlpha_min = fAlpha - 10**(-10)

        part_1_plus = (fTheta/(fTheta - theta_star))**iK_i * \
        (Zeta_Q_k(theta_star,fW,iK_i) - Zeta_Q_k(fAlpha_plus,fW,iK_i))/(fAlpha_plus - theta_star)

        part_1_min = (fTheta/(fTheta - theta_star))**iK_i * \
        (Zeta_Q_k(theta_star,fW,iK_i) - Zeta_Q_k(fAlpha_min,fW,iK_i))/(fAlpha_min - theta_star)

        part_1 = (part_1_plus + part_1_min)/2
    else:
        part_1 = (fTheta/(fTheta - theta_star))**iK_i * \
            (Zeta_Q_k(theta_star,fW,iK_i) - Zeta_Q_k(fAlpha,fW,iK_i))/(fAlpha - theta_star)
    
    # Calculation Part 2 of I^circ
    part_2 = 0
    for n in range(iK_i):
        som_m = 0
        for m in range(n+1):
            som_m += lDZeta_Theta[i-1][m] * ((fAlpha - fTheta)**m)/factorial(m)
        part_2_1 = (fTheta**iK_i)/((fTheta - theta_star)**(iK_i)) * (fTheta - theta_star)**n/((fTheta - fAlpha)**(n+1))
        part_2 += part_2_1 * (Zeta_Q_k(fAlpha,fW,iK_i) - som_m)

    return part_1 - part_2

def Delta_Q(n, i, parameters, lDZeta_Theta):
    fTheta = Cal_Theta(parameters)
    return ((-fTheta)**n/factorial(n))*lDZeta_Theta[i-1][n]

def J_bar_circle(fAlpha,i,j,parameters, lDZeta_Theta):
    iK = parameters[1]
    fTheta = Cal_Theta(parameters)
    if i == 1:
        iK_i = int(iK -1)
    else:
        iK_i = int(iK)
    som_m = 0
    for m in range(j,iK_i):
        som_m += ((fTheta/(fTheta - fAlpha))**(m+1))*Delta_Q(iK_i-1-m,i, parameters, lDZeta_Theta)*((fAlpha-fTheta)**j)/factorial(j) 
    return som_m

def J_circle(fAlpha,i,parameters,lDZeta_Theta):
    iK = parameters[1]
    if i == 1:
        iK_i = int(iK - 1)
    else:
        iK_i = int(iK)
    z_list = []
    for k in range(int(iK)):
        name_zi = "z_" + str(k) 
        z_list.append(sp.Symbol(name_zi))
    som_j = 0
    for j in range(iK_i):
        som_j += J_bar_circle(fAlpha,i,j,parameters,lDZeta_Theta)*z_list[j]
    return som_j

# Find array of alphas
def Cal_Alpha(parameters):
    alpha = sp.Symbol('alpha')
    fW, iK, fP = parameters
    iK = int(iK)
    vAlphas = np.zeros(iK, dtype=complex)
    fTerm = alpha - theta_star + fMu
    # The equation rewritten to make into polynomial
    EQ = (fW**iK) * fTerm**(iK-1) *(fW * fTerm - fP * phi_tilde(alpha)) - fW*(fW*fTerm - phi_tilde(alpha))**iK
    EQ_Poly = sp.poly(EQ, alpha)
    vCoeffs = EQ_Poly.all_coeffs()
    vSol_alpha = np.roots(vCoeffs)

    # Check if the roots are actually solutions to the original equation (denominator of pi(alpha)=0)
    iCounter_alphas = 0
    for solution in vSol_alpha:
        if iCounter_alphas == iK :
            break
        if solution.real > 0:
            EQ_val_check = 1 - fP*(fW/(fW - phi_q(solution)))**(iK - 1) - (1 - fP)*(fW/(fW - phi_q(solution)))**iK
            if -10**(-7) < EQ_val_check.real < 10**(-7):
                vAlphas[iCounter_alphas] = solution
                iCounter_alphas += 1
    return vAlphas


def Find_z_i(parameters, lDZeta_Theta):
    """
    This function finds the unknown constants z_0,...,z_k.
    """
    vAlphas = Cal_Alpha(parameters)
    _ , iK, fP = parameters
    z_list = []
    for k in range(int(iK)):
        name_zi = "z_" + str(k) 
        z_list.append(sp.Symbol(name_zi))
    lEQ = []
    for j in range(int(iK)):
        fAlpha_j = vAlphas[j]
        EQ_1 = fP*(I_circle(fAlpha_j,1, parameters, lDZeta_Theta) - J_circle(fAlpha_j,1,parameters,lDZeta_Theta))
        EQ_2 = (1 - fP)*(I_circle(fAlpha_j,2, parameters, lDZeta_Theta) - J_circle(fAlpha_j,2,parameters,lDZeta_Theta))
        lEQ.append(EQ_1 + EQ_2)
    vSol_z = sp.solve(lEQ,z_list)
    return vSol_z

def Cal_Gamma_HErlang(parameters):
    # Calculating needed constants
    fW, iK, fP = parameters
    lDZeta_Theta_k = D_Zeta_Q_k(parameters, iK)
    lDZeta_Theta_k_1 = D_Zeta_Q_k(parameters, iK - 1)
    lDZeta_Theta = [lDZeta_Theta_k_1, lDZeta_Theta_k]
    vSol_z = Find_z_i(parameters,lDZeta_Theta).values()

    # Calculate numerator (with z_0,...,z_k symbols)
    T_1 = fP*(I_circle(0,1,parameters, lDZeta_Theta) - J_circle(0,1,parameters,lDZeta_Theta))
    T_2 = (1 - fP)*(I_circle(0,2, parameters, lDZeta_Theta) - J_circle(0,2,parameters,lDZeta_Theta))
    Numerator = T_1 + T_2
    # z_list = []
    # for k in range(int(iK)):
    #     name_zi = "z_" + str(k) 
    #     z_list.append(sp.Symbol(name_zi)) 
    
    # Calculate denominator
    Denominator = -fP*(iK - 1)*fDer_Phi_min_thetastar/fW - (1-fP)*iK*fDer_Phi_min_thetastar/fW

    # Gamma with unknown constants as symbols
    Gamma_Exp_symbolic = Numerator/Denominator

    # Fill in unknown constants in Gamma expression
    z_list = []
    for k in range(int(iK)):
        name_zi = "z_" + str(k) 
        z_list.append(sp.Symbol(name_zi)) 
    f = sp.lambdify([z_list], Gamma_Exp_symbolic)
    return f(vSol_z)

# Below follows the code for calculating the values for the graph of gamma for HypExp and hypErlang
def Make_Plot_Gamma_Hexp():
    vVar = np.linspace(1.02020202,3,99)
    vGammas = []
    for fVar_Omega in vVar:
        parameters = search_par_hypExp_2(fVar_Omega)
        Gamma = Theoretic_Gamma_Hexp(parameters)
        vGammas.append(Gamma)
    return vVar, vGammas

def Make_Plot_Gamma_HErlang():
    vVar = np.linspace(0.16393878,0.999,42)
    # vVar = np.linspace(0.16393878, 0.99, 30)
    gamma_list = []
    for var in vVar: 
        parameters = search_par_hypErlang(var)
        gamma = Cal_Gamma_HErlang(parameters)
        gamma_list.append(gamma.as_coeff_Add()[0])
    return vVar, gamma_list
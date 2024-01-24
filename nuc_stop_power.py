import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as optimize
import universal_stopping as us

# H-1 mass 1, Z 1
# Si-28 mass 28, Z 14
# Au-197 mass 197, Z 79

#####################################################################################
#               Calculation of nuclear stopping power of an ion                     #
#                                                                                   #
#   The program calculates the nuclear stopping power using two nested integrals.   #
#   The integrals are evaluated using Simpson's method and r_min is evaluated       #
#   using scipy.optimize.minimize function. The program calculates both cases and   #
#   returns the integral values for all eV values from 10 eV -> 5 MeV. These        #
#   results are then plotted on two separate images.                                #
#                                                                                   #
#####################################################################################


eVtoJ = 1.6e-19 # Conversion multiplication from eV to J
eps_0 = 8.854187817e-12 # epsilon_0 used in universal screened Coulomb potential
e = 1.6021892e-19 # elementary charge

def main():
    b_max = 1e-9 # Same b_max is used across all simulations

    elements = [[1,14],[14,79]] # Z1,Z2 for both cases
    E = np.logspace(1,6.7, num=1000, base=10.0) # eV energies from 10 eV -> 5 MeV
    cases = [[[1,1],[14,28]],[[14,28],[79,197]]] # Both cases in the form [[Z1,M1],[Z2,M2]]

    S_power_data = []
    for case in cases: # Simulations for all cases
        S_power_data.append(run_simulations(case, b_max))
    fig, axs = plt.subplots(1,2, sharey = True, sharex = True)
    i = 0
    for data in S_power_data:
        Z1, Z2 = elements[i]
        axs[i].plot(E,data[0],label=f"Universal nuclear stopping power")
        axs[i].plot(E,data[1],label=f"Calculated nuclear stopping power")
        axs[i].set_title(f"When $Z_1$: {Z1} and $Z_2$: {Z2}")
        plt.xlabel("Energy in eV")
        plt.ylabel("eV/(atoms/$cm^2$)")
        i += 1
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig("stop_power.png")
    plt.show()



def run_simulations(case, b_max):
    Z1,M1 = case[0]
    Z2,M2 = case[1]

    E1 = [Z1,M1]
    E2 = [Z2,M2]

    E = np.logspace(1,6.7, num=1000, base=10.0) # eV energies from 10 eV -> 5 MeV

    if Z1 == 1:
        print(f"H-1 --> Si-28, E_lab = 10eV,...,5 MeV")
    else:
        print(f"Si-28 --> Au-197, E_lab = 10eV,...,5 MeV")
    print("---------------------------------------")
    E_k = E/1000 # keV energies for universal formula
    S1 = []
    for e in E_k:
        S1.append(us.sp(e,E1,E2)) # Universal nuclear stopping power
    S2 = []
    i = 0
    for e in E:
        S2.append(stopping_power(e,b_max,E1,E2)) # Calculated nuclear stopping power
        i += 1
        if i%100 == 0:
            print(f"{(i/1000)*100} %") # Prints the progress every 10%
    print("")
    return S1,S2

def stopping_power(E_lab, b_max, E1, E2):
    _,M1 = E1
    _,M2 = E2
    gamma = 4*M1*M2/((M1+M2)**2)
    coefs = [E_lab,E1,E2]
    res_int = simpson(1e-10,b_max,1e-10,0,5,sp_f,coefs) # Integral starts from 1e-10 in order to avoid nan return
    return 2*np.pi*gamma*E_lab*res_int*1e4 # Multiplied by 1e4 to make conversion from m2 -> cm2




def theta(b,E_com,elements):
    Z1,Z2 = elements
    r_min = find_root_r(b,E_com,elements)
    coefs = [b,E_com,r_min[0],Z1,Z2]
    res_int = simpson(1e-10,1,1e-10,0,5,theta_f,coefs) # Integral starts from 1e-10 in order to avoid nan return
    return np.pi-res_int*4*b 

   

def find_root_r(b,E_com,elements):
    Z1,Z2 = elements
    coefs = [b,Z1,Z2,E_com]
    init_guess = 1e-10 # Init guess in the order of angstrom
    res = optimize.root(g,init_guess, tol=1e-10, args=coefs) # Uses the default hybr which is a modification of the Powell hybrid method
    return res.x



def V(r,Z1,Z2):
    a_u = 0.46848e-10/(Z1**0.23+Z2**0.23)
    return Z1*Z2*e**2/(4*np.pi*eps_0*r)*screen_func(r/a_u)


def screen_func(x): # The screening function phi
    alphas = [0.1818, 0.5099, 0.2802, 0.02817]
    betas = [3.2, 0.9423, 0.4028, 0.2016]
    func_val = 0
    for i in range(4):
        func_val += alphas[i]*np.exp(-betas[i]*x)
    return func_val

def g(var,coefs):
    r = var
    b,Z1,Z2,E_com = coefs
    return (1-(b/r)**2-V(r,Z1,Z2)/E_com) # Find the root of g(r)^2



def theta_f(u,coefs): #F(u) function used in theta
    b,E_com,r_min,Z1,Z2 = coefs
    if(u == 1): # Exception so that no error is given in the case 0 is in the division in (V(r_min/(1-u**2)))
        return (b**2*(2-u**2)+r_min**2/(u**2*E_com)*(V(r_min,Z1,Z2)))**(-1/2)
    return (b**2*(2-u**2)+r_min**2/(u**2*E_com)*(V(r_min,Z1,Z2)-V(r_min/(1-u**2),Z1,Z2)))**(-1/2)


def sp_f(b,coefs): # Nuclear stopping power S_n integral function
    E_lab, E1, E2 = coefs
    Z1,M1 = E1
    Z2,M2 = E2
    elements = [Z1,Z2]
    E_com = E_lab*(M2/(M1+M2))*eVtoJ
    return b*np.sin(theta(b,E_com,elements)/2.0)**2



def simpson(a,b, eps, level, level_max,f,coefs): # Used Simpson's method for integral evaluation
    h = b-a
    c = (a+b)/2.0
    one_simpson = h/6*(f(a,coefs)+4*f(c,coefs)+f(b,coefs))
    d = (a+c)/2.0
    e = (c+b)/2.0
    two_simpson = h/12*(f(a,coefs)+4*f(d,coefs)+2*f(c,coefs)+4*f(e,coefs)+f(b,coefs))
    if(level+1 >= level_max):
        result = two_simpson

    else:
        if(np.abs(two_simpson-one_simpson) < 15.0*eps):
            result = two_simpson +(two_simpson-one_simpson)/15.0
        else:
            left_simpson = simpson(a,c,eps/2.0,level+1,level_max,f,coefs)
            right_simpson = simpson(c,b,eps/2.0,level+1,level_max,f,coefs)
            result = left_simpson+right_simpson
    return result

if __name__ == '__main__':
    main()



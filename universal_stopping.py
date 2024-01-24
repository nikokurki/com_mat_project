import numpy as np

#####################################################
#    This module is used to calculate universal     #
#    nuclear stopping power formula.                #
#                                                   #
#####################################################

def sp(E,E1,E2):
    Z1,M1 = E1
    Z2,M2 = E2
    eps = epsilon(E,Z1,Z2,M1,M2)
    s_n = sn(eps)
    return 8.462e-15*Z1*Z2*M1/((M1+M2)*(Z1**0.23+Z2**0.23))*s_n



def epsilon(E,Z1,Z2,M1,M2):
    return 32.35*M2*E/(Z1*Z2*(M1+M2)*(Z1**0.23+Z2**0.23))


def sn(e):
    if e > 30:
        return np.log(e)/(2*e)
    return np.log(1+1.138*e)/(2*(e+0.01321*e**0.21226+0.19593*e**0.5))
# Project work
 Numerical Methods in Scientific Computing -course project work involving stopping power.

 The task was to do calculation of nuclear stopping power of an ion and compare it with a universal nuclear stopping power formula that is based on fitting a suitable formula to a large number of individually calculated projectile-target combinations. 

 Numerical calculations were based on the general stopping power formula which included classical scattering integral of the ion and the atom interacting via a screened Coulomb potential. The interaction between energetic ions and atoms was approximated using universal screened Coulomb potential developed by Ziegler, Biersack and Littmark (ZBL model). Using the scattering integral, distance of minimum approach of the ion and the atom _r_min_ was found although it also depended on the collision parameter _b_. 
 Now using the classical scattering integral, an expression for the nuclear stopping power can be derived as the integral of the energy loss in a collision over all possible collision 
 parameters _b_.

 The distance of minimum approach or the root of _r_min_ was found using scipy and all integrals were calculated using Simpson's method (5 levels in this code). More accurate results could have been achieved using for example Gauss-Legendre quadrature but Simpson's method proved to be sufficiently accurate.

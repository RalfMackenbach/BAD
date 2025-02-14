from BAD.bounce_fieldline import bounce_int_lambda, bounce_int_zbp
import numpy as np
from scipy.special import ellipk
from scipy.integrate import quad

def B(z):
    return 1.0 - 0.5 * np.cos(z)

def h(z):
    return np.ones_like(z)

def t_bounce_exact(lam, a=0.5):
    # ArcCos[(-1 + \[Lambda])/(a \[Lambda])]
    bounce_point = np.arccos((-1.0+lam)/(a*lam))
    # define integrand
    integrand = lambda z: 1/np.abs(1.0-lam*(1.0-a*np.cos(z)))**0.5
    # compute integral
    integral = quad(integrand, -bounce_point, bounce_point, epsabs=1e-12, epsrel=1e-12, limit=1000)[0]
    return integral

    
    


z = np.linspace(-2*np.pi, +2*np.pi, 301)

lam_val = 1.5
z_bp = np.arccos((-1.0+lam_val)/(0.5*lam_val))
bc = 'wall'
int_mode = 'fast'

B_z = B(z)
h_z = h(z)

func_mult = 2.0

h_z = [h_z, func_mult*h_z]

print('Testing bounce_int_lambda')
dict = bounce_int_lambda(B, [h, lambda z: func_mult*h(z)], z, lam_val, mode=int_mode, boundary_condition=bc)
print(dict)
print('Diff:', np.asarray(dict['integrals']) - t_bounce_exact(lam_val))

print('Testing bounce_int_lambda array')
dict = bounce_int_lambda(B_z, h_z, z, lam_val, mode=int_mode, boundary_condition=bc)
print(dict)
print('Diff:', np.asarray(dict['integrals']) - t_bounce_exact(lam_val))

print('Testing bounce_int_zbp')
dict = bounce_int_zbp(B, [h, lambda z: func_mult*h(z)], z, z_bp, mode=int_mode, boundary_condition=bc)
print(dict)
print('Diff:', np.asarray(dict['integrals']) - t_bounce_exact(lam_val))

print('Testing bounce_int_zbp array')
dict = bounce_int_zbp(B_z, h_z, z, z_bp, mode=int_mode, boundary_condition=bc)
print(dict)
print('Diff:', np.asarray(dict['integrals']) - t_bounce_exact(lam_val))
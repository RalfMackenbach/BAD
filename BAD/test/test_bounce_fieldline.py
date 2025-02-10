from BAD.bounce_fieldline import *
import numpy as np

def B(z):
    return 1.0 - 0.5 * np.cos(z)

def h(z):
    return np.ones_like(z)

z = np.linspace(-2*np.pi, +2*np.pi, 3001)

lam_val = 1.4
z_bp = -np.pi/2 - np.pi
bc = 'periodic'

B_z = B(z)
h_z = h(z)

print('Testing bounce_int_lambda')
dict = bounce_int_lambda(B, h, z, lam_val, mode='fast', boundary_condition=bc)
print(dict)

print('Testing bounce_int_lambda array')
dict = bounce_int_lambda(B_z, h_z, z, lam_val, mode='fast', boundary_condition=bc)
print(dict)

print('Testing bounce_int_zbp')
dict = bounce_int_zbp(B, h, z, z_bp, mode='fast', boundary_condition=bc)
print(dict)

print('Testing bounce_int_zbp array')
dict = bounce_int_zbp(B_z, h_z, z, z_bp, mode='fast', boundary_condition=bc)
print(dict)
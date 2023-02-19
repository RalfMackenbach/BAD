# this code shows how to extend the domain so thay the quasi-periodic
# boundary condition is easily enforced. This has been wrapped into
# the mag_reader module, but since this problem is not trivial
# we include the logic in this file. Hopefully this helps
# if one were to build such an extender from scratch


import sys
sys.path.append('/Users/ralfmackenbach/Documents/GitHub/Bounce-averaged-drift/BAD/src/')  
import mag_reader
import bounce_int
import numpy as np
import time
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib        as mpl
import scipy 
import pandas as pd


plot_quasiperiod = True

plt.close('all')

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10}

mpl.rc('font', **font)



def CHM_analytical(k2,s,alpha,q):
    ellipk = scipy.special.ellipk(k2, out=np.empty_like(k2))
    ellipe = scipy.special.ellipe(k2, out=np.empty_like(k2))
    G1 = ellipe/ellipk - 0.5
    G2 = ellipe/ellipk + k2 - 1
    G3 = 2/3 * ( ellipe/ellipk * (2 * k2 - 1) + 1 - k2 )
    return G1 - alpha/(4 * q**2) + 2 * s * G2 - alpha * G3



# read the gist data, include endpoint, and plot
data = mag_reader.mag_data("s7_alpha5.txt")
data.include_endpoint()
data.plot_geometry()





# make the arrays
L2              = data.L2
L1              = data.L1
g11             = data.g11
g12             = data.g12
modb            = data.modB
sqrtg           = data.sqrtg
theta           = data.theta
my_dpdx         = data.my_dpdx
gridpoints      = data.gridpoints




# we now extract the non-periodic part
# kappa_g can readily be extracted from L1
kappa_g = L1 / np.sqrt(g11)
# the cotangent of the angle is also readily found
cot_ang = g12 / modb
# construct the quasi-periodic term of L2
L2qsp =-kappa_g * cot_ang * modb/ np.sqrt(g11)
# subtracting the quasi-periodic part should result 
# in a periodic function
L2per = L2 - L2qsp




# plot to verify that the only quasi-periodic part is cot_ang
if plot_quasiperiod:
    c= 3.0
    fig, ax = plt.subplots(1, 3, tight_layout=True,figsize=(c*5.0,c*1.5))
    ax[0].plot(theta/np.pi,L2per)
    ax[0].set_title(r'$\frac{\mathbf{B} \times \nabla B \cdot \nabla \alpha}{B^2} + \kappa_g \frac{B}{|\nabla \psi|} \cot \vartheta_s$')
    ax[2].plot(theta/np.pi,cot_ang,color='tab:red')
    ax[2].set_title(r'$\cot \vartheta_s$')
    ax[1].plot(theta/np.pi,kappa_g * modb/ np.sqrt(g11),color='tab:green')
    ax[1].set_title(r'$\kappa_g\frac{B}{|\nabla \psi|}$')

    ax[0].set_xlabel(r'$\theta/\pi$')
    ax[1].set_xlabel(r'$\theta/\pi$')
    ax[2].set_xlabel(r'$\theta/\pi$')
    plt.show()

    plt.close('all')




# to enforce the quasiperiodic boundary condition we simply extend the domain
# we first find all the positions where the magnetic field is maximal
max_idx = np.asarray(np.argwhere(modb == np.amax(modb))).flatten()
l_max   = max_idx[0]
r_max   = max_idx[-1]


# now we extend the left and right sides of the domain
# we first focus on appending the various parts needed
cot_app = cot_ang[0:l_max+1]- cot_ang[0]
the_app = theta[0:l_max+1]  - theta[0]

# print to check if the first index is zero
print(cot_app[0])
print(cot_app[0])

# append! 
cot_ext     = np.append(cot_ang,cot_app[1::]+cot_ang[-1])
theta_ext   = np.append(theta,the_app[1::]+theta[-1])
g11_ext     = np.append(g11,g11[1:l_max+1])
modb_ext    = np.append(modb,modb[1:l_max+1])
kappa_g_ext = np.append(kappa_g,kappa_g[1:l_max+1])
L2per_ext   = np.append(L2per,L2per[1:l_max+1])

# now we focus on prepending the various parts needed
cot_pre = cot_ang[r_max::]- cot_ang[-1]
the_pre = theta[r_max::]  - theta[-1]

# print to check if the last index is zero
print(cot_pre[-1])
print(the_pre[-1])

# prepend! 
cot_ext     = np.concatenate((cot_pre[0:-1]+cot_ang[0],cot_ext))
theta_ext   = np.concatenate((the_pre[0:-1]+theta[0],theta_ext))
g11_ext     = np.concatenate((g11[r_max:-1],g11_ext))
modb_ext    = np.concatenate((modb[r_max:-1],modb_ext))
kappa_g_ext = np.concatenate((kappa_g[r_max:-1],kappa_g_ext))
L2per_ext   = np.concatenate((L2per[r_max:-1],L2per_ext))

# now construct the extended L2 
L2qsp_ext = -kappa_g_ext * cot_ext * modb_ext/ np.sqrt(g11_ext)
L2_ext = L2per_ext + L2qsp_ext

# plot!
plt.plot(theta_ext,L2_ext)
plt.show()

# extension looks spot on!
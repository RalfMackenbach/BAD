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


def CHM_analytical(k2,s,alpha,q):
    ellipk = scipy.special.ellipk(k2, out=np.empty_like(k2))
    ellipe = scipy.special.ellipe(k2, out=np.empty_like(k2))
    G1 = ellipe/ellipk - 0.5
    G2 = ellipe/ellipk + k2 - 1
    G3 = 2/3 * ( ellipe/ellipk * (2 * k2 - 1) + 1 - k2 )
    return G1 - alpha/(4 * q**2) + 2 * s * G2 - alpha * G3



# read the gist data
data = mag_reader.mag_data("s7_alpha5.txt")

# make the arrays
L2      = data.L2
modb    = data.modB
sqrtg   = data.sqrtg
theta   = data.theta
my_dpdx = data.my_dpdx

# since the arrays are not perfectly periodic, we enforce periodicity
L2      = np.append(L2,L2[0])
modb    = np.append(modb,modb[0])
sqrtg   = np.append(sqrtg,sqrtg[0])
delta_theta = theta[1]-theta[0]
theta   = np.append(theta,theta[-1]+delta_theta)


# we exclude the endpoint and starting point
lam_arr     = np.linspace(1/np.max(modb),1/np.min(modb),999,endpoint=False)
lam_arr     = np.delete(lam_arr,  0)
gtrapz_num  = []
gtrapz_den  = []
gtrapz_ave  = []
# now calculate for various lambda
for idx, lam_val in enumerate(lam_arr):
    f = 1.0 - lam_val*modb
    dldtheta = sqrtg/modb
    tau_b   = bounce_int.bounce_integral_wrapper(f,dldtheta,theta,is_func=False)
    alpha_arr = ( lam_val- 2 * (1/modb - lam_val) ) * L2 - my_dpdx * (1 - lam_val * modb)/modb**2
    d_alpha = bounce_int.bounce_integral_wrapper(f,alpha_arr * dldtheta,theta,is_func=False)
    gtrapz_den.append(np.asarray(tau_b))
    gtrapz_den.append(np.asarray(d_alpha))
    gtrapz_ave.append(list(np.asarray(d_alpha)/np.asarray(tau_b)))
# make into array
gtrapz_ave=np.asarray(gtrapz_ave)

# make k2 array
k2 = (modb.max() - lam_arr * modb.max() * modb.min())/(modb.max()-modb.min())

# calculate analytical result
conversion_fac = -2*np.sqrt(data.s0)/0.01
CHM_res = conversion_fac*CHM_analytical(k2,7,5,data.q0)

plt.close('all')

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10}

mpl.rc('font', **font)

fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(3.5, 1.8*2.5))
ax[0].plot(theta/np.pi,modb,color='black')
twinax0 = ax[0].twinx()
ax[0].set_xlabel(r'$\theta/\pi$')
ax[0].set_ylabel(r'$B  \quad \mathrm{[a.u.]}$')
twinax0.set_ylabel(r'$\frac{\mathbf{B} \times \nabla B \cdot \nabla \alpha}{B^2} \quad \mathrm{[a.u.]}$',color='tab:blue')
twinax0.plot(theta/np.pi,L2,color='tab:blue')
twinax0.plot(theta/np.pi,0.0*L2,color='red',linestyle='dashed')
twinax0.set_ylim(-L2.max(), 1.1*L2.max())
ax[0].set_xlim(-data.n_pol,data.n_pol)
ax[1].plot(k2,gtrapz_ave[:,1],label='CW')
ax[1].plot(k2,gtrapz_ave[:,0],label='EW',linestyle='dashed')
ax[1].plot(k2,CHM_res,linestyle='dotted',color='black',label='CHM')
ax[1].set_xlabel(r'$k^2$')
ax[1].set_xlim(0,1)
ax[1].set_ylabel(r'$\langle \mathbf{v}_D \cdot \nabla \alpha \rangle$')
ax[1].legend()
plt.savefig('CHM_direct_averaging.eps')
plt.show()
plt.semilogy(k2,np.abs(CHM_res-gtrapz_ave[:,1]))
plt.show()
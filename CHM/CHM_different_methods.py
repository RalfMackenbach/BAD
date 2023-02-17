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
lam_arr     = np.linspace(1/np.max(modb),1/np.min(modb),99,endpoint=False)
lam_arr     = np.delete(lam_arr,  0)
# make list holding drifts
gtrapz_num  = []
gtrapz_den  = []
gtrapz_ave  = []
# make interpolated functions
kind='cubic'
modb_interp     = interp.interp1d(theta, modb,                  kind=kind)
dldtheta_interp = interp.interp1d(theta, sqrtg*modb,            kind=kind)
L2_interp       = interp.interp1d(theta, L2,                    kind=kind)
K2_interp       = interp.interp1d(theta, L2 - my_dpdx/(2*modb), kind=kind)
# now calculate for various lambda
for idx, lam_val in enumerate(lam_arr):
    f = lambda x: 1.0 - lam_val*modb_interp(x)
    delta_alpha = lambda x: -1.0 * (lam_val * L2_interp(x) + 2 * (1/modb_interp(x) - lam_val) * K2_interp(x)) * dldtheta_interp(x)
    tau_b   = bounce_int.bounce_integral_wrapper(f,dldtheta_interp,theta,is_func=True)
    d_alpha = bounce_int.bounce_integral_wrapper(f,delta_alpha,theta,is_func=True)
    gtrapz_den.append(np.asarray(tau_b))
    gtrapz_den.append(np.asarray(d_alpha))
    gtrapz_ave.append(list(np.asarray(d_alpha)/np.asarray(tau_b)))
# make into array
gtrapz_ave=np.asarray(gtrapz_ave)

# make k2 array
k2 = (modb.max() - lam_arr * modb.max() * modb.min())/(modb.max()-modb.min())

# calculate analytical result
conversion_fac = 2*np.sqrt(data.s0)/0.01
CHM_res = conversion_fac*CHM_analytical(k2,7,5,data.q0)

plt.close('all')

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10}

mpl.rc('font', **font)

fig, ax = plt.subplots(3, 1, tight_layout=True, figsize=(3.5, 2.5*2.5))
ax[0].plot(theta/np.pi,modb,color='black')
twinax0 = ax[0].twinx()
ax[0].set_xlabel(r'$\theta/\pi$')
ax[0].set_ylabel(r'$B  \quad [B_0]$')
twinax0.set_ylabel(r'$\frac{\mathbf{B} \times \nabla B \cdot \nabla \alpha}{B^2} \quad [a^{-2}]$',color='tab:blue')
twinax0.plot(theta/np.pi,L2,color='tab:blue')
twinax0.plot(theta/np.pi,0.0*L2,color='red',linestyle='dashed')
twinax0.set_ylim(-L2.max(), 1.1*L2.max())
ax[0].set_xlim(-data.n_pol,data.n_pol)
ax[1].plot(k2,gtrapz_ave[:,1],label='CW')
ax[1].plot(k2,gtrapz_ave[:,0],label='EW',linestyle='dashed')
ax[1].plot(k2,CHM_res,linestyle='dotted',color='black',label='CHM')
ax[1].set_xlabel(r'$k^2$')
ax[1].set_xlim(0,1)
ax[1].set_ylabel(r'$ \langle \mathbf{v}_D \cdot \nabla \alpha \rangle \quad \left[ \frac{H}{q a^2 B_0} \right]$')
ax[1].legend()

## import data from joey
df = pd.read_table("CHMs7a5_mulitple_methods.dat", sep="\s+")
joey_dat = df.to_numpy()
k2      = joey_dat[:,0]
djdpsi  = 2*joey_dat[:,2]
CHM     = 2*joey_dat[:,4]


ax[2].plot(k2,djdpsi,label='CW')
ax[2].plot(k2,djdpsi,label='EW',linestyle='dashed')
ax[2].plot(k2,CHM,linestyle='dotted',color='black',label='CHM')
ax[2].set_xlabel(r'$k^2$')
ax[2].set_xlim(0,1)
ax[2].set_ylabel(r'$\langle \mathbf{v}_D \cdot \nabla \alpha \rangle \quad \left[ \frac{H}{q a^2 B_0} \right]$')
ax[2].legend()
ax[0].set_aspect('auto')
ax[1].set_aspect('auto')
ax[2].set_aspect('auto')


x_text = 0.1
theta_range = theta.max() - theta.min()
theta_val   =(theta.min() + x_text*theta_range)/np.pi
k_val        = x_text

y_text      = 0.05
modb_range  = modb.max() - modb.min()
modb_val    = modb.min() + modb_range*y_text 
w_range     = gtrapz_ave.max() - gtrapz_ave.min()
w_val       = gtrapz_ave.min() + w_range*y_text 
w_range2    = djdpsi.max() - djdpsi.min()
w2_val      = djdpsi.min() + w_range2*y_text


ax[0].text(theta_val,modb_val,r'(a)',   ha='center',va='center')
ax[1].text(k_val,w_val,r'(b)',          ha='center',va='center')
ax[2].text(1-k_val,w2_val,r'(c)',         ha='center',va='center')
plt.savefig('CHM_comparison.eps')

plt.show()
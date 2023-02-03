import sys
sys.path.append('/Users/ralfmackenbach/Documents/GitHub/Bounce-averaged-drift/BAD/src/')  
import gtrapz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib        as mpl


# we set numerical parameters here
lam_res = 10000



# import all data from NCSX
modb        = np.load('modb.npy')
grad_psi    = np.load('grad_psi.npy')
curv_psi    = np.load('curv_psi.npy')
grad_alpha  = np.load('grad_alpha.npy')
curv_alpha  = np.load('curv_alpha.npy')
jac         = np.load('jac.npy')
theta       = np.load('theta.npy')
dldtheta    = np.abs(modb/jac)
a_minor     = 0.32263403803766705
psi_edge    =-0.4970702058373358
flux_sign   =-1
Bref        = 1.5200136651694336
s_val = 0.5
drdpsi      = a_minor/(2* np.sqrt(s_val) * psi_edge)
dydalpha    = a_minor * np.sqrt(s_val)


# make lambda array
lam_arr     = np.linspace(1/modb.max(),1/modb.min(),1001,endpoint=False)
lam_arr     = np.delete(lam_arr, 0)
gtrapz_arr_alpha    = []
gtrapz_arr_psi      = []


# loop over all lambda values
for idx, lam_val in enumerate(lam_arr):
    num_arr_alpha           = flux_sign * ( (lam_val * grad_alpha - 2 * (1./modb - lam_val) * curv_alpha) * dydalpha * a_minor )
    num_alpha, den_alpha    = gtrapz.w_bounce(dldtheta,modb,num_arr_alpha,theta,lam_val)
    gtrapz_arr_alpha.append(num_alpha/den_alpha)
    num_arr_psi             = (lam_val * grad_psi - 2 * (1./modb - lam_val) * curv_psi) * drdpsi * a_minor
    num_psi, den_psi        = gtrapz.w_bounce(dldtheta,modb,num_arr_psi,theta,lam_val)
    gtrapz_arr_psi.append(num_psi/den_psi)





# Plottng Parameters #
plt.close('all')

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10}

mpl.rc('font', **font)

# make k2 array
k2_arr   = ((1 - lam_arr*np.amin(modb))*np.amax(modb)/(np.amax(modb)-np.amin(modb)) )
# some reshaping is necessary for the trapz method, 
# as it saves the bounce well in a list of lists.
walp_arr = np.nan*np.zeros([len(gtrapz_arr_alpha),len(max(gtrapz_arr_alpha,key = lambda x: len(x)))])
for i,j in enumerate(gtrapz_arr_alpha):
    walp_arr[i][0:len(j)] = j
wpsi_arr = np.nan*np.zeros([len(gtrapz_arr_psi),len(max(gtrapz_arr_psi,key = lambda x: len(x)))])
for i,j in enumerate(gtrapz_arr_psi):
    wpsi_arr[i][0:len(j)] = j
alp_l = np.shape(walp_arr)[1]
psi_l = np.shape(wpsi_arr)[1]
k2_psi = np.repeat(k2_arr,psi_l)
k2_alp = np.repeat(k2_arr,alp_l)

# make scatter plot
fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(3.5, 5.0))
ax[0].scatter(k2_alp,walp_arr,s=0.2,marker='.',color='black',label='g-trapz')
ax[1].scatter(k2_psi,wpsi_arr,s=0.2,marker='.',color='black',label='g-trapz')
ax[0].set_xlabel(r'$k^2$')
ax[0].set_xlim(0,1)
ax[0].set_ylabel(r'$\langle \hat{\mathbf{v}}_D \cdot \nabla y \rangle$')
ax[1].set_xlabel(r'$k^2$')
ax[1].set_xlim(0,1)
ax[1].set_ylabel(r'$\langle \hat{\mathbf{v}}_D \cdot \nabla r \rangle$')
ax[1].legend()

plt.savefig('precession_NCSX.png',dpi=1000)
plt.show()
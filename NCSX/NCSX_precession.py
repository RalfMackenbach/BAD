import sys
sys.path.append('/Users/ralfmackenbach/Documents/GitHub/Bounce-averaged-drift/BAD/src/')  
import bounce_int
import numpy as np
import matplotlib.pyplot as plt
import matplotlib        as mpl
from scipy import interpolate as interp


# we set numerical parameters here
lam_res = 10000

# constants
a_minor     = 0.32263403803766705
psi_edge    =-0.4970702058373358 / ( 2 * np.pi)
flux_sign   =-1
Bref        = 1.5200136651694336
s_val       = 0.5

# import all data from NCSX
modB_SI        = np.load('modb.npy')
modb = modB_SI/Bref
grad_psi    = np.load('grad_psi.npy')
curv_psi    = np.load('curv_psi.npy')
grad_alpha  = np.load('grad_alpha.npy')
curv_alpha  = np.load('curv_alpha.npy')
jac         = np.load('jac.npy')
theta       = np.load('theta.npy')
dldtheta    = np.abs(modb/jac)

rho = np.sqrt(s_val)
drdpsi      = a_minor/(2* np.sqrt(s_val) * psi_edge)
dydalpha    = a_minor * np.sqrt(s_val)


# make lambda array
lam_arr     = np.linspace(1/modb.max(),1/modb.min(),lam_res,endpoint=False)
lam_arr     = np.delete(lam_arr, 0)
gtrapz_arr_alpha    = []
gtrapz_arr_psi      = []
boundary_list = []


# gtrapz is good enough here


# loop over all lambda values
for idx, lam_val in enumerate(lam_arr):
    f = 1 - lam_val * modb
    tau_b_arr               = dldtheta
    bounce_time, roots      = bounce_int.bounce_integral_wrapper(f,tau_b_arr,theta,return_roots=True)
    alpha_arr               = dldtheta * ( lam_val * grad_alpha + 2 * ( 1/modb - lam_val ) * curv_alpha )
    num_arr_alpha           = bounce_int.bounce_integral_wrapper(f,alpha_arr,theta)
    psi_arr                 = dldtheta * ( lam_val * grad_psi   + 2 * ( 1/modb - lam_val ) * curv_psi )
    num_arr_psi             = bounce_int.bounce_integral_wrapper(f,psi_arr,theta)
    # check if roots cross boundary
    cross_per_lam           = []
    for idx2 in range(int(len(roots)/2)):
        boundary_cross = roots[2*idx2] > roots[2*idx2+1]
        cross_per_lam.append(boundary_cross)
    # make into list of lists
    boundary_list.append(np.asarray(cross_per_lam))
    gtrapz_arr_psi.append(np.asarray(num_arr_psi)/np.asarray(bounce_time))
    gtrapz_arr_alpha.append(np.asarray(num_arr_alpha)/np.asarray(bounce_time))






# Plottng Parameters #
plt.close('all')

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10}

mpl.rc('font', **font)

# make k2 array
k2_arr   = ((1 - lam_arr*np.amin(modb))*np.amax(modb)/(np.amax(modb)-np.amin(modb)) )
# some reshaping is necessary, 
# as it saves the bounce wells in a list of lists.
walp_arr = np.nan*np.zeros([len(gtrapz_arr_alpha),len(max(gtrapz_arr_alpha,key = lambda x: len(x)))])
for i,j in enumerate(gtrapz_arr_alpha):
    walp_arr[i][0:len(j)] = j
wpsi_arr = np.nan*np.zeros([len(gtrapz_arr_psi),len(max(gtrapz_arr_psi,key = lambda x: len(x)))])
for i,j in enumerate(gtrapz_arr_psi):
    wpsi_arr[i][0:len(j)] = j
# make masks for boundary and centre
mask_bound = np.nan*np.zeros([len(boundary_list),len(max(gtrapz_arr_psi,key = lambda x: len(x)))])
mask_centr = np.nan*np.zeros([len(boundary_list),len(max(gtrapz_arr_psi,key = lambda x: len(x)))])
for i,j in enumerate(boundary_list):
    mask_centr[i][0:len(j)] = j
    mask_bound[i][0:len(j)] = np.logical_not(j)
alp_l = np.shape(walp_arr)[1]
psi_l = np.shape(wpsi_arr)[1]
k2_psi = np.repeat(k2_arr,psi_l)
k2_alp = np.repeat(k2_arr,alp_l)




# rescale arrays in line with dimensions of manuscript
walp_arr    = a_minor * dydalpha * walp_arr
wpsi_arr    = a_minor * drdpsi * wpsi_arr

# make masks
mask_centr[mask_centr == False] = np.nan
mask_bound[mask_bound == False] = np.nan
print(mask_bound)

# make scatter plot
fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(3.5, 5.0))
ax[0].scatter(k2_alp,mask_bound*walp_arr,s=0.2,marker='.',color='black',label='g-trapz')
ax[0].scatter(k2_alp,mask_centr*walp_arr,s=0.2,marker='.',color='red',label='g-trapz')
ax[1].scatter(k2_psi,mask_bound*wpsi_arr,s=0.2,marker='.',color='black',label='g-trapz')
ax[1].scatter(k2_psi,mask_centr*wpsi_arr,s=0.2,marker='.',color='red',label='g-trapz')
ax[0].set_xlabel(r'$k^2$')
ax[0].set_xlim(0,1)
ax[0].set_ylabel(r'$\langle \hat{\mathbf{v}}_D \cdot \nabla y \rangle$')
ax[1].set_xlabel(r'$k^2$')
ax[1].set_xlim(0,1)
ax[1].set_ylabel(r'$\langle \hat{\mathbf{v}}_D \cdot \nabla r \rangle$')

plt.savefig('precession_NCSX.eps',dpi=1000)
plt.show()
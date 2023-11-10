import numpy as np
import matplotlib.pyplot as plt
import matplotlib        as mpl


# import the data
modb        = np.load('modb.npy')
grad_psi    = np.load('grad_psi.npy')
curv_psi    = np.load('curv_psi.npy')
grad_alpha  = np.load('grad_alpha.npy')
curv_alpha  = np.load('curv_alpha.npy')
theta       = np.load('theta.npy')
a_minor     = 0.32263403803766705
psi_edge    =-0.4970702058373358
flux_sign   =-1
Bref        = 1.5200136651694336
s_val = 0.5
drdpsi      = a_minor/(2* np.sqrt(s_val) * psi_edge)
dydalpha    = a_minor * np.sqrt(s_val)


# rescale theta
theta = theta/np.pi


# find maximal values 
max_idx = np.asarray(np.argwhere(modb == np.amax(modb))).flatten()
l_max   = max_idx[0]
r_max   = max_idx[-1]

# Plottng Parameters #
plt.close('all')

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10}

mpl.rc('font', **font)


theta_min = theta[l_max]
theta_max = theta[r_max]

fig, ax1 = plt.subplots(1, 1, tight_layout=True, figsize=(3.5, 2.5))
ax1.plot(theta,modb,color='black')
ax1.set_xlim(theta_min,theta_max)
ax1.set_xlabel(r'$\theta/\pi$')
ax1.set_ylabel(r'$B \quad [\mathrm{T}]$')
ax2 = ax1.twinx()
ax2.plot(theta,-grad_alpha)
ax2.plot(theta,0.0*grad_alpha,color='red',linestyle='dashed')
ax2.set_ylim(-5,10)
ax1.set_zorder(1)
ax2.set_ylabel(r'$\frac{\mathbf{B} \times \nabla B \cdot \nabla \alpha}{B^2} \quad [\mathrm{m}^{-2}]$',color='tab:blue')
ax1.patch.set_visible(False)

plt.savefig('modb_ncsx.png',dpi=1000)
plt.show()
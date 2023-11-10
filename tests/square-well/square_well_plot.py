import numpy as np
import matplotlib.pyplot as plt
import matplotlib        as mpl


# B hat from the square well example
def modb(l,al=1.0):
    return 1 + al * l**2

# radial derivative of modB of square well example
def dbdpsi(l,apsi=1.5,b=1.0):
    return apsi * ( l**2 - b )

# analytical bounce-time
def bounce_time(lam_val,al=1.0):
    return np.pi/np.sqrt(lam_val*al)

# analytical Delta alpha
def delta_alpha(lam_val,al=1.0,apsi=1.0,b=1.0):
    return np.pi * apsi * (lam_val + 2 * lam_val * b * al - 1) / ( 2 * al**1.5 * lam_val**0.5 )

# analytical bounce averaged drift
def drift(lam_val,al=1.0,apsi=1.0,b=1.0):
    return apsi * ( 2 * b * al * lam_val + lam_val - 1.0 ) / ( 2 * al )

# we set numerical parameters here
l_res = 10000

l_arr = np.linspace(-2,2,l_res)




# Plottng Parameters #
plt.close('all')

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10}

mpl.rc('font', **font)

fig, ax1 = plt.subplots(1, 1, tight_layout=True, figsize=(3.5, 2.5))
ax1.plot(l_arr,modb(l_arr),color='black')
ax1.set_xlim(-2,2)
ax1.set_xlabel(r'$\hat{\ell}$')
ax1.set_ylabel(r'$\hat{B}$')
ax2 = ax1.twinx()
ax2.plot(l_arr,dbdpsi(l_arr))
ax2.plot(l_arr,0.0*dbdpsi(l_arr),color='red',linestyle='dashed')
ax2.set_ylim(-2,2)
ax1.set_zorder(1)
ax2.set_ylabel(r'$\partial_{\hat{\psi}} \hat{B}$',color='tab:blue')
pad = 0.1
ax2.annotate('', xy=(0,0+pad), xytext=(0,-1.5-pad), 
             arrowprops=dict(color='darkgreen',facecolor='darkgreen', lw=1,arrowstyle='<|-|>'))
ax2.text(0.3,-1.5/2,r'$b a_\psi$',ha='center',va='center',color='darkgreen')
ax1.patch.set_visible(False)

plt.savefig('square_well_plot.png',dpi=1000)
plt.show()
import gtrapz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib        as mpl


# B hat from the square well example
def modb(l,al=1.0):
    return 1 + al * l**2

# radial derivative of modB of square well example
def dbdpsi(l,apsi=1.0,b=1.0):
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
theta_res = 10000
lam_res = 10000

# l is field-line following coordinate
l_arr       = np.linspace(-10,10,theta_res)
# dl/dl is unity
h_arr       = np.ones_like(l_arr)
# mod b
modb_arr    = modb(l_arr)
# dbdpsi 
dbdpsi_arr  = dbdpsi(l_arr)


# make array with lambdas and bounce-averaged quantities
# we exclude the endpoint and starting point
lam_arr     = np.linspace(1/np.max(modb_arr),1/np.min(modb_arr),lam_res,endpoint=False)
lam_arr     = np.delete(lam_arr, 0)
gtrapz_num  = np.empty_like(lam_arr)
gtrapz_den  = np.empty_like(lam_arr)
gtrapz_ave  = np.empty_like(lam_arr)


for idx, lam_val in enumerate(lam_arr):
    num, den = gtrapz.w_bounce(h_arr,modb_arr,-dbdpsi_arr*lam_val,l_arr,lam_val)
    gtrapz_num[idx] = num
    gtrapz_den[idx] = den
    gtrapz_ave[idx] = num/den




# Plottng Parameters #
plt.close('all')

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 18}

mpl.rc('font', **font)

fig, ax1 = plt.subplots(1, 1, tight_layout=True)
true_res = drift(lam_arr)
diff_gtrapz = np.abs(gtrapz_ave-true_res)
ax1.semilogy(1-lam_arr,diff_gtrapz,label='gtrapz')
ax1.set_ylabel(r'Absolute difference')
ax1.set_xlabel(r'$k^2$')
ax1.set_xlim(0,1)

plt.show()
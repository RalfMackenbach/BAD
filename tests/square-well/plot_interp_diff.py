import sys
from BAD import bounce_int
import numpy as np
import time
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib        as mpl


# B hat from the square well example
def modb(l,al=1.0):
    return 1 + al * l**2

# make low res array for interpolation
l_interp = np.linspace(-1,1,4)
modb_interp = modb(l_interp)

# now construct the interpolation function. Using cubic monotonic spline
modb_interp_func = interp.PchipInterpolator(l_interp,modb_interp)

# now make high res array for plotting
l_plot = np.linspace(-1,1,1001)
modb_plot = modb_interp_func(l_plot)

# also plot the true function
modb_true = modb(l_plot)

# plot
fig, ax = plt.subplots(1,1,figsize=(6,4),constrained_layout=True)
ax.plot(l_plot,modb_plot,'k',label='m-quad')
ax.plot(l_plot,modb_true,'r',label='true function')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$B$')

# also plot the interpolation points
ax.scatter(l_interp,modb_interp,s=15,c='k',zorder=10,label='interp points')

ax.legend()

# save
plt.savefig('modb_interp.png',dpi=300)

plt.show()

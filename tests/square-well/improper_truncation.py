import numpy as np
import matplotlib.pyplot as plt

# let us first construct a simple periodic function

# define the function
def f(x):
    return -np.cos(x)

# now let us construct the grid
x = np.linspace(-np.pi,np.pi,1001)

# now let us evaluate the function on the grid
f1 = f(x)
f2 = f(x*np.sqrt(2))

# let us plot the function three times: once from -3pi to -pi, once from -pi to pi, and once from pi to 3pi
# to this end, let us first construct the grid
x_left = x - 2*np.pi
x_right = x + 2*np.pi

# now let us now plot
# make the sqrt(2) plot red and dashed, the others black and solid
# we plot 
fig, ax = plt.subplots(1,1,figsize=(6,4),constrained_layout=True)
ax.plot(x_left,f1,'g',linestyle='dashed')
ax.plot(x,f1,'g')
ax.plot(x_right,f1,'g',linestyle='dashed')
ax.plot(x_left,f2,'r',linestyle='dashed')
ax.plot(x,f2,'r')
ax.plot(x_right,f2,'r',linestyle='dashed')

# make legend saying dashed is enforced periodicity, solid is the function
ax.legend(['enforced periodicity','function'],loc='upper left')


ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f(x)$')

# save
plt.savefig('periodic_function.png',dpi=1000)

plt.show()
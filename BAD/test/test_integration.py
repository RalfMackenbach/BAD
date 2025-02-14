from BAD.util_integrals import bounce_integral_fn, bounce_integral_discrete
import numpy as np
from scipy.special import ellipk
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import rc
import timeit


rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 16})
rc('text', usetex=True)


def B(z):
    return 1.0 - 0.5 * np.cos(z)

def h(z):
    return np.ones_like(z)

def exact_bounce_time(lam_grid, Delta = 0.5):
    # k parameter
    k = np.sqrt(0.5*(1-lam_grid*(1-Delta))/lam_grid/Delta)

    # Bounce time
    tb = 2 * np.sqrt(2/lam_grid/Delta) * ellipk(k*k)

    return tb    

lam = 0.7
x_l = -np.arccos((1-1/lam)/0.5)
x_r = -x_l
h = lambda x: 1 + 0*x
f = lambda x: 1 - lam * B(x)

## TRY ##
N_arr = np.linspace(10, 1000, 51, dtype=int)
data = []
# Create a dictionary to store the execution times
execution_times = {
    "tak": [],
    "cg1": [],
    "cg2": [],
    "clem": [],
    "GL": [],
    "q": [],
    "q2": [],
    "q4": [],
    "gtrapz": [],
    "gquadz": []
}
anal = exact_bounce_time(lam)
for N in N_arr:
    start = timeit.default_timer()
    tak = bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, N=N, method="quad", order=1, mapping="takashi", scale=4.0)
    execution_times["tak"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    cg1 = bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, N=N, method="chebgauss1", order=1, mapping="sin")
    execution_times["cg1"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    cg2 = bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, N=N, method="chebgauss2", order=1, mapping="sin")
    execution_times["cg2"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    cc = bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, N=N, method="clenshaw", order=1, mapping="sin")
    execution_times["clem"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    GL = bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, N=N, method="GL", order=1, mapping="sin")
    execution_times["GL"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    q = bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, N=N, method="quad", order=1, mapping="sin", trim=True)
    execution_times["q"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    q2 = bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, N=N, method="quad", order=2, mapping="sin", trim=True)
    execution_times["q2"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    q4 = bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, N=N, method="quad", order=4, mapping="sin", trim=True)
    execution_times["q4"].append(timeit.default_timer() - start)

    # Spatial grid 
    ell = np.linspace(x_l, x_r, N)

    start = timeit.default_timer()
    gtrapz = bounce_integral_discrete(f(ell), np.ones(N), ell, method = "gtrapz")
    execution_times["gtrapz"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    gquadz = bounce_integral_discrete(f(ell), np.ones(N), ell, method = "gquadz")
    execution_times["gquadz"].append(timeit.default_timer() - start)


    data.append([tak, cg1, cg2, cc, GL, q, q2, q4, gtrapz, gquadz])
    print(anal,tak, cg1, cg2, cc, GL, q, q2, q4, gtrapz, gquadz)

quad_quad = quad(lambda x: h(x)/np.sqrt(f(x)), x_l, x_r, epsrel=1e-12)

names = ["Takashi", "CG1-sin", "CG2-sin", "CC-sin", "GL-sin", "Trapz-sin", "Simpson-sin", "Boole-sin", "gtrapz", "gquadz"]
data = np.array(data)
for j in range(np.shape(data)[1]):
    plt.plot(N_arr, np.abs(data[:,j]-anal), '.-', label = names[j])
    print(np.abs(data[:,j]-anal))
plt.xlabel(r'$N$')
plt.ylabel(r'$\Delta f$')
plt.yscale('log')
plt.legend(ncols = 2, loc = "upper right", fontsize = 15)
plt.tight_layout()
plt.show()

plt.figure()
for j in range(np.shape(data)[1]):
    plt.plot(N_arr, execution_times[list(execution_times.keys())[j]], '.-', label = names[j])
plt.yscale('log')
plt.legend(ncol = 2, loc = "upper left", fontsize = 12)
plt.show()


from BAD.util_integrals import cum_bounce_integral_fn, cum_bounce_integral_discrete
import numpy as np
from scipy.special import ellipk, ellipkinc
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import rc
import timeit


rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 16})
rc('text', usetex=True)


def exact_bounce_time(Delta, lam_grid):
    # k parameter
    k = np.sqrt(0.5*(1-lam_grid*(1-Delta))/lam_grid/Delta)

    # Bounce time
    tb = 2/np.pi * np.sqrt(2/lam_grid/Delta) * ellipk(k*k)

    return tb

# # B field
# Delta = 0.1
# def B(x, derivative = 0):
#     if derivative == 0:
#         return 1 - Delta * np.cos(np.pi * x)
#     elif derivative == 1:
#         return Delta * np.pi * np.sin(np.pi * x)
#     if derivative == 2:
#         return Delta *np.pi*np.pi * np.cos(np.pi * x)
#     if derivative == 3:
#         return -Delta * np.pi*np.pi*np.pi * np.sin(np.pi * x)
#     else:
#         raise ValueError('Invalid derivative specified. Note that only up to 3rd derivative is implemented!')

# ## DATA ##
# B_max = 1 + Delta
# B_min = 1 - Delta

# lam = 1.0
# x_l = -0.5
# x_r = 0.5

k = 0.9
h = lambda x: 1 + 0*x
f = lambda x: 1 - k*k * np.sin(x)**2

## TRY ##
k_arr = [5, 10, 20, 50, 100, 200, 500, 2000]
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
    "q_edge": [],
    "gtrapz": [],
}

N = 100
x_out = np.linspace(0.0, 1.0, N) * 0.5*np.pi
x_l = x_out[0]
x_r = x_out[-1]
anal = lambda x: ellipkinc(x, k*k)
N_time = 10

for k_coeff in k_arr:
    start = timeit.default_timer()
    for i in range(N_time):
        features = {"method": "quad", "which_sub": 0, "order": 1, "k": k_coeff, "mapping": "takashi", "scale": 3.0, "trim": False}
        tak = cum_bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, x_out = x_out, approach = "subdomains", features = features)
    execution_times["tak"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    for i in range(N_time):
        features = {"method": "chebgauss1", "which_sub": 0, "order": 1, "k": k_coeff, "mapping": "sin", "scale": 3.0, "trim": False}
        cg1 = cum_bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, x_out = x_out, approach = "subdomains", features = features)
    execution_times["cg1"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    for i in range(N_time):
        features = {"method": "chebgauss2", "which_sub": 0, "order": 1, "k": k_coeff, "mapping": "sin", "scale": 3.0, "trim": False}
        cg2 = cum_bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, x_out = x_out, approach = "subdomains", features = features)
    execution_times["cg2"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    for i in range(N_time):
        features = {"method": "clenshaw", "which_sub": 0, "order": 1, "k": k_coeff, "mapping": "sin", "scale": 3.0, "trim": False}
        cc = cum_bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, x_out = x_out, approach = "subdomains", features = features)
    execution_times["clem"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    for i in range(N_time):
        features = {"method": "GL", "which_sub": 0, "order": 1, "k": k_coeff, "mapping": "sin", "scale": 3.0, "trim": False}
        GL = cum_bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, x_out = x_out, approach = "subdomains", features = features)
    execution_times["GL"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    for i in range(N_time):
        features = {"method": "quad", "which_sub": 0, "order": 1, "k": k_coeff, "mapping": "sin", "scale": 3.0, "trim": True}
        q = cum_bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, x_out = x_out, approach = "subdomains", features = features)
    execution_times["q"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    for i in range(N_time):
        features = {"method": "quad", "which_sub": 0, "order": 2, "k": k_coeff, "mapping": "sin", "scale": 3.0, "trim": True}
        q2 = cum_bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, x_out = x_out, approach = "subdomains", features = features)
    execution_times["q2"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    for i in range(N_time):
        features = {"method": "quad", "which_sub": 0, "order": 4, "k": k_coeff, "mapping": "sin", "scale": 3.0, "trim": True}
        q4 = cum_bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, x_out = x_out, approach = "subdomains", features = features)
    execution_times["q4"].append(timeit.default_timer() - start)

    start = timeit.default_timer()
    for i in range(N_time):
        features = {"method": ["quad", "quad", "quad"], "which_sub": [0, 0, N-2], "order": [2, 2, 2], \
                    "k": [k_coeff, k_coeff, k_coeff], "mapping": ["normal","takashi","takashi"], \
                    "scale": [3.0, 3.0, 3.0], "trim": [False, False, False]}
        q_edge = cum_bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, x_out = x_out, approach = "subdomains", features = features)
    execution_times["q_edge"].append(timeit.default_timer() - start)

    # Spatial grid 
    ell = np.linspace(x_l, x_r, N*k_coeff)

    start = timeit.default_timer()
    for i in range(N_time):
        gtrapz = cum_bounce_integral_fn(f, h, x_l, x_r, ell, approach = "discrete", features = {"method": "gtrapz"})
        # gtrapz = cum_bounce_integral_discrete(f(ell), np.ones(N*k_coeff), ell, method = "gtrapz")
    execution_times["gtrapz"].append(timeit.default_timer() - start)
    # q_d = bounce_integral_discrete(1 - lam*B(ell), np.ones(N), ell, method = "trapz")
    data.append([tak, cg1, cg2, cc, GL, q, q2, q4, q_edge, gtrapz])
    # print(anal,tak, cg1, cg2, cc, GL, q, q2, q4)

# quad_quad = quad(lambda x: h(x)/np.sqrt(f(x)), x_l, x_r, epsrel=1e-12)

names = ["Takashi", "CG1-sin", "CG2-sin", "CC-sin", "GL-sin", "Trapz-sin", "Simpson-sin", "Boole-sin", "Simpson_mixed", "gtrapz"]

for j in range(len(data[0])):
    error = [np.sqrt(np.mean(np.square(np.abs(data[ind][j][1]-anal(data[ind][j][0]))))) for ind in range(len(k_arr))]
    plt.plot(k_arr, error, '.-', label = names[j])
    print(error)

plt.xlabel(r'$k$')
plt.ylabel(r'$\Delta f$')
plt.yscale('log')
plt.legend(ncols = 2, loc = "upper right", fontsize = 15)
plt.tight_layout()
plt.show()

for j in range(len(data[0])):
    plt.plot(data[4][j][0], data[4][j][1], '.-', label = names[j])
plt.plot(data[4][j][0], anal(data[4][j][0]), '.-', label = "Anal")
plt.xlabel(r'$x$')
plt.ylabel(r'$f$')
plt.legend(ncols = 2, loc = "upper right", fontsize = 15)
plt.tight_layout()
plt.show()

plt.figure()
for j in range(len(data[0])):
    plt.plot(k_arr, np.array(execution_times[list(execution_times.keys())[j]])/N_time, '.-', label = names[j])
plt.yscale('log')
plt.legend(ncol = 2, loc = "upper left", fontsize = 12)
plt.show()


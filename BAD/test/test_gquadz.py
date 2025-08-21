from BAD.util_integrals import bounce_integral_discrete
import numpy as np
from scipy.special import ellipk
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import rc
import timeit


rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 16})
rc('text', usetex=True)


def main():
    k = 0.25
    x_l = 0
    x_r = 0.5*np.pi
    h = lambda x: 1 + 0*x
    f = lambda x: 1 - k*k * np.sin(x)**2

    anal = ellipk(k*k)

    N_arr = np.linspace(10, 4000, 50, dtype=int)
    data = []
    # Create a dictionary to store the execution times
    execution_times = {
        "gtrapz": [],
        "gquadz": [],
    }
    anal = ellipk(k*k)
    for N in N_arr:
        # Spatial grid 
        ell = np.linspace(x_l, x_r, N)

        start = timeit.default_timer()
        gtrapz = bounce_integral_discrete(f(ell), np.ones(N), ell, method = "gtrapz")
        execution_times["gtrapz"].append(timeit.default_timer() - start)

        start = timeit.default_timer()
        gquadz = bounce_integral_discrete(f(ell), np.ones(N), ell, method = "gquadz")
        execution_times["gquadz"].append(timeit.default_timer() - start)

        data.append([gtrapz, gquadz])
        print(anal, gtrapz, gquadz)

    names = ["gtrapz", "gquadz"]
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








if __name__ == "__main__":
    main()
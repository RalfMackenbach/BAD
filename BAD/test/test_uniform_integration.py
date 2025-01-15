import numpy as np
from scipy.special import erf, eval_chebyt, eval_chebyu, eval_legendre

from BAD.util_integrals import newton_cotes_weights_uniform, chebgauss1, chebgauss2, leggauss, clenshaw_curtis

# Function for numerical integration using Newton-Cotes weights
def numerical_integration(func, x, w):
    """
    Perform numerical integration using the given function, grid points, and weights.

    Parameters:
        func (callable): Function to integrate.
        x (array-like): Grid points.
        w (array-like): Weights.

    Returns:
        float: Approximation of the integral.
    """
    return np.sum(func(x) * w)

# Test cases

# Function for numerical integration using Newton-Cotes weights
def numerical_integration(func, x, w):
    """
    Perform numerical integration using the given function, grid points, and weights.

    Parameters:
        func (callable): Function to integrate.
        x (array-like): Grid points.
        w (array-like): Weights.

    Returns:
        float: Approximation of the integral.
    """
    return np.sum(func(x) * w)

# Test cases
def test_trapezoidal_rule():
    """Test the trapezoidal rule (order 1)."""
    N = 10
    order = 1
    func = lambda x: x  # Linear function
    exact_integral = 0  # Integral of x from -1 to 1

    x, w = newton_cotes_weights_uniform(N, order)
    numerical_result = numerical_integration(func, x, w)

    assert np.isclose(numerical_result, exact_integral, atol=1e-6), \
        f"Trapezoidal rule failed: {numerical_result} != {exact_integral}"

def test_simpsons_rule():
    """Test Simpson's rule (order 2)."""
    N = 11  # N must be odd for Simpson's rule
    order = 2
    func = lambda x: x**3  # Cubic function
    exact_integral = 0  # Integral of x^3 from -1 to 1

    x, w = newton_cotes_weights_uniform(N, order)
    numerical_result = numerical_integration(func, x, w)

    assert np.isclose(numerical_result, exact_integral, atol=1e-6), \
        f"Simpson's rule failed: {numerical_result} != {exact_integral}"

def test_large_N_non_trivial_function():
    """Test with a large N and a non-trivial function."""
    N = 20000
    func = lambda x: np.exp(-x**2)  # Gaussian function
    exact_integral = np.sqrt(np.pi) * (erf(1) - erf(-1)) / 2  # Approximation of the integral of exp(-x^2) from -1 to 1

    orders = [1, 2, 4]
    for order in orders:
        x, w = newton_cotes_weights_uniform(N, order)
        numerical_result = numerical_integration(func, x, w)
        assert np.isclose(numerical_result, exact_integral, atol=1e-6), \
            f"Large N test failed at order {order}: {numerical_result} != {exact_integral}"
        
# Integration test examples
def test_chebgauss1():
    """Test the chebgauss1 function with known analytic integrals."""

    # Example 1: Integrating a Chebyshev polynomial T_n(x) with the correct weight function
    N = 100
    n_array = [0, 3, 4, 5, 6, 10]
    for n in n_array:
        def f_cheb(x):
            return eval_chebyt(n, x)/np.sqrt(1-x**2)/np.pi

        x, w = chebgauss1(N)
        integral = np.sum(w * f_cheb(x))
        # Exact integral of T_n(x) for n > 0 is 0 due to orthogonality of Chebyshev polynomials
        expected_integral = 0.0 if n > 0 else 1.0  # Handle the special case of T_0(x) = 1
        assert np.isclose(integral, expected_integral, atol=1e-12), f"Failed Chebyshev polynomial test order {n}: {integral} != {expected_integral}"

    # Example 2: Integrating a constant function f(x) = 1
    N = 1000
    x, w = chebgauss1(N)
    integral = np.sum(w * 1)  # Exact integral of f(x) = 1 over [-1, 1] is 2
    assert np.isclose(integral, 2.0, atol=1e-12), f"Failed constant function test : {integral} != {2.0}"

    # Example 3: Integrating f(x) = x^2
    N = 1000
    def f(x):
        return x**2

    x, w = chebgauss1(N)
    integral = np.sum(w * f(x))  # Exact integral of x^2 over [-1, 1] is 2/3
    assert np.isclose(integral, 2 / 3, atol=1e-12), "Failed x^2 test"

def test_chebgauss2():
    """Test the chebgauss2 function with known analytic integrals."""

    # Example 1: Integrating a constant function f(x) = 1
    N = 1000
    x, w = chebgauss2(N)
    integral = np.sum(w * 1)  # Exact integral of f(x) = 1 over [-1, 1] is 2
    assert np.isclose(integral, 2.0, atol=1e-12), f"Failed constant function test: : {integral} != {2.0}"

    # Example 2: Integrating f(x) = x^2
    N = 1000
    def f(x):
        return x**2

    x, w = chebgauss2(N)
    integral = np.sum(w * f(x))  # Exact integral of x^2 over [-1, 1] is 2/3
    assert np.isclose(integral, 2 / 3, atol=1e-12), f"Failed x^2 test: {integral} != {2/3}"

    # Example 3: Integrating Chebyshev polynomials U_n(x) for different n
    N = 10
    for n in range(6):  # Test for n = 0, 1, 2, ..., 5
        def f_cheb2(x):
            return eval_chebyu(n, x) * np.sqrt(1 - x**2) * 2/np.pi  # Second kind polynomials include weight sqrt(1-x^2)

        x, w = chebgauss2(N)
        integral = np.sum(w * f_cheb2(x))
        # Exact integral of U_n(x) for n > 0 is 0 due to orthogonality of Chebyshev polynomials of the second kind
        expected_integral = 0.0 if n > 0 else 1.0  # Handle the special case of U_0(x) = 1
        assert np.isclose(integral, expected_integral, atol=1e-12), f"Failed Chebyshev second kind polynomial test for n={n}: {integral} != {expected_integral}"

def test_leggaus():
    """Test the leggaus function with known analytic integrals."""
    # Example 1: Integrating a constant function f(x) = 1
    N = 10
    x, w = leggauss(N)
    integral = np.sum(w * 1)  # Exact integral of f(x) = 1 over [-1, 1] is 2
    assert np.isclose(integral, 2.0, atol=1e-12), "Failed constant function test for leggaus"

    # Example 2: Integrating f(x) = x^2
    def f(x):
        return x**2

    x, w = leggauss(N)
    integral = np.sum(w * f(x))  # Exact integral of x^2 over [-1, 1] is 2/3
    assert np.isclose(integral, 2 / 3, atol=1e-12), "Failed x^2 test for leggaus"

    # Example 3: Integrating Legendre polynomials P_n(x) for different n
    for n in range(6):  # Test for n = 0, 1, 2, ..., 5
        def f_leg(x):
            return eval_legendre(n, x)

        x, w = leggauss(N)
        integral = np.sum(w * f_leg(x))
        # Exact integral of P_n(x) for n > 0 is 0 due to orthogonality of Legendre polynomials
        expected_integral = 0.0 if n > 0 else 2.0  # Handle the special case of P_0(x) = 1
        assert np.isclose(integral, expected_integral, atol=1e-12), f"Failed Legendre polynomial test for n={n}"

    # Example 4: Integrating f(x) = exp(-x^2)
    N = 100
    func = lambda x: np.exp(-x**2)  # Gaussian function
    exact_integral = np.sqrt(np.pi) * (erf(1) - erf(-1)) / 2  # Approximation of the integral of exp(-x^2) from -1 to 1
    x, w = leggauss(N)
    numerical_result = np.sum(w * func(x))
    assert np.isclose(numerical_result, exact_integral, atol=1e-6), \
        f"Complicated function test failed: {numerical_result} != {exact_integral}"

def test_clenshaw_curtis():
    """Test the clenshaw_curtis function with known analytic integrals."""

    # Example 1: Integrating f(x) = exp(x)
    N = 10
    def f(x):
        return np.exp(x)

    x, w = clenshaw_curtis(N)
    integral = np.sum(w * f(x))  # Exact integral of exp(x) over [-1, 1] is exp(1) - exp(-1)
    exact_integral = np.exp(1) - np.exp(-1)
    assert np.isclose(integral, exact_integral, atol=1e-12), f"Failed exp(x) test for N={N}: {integral} != {exact_integral}"

    # Example 2: Checking symmetry of the weights
    # The Clenshaw-Curtis weights should be symmetric
    for N in range(2, 11):  # Start from N=2 to avoid trivial case
        x, w = clenshaw_curtis(N)
        assert np.allclose(w, w[::-1]), f"Failed symmetry test for N={N}"

    # Example 3: Testing higher-order polynomials (x^n)
    N = 20
    for n in range(1, 11):  # Test for n = 1, 2, ..., 10
        def f(x):
            return x**n

        x, w = clenshaw_curtis(N)
        integral = np.sum(w * f(x))
        # Exact integral of x^n over [-1, 1] is 0 for odd n, and 2/(n+1) for even n
        expected_integral = 0.0 if n % 2 != 0 else 2 / (n + 1)
        assert np.isclose(integral, expected_integral, atol=1e-12), f"Failed x^{n} test for N={N}: {integral} != {expected_integral}"

# Run tests
if __name__ == "__main__":
    test_trapezoidal_rule()
    test_simpsons_rule()
    test_large_N_non_trivial_function()
    test_chebgauss1()
    test_chebgauss2()
    test_leggaus()
    test_clenshaw_curtis()
    print("All tests passed!")
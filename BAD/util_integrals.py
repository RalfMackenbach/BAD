import numpy as np
from scipy.integrate import simpson
from scipy.special import legendre
from scipy.linalg import eigh_tridiagonal
from numpy.polynomial.chebyshev import chebgauss, chebweight
from numpy.polynomial.legendre import legder, legval, leggauss


########################
# DISCRETE INTEGRATION #
########################
def check_monotonic_array(x):
    assert np.any(np.less(x[1:],x[:-1])), Warning('x array must be monotonically increasing!')
    return 0

def trapezoidal_weights(x):
    """
    Compute weights for the trapezoidal rule more efficiently.
    
    Parameters:
        x (array-like): Grid points.
        
    Returns:
        weights (array-like): Trapezoidal rule weights.
    """
    N = len(x)
    weights = np.zeros(N)

    # Add the contributions from the intervals directly
    weights[1:-1] = (x[2:] - x[:-2]) / 2  # Interior points
    weights[0] = (x[1] - x[0]) / 2        # Left endpoint
    weights[-1] = (x[-1] - x[-2]) / 2     # Right endpoint

    return weights

def simpsons_weights(x):
    """
    Compute weights for Simpson's rule with non-uniform grid points.
    
    Parameters:
        x (array-like): Grid points (must have an odd number of points).
        
    Returns:
        weights (array-like): Simpson's rule weights.
    """
    N = len(x)
    if (N - 1) % 2 != 0:
        raise ValueError("Simpson's rule requires an odd number of points (even number of intervals).")

    # Compute intervals
    h0 = x[1:-1:2] - x[:-2:2]  # Left intervals
    h1 = x[2::2] - x[1:-1:2]   # Right intervals
    h_sum = h0 + h1            # Combined interval widths

    # Weights for each triplet of points
    w_left = h0 * (2 * h0 + h1) / (6 * h_sum)
    w_mid = (h0 + h1) ** 2 / (6 * h_sum)
    w_right = h1 * (2 * h1 + h0) / (6 * h_sum)

    # Assemble full weights array
    weights = np.zeros(N)
    weights[:-2:2] += w_left  # Left points
    weights[1:-1:2] += w_mid  # Middle points
    weights[2::2] += w_right  # Right points

    return weights

def midpoint_weights(x):
    """
    Compute weights for the midpoint rule with non-uniform grid points (vectorized).
    
    Parameters:
        x (array-like): Grid points.
        
    Returns:
        weights (array-like): Midpoint rule weights.
    """
    N = len(x)
    if N < 2:
        raise ValueError("At least two grid points are required.")

    # Compute interval widths
    dx_left = np.zeros(N)   # Interval widths to the left of each point
    dx_right = np.zeros(N)  # Interval widths to the right of each point

    dx_left[1:] = (x[1:] - x[:-1]) / 2   # Left intervals
    dx_right[:-1] = (x[1:] - x[:-1]) / 2  # Right intervals

    # Combine contributions
    weights = dx_left + dx_right

    return weights

def newton_cotes_weights(x, order):
    """
    Compute the weights for Newton-Cotes quadrature on a non-uniform grid for a given order.
    
    Parameters:
        x (array-like): Grid points (non-uniform).
        order (int): Order of the Newton-Cotes rule (1 to 4).
        
    Returns:
        weights (array-like): The corresponding weights for the given order.
    """
    
    # Ensure the order is between 1 and 4, or use the Vermont approach for higher orders
    if order < 1:
        raise ValueError("Order must be greater than or equal to 1.")
    
    N = len(x)
    weights = np.zeros(N)
    
    # Calculate the distances (dx) between adjacent points
    dx = x[1:] - x[:-1]
    
    if order == 1:  # Trapezoidal Rule (order 1)
        # Trapezoidal rule: weights for each pair of points
        weights[:-1] += dx / 2
        weights[1:] += dx / 2
    
    elif order == 2:  # Simpson's Rule (order 2)
        # Simpson's rule: weights for every three consecutive points
        weights[::2] += (dx[:-1] + dx[1:]) / 6  # Odd-index points
        weights[1:-1:2] += 2 * (dx[:-1] + dx[1:]) / 3  # Even-index points
    
    elif order == 3:  # 3rd-order Newton-Cotes (order 3)
        # 3rd-order weights: calculated for every set of 4 consecutive points
        weights[::3] += 3 * (dx[2:] + dx[1:-1] + dx[:-2]) / 8
        weights[1:-2:3] += 9 * (dx[2:] + dx[1:-1] + dx[:-2]) / 8
        weights[2:-1:3] += 3 * (dx[2:] + dx[1:-1] + dx[:-2]) / 8
    
    elif order == 4:  # Boole's Rule (order 4)
        # Boole's rule: weights for every five consecutive points
        weights[::4] += 7 * (dx[3:] + dx[2:-1] + dx[1:-2] + dx[:-3]) / 90
        weights[1:-3:4] += 32 * (dx[3:] + dx[2:-1] + dx[1:-2] + dx[:-3]) / 90
        weights[2:-2:4] += 12 * (dx[3:] + dx[2:-1] + dx[1:-2] + dx[:-3]) / 90
        weights[3:-1:4] += 32 * (dx[3:] + dx[2:-1] + dx[1:-2] + dx[:-3]) / 90
        weights[4::4] += 7 * (dx[3:] + dx[2:-1] + dx[1:-2] + dx[:-3]) / 90
    
    elif order > 4:  # For orders higher than 4, use Vermont's method or a similar high-order Newton-Cotes
        N = len(x)
    
        # Generate the Vandermonde matrix for the grid points
        V = np.vander(x, order+1, increasing=True)
        
        # Right-hand side for the quadrature (unitary function values at grid points)
        b = np.zeros(N)
        b[0] = 1  # First weight is 1 for the first grid point (this can vary)
        
        # Solve the system V * w = b to find the weights
        weights = np.linalg.solve(V, b)
    
    return weights

def weight_integration(x, method):

    if method == "trapz":
        weight = trapezoidal_weights(x)
    elif method == "simpsons":
        weight = simpsons_weights(x)
    elif method == "midpoint":
        weight = midpoint_weights(x)
    else:
        raise Warning("No other discrete method to work on the grid has been implemented!")

    return weight

def _gtrapz(x, f, h, tol = 1e-10):
    r"""
    ``gtrapz`` estimates integrals of the form
    .. math::
       \int \frac{h(x)}{\sqrt{f(x)}} \mathrm{d}x
    by means of a generalisation of the trapezoidal rule.
     Args:
        xi: array containing left points xi
        xj: array containing right points xj
        fi: array containing left points of f(x)
        fj: array containing right points of f(x)
        hi: array containing left points of h(x)
        hj: array containing right points of h(x)
    """
    # Arrays for computing the integrals 
    xi = x[:-1]; fi = f[:-1]; hi = h[:-1]
    xj = x[1:];  fj = f[1:];  hj = h[1:]
    # include limit case where fi=fj
    ans = np.asarray(1/2 * (xj - xi) * (hi + hj)/ np.sqrt(1/2*(fi+fj)))
    # do division, keeping limit whenever fi=fj
    ans = np.divide( 2 * (xj - xi) * (hj * np.sqrt(fj) - hi * np.sqrt(fi))*(fj-fi) - 4/3 * (xj - xi) * (hj - hi) * (np.power(fj,3/2)- np.power(fi,3/2)), np.square(fj-fi),
                    out=ans,where=np.abs(fi-fj)>tol)       
    return np.sum(ans)

def _cum_gtrapz(x, f, h, tol = 1e-10):
    r"""
    ``gtrapz`` estimates integrals of the form
    .. math::
       \int^x \frac{h(x)}{\sqrt{f(x)}} \mathrm{d}x
    by means of a generalisation of the trapezoidal rule.
     Args:
        xi: array containing left points xi
        xj: array containing right points xj
        fi: array containing left points of f(x)
        fj: array containing right points of f(x)
        hi: array containing left points of h(x)
        hj: array containing right points of h(x)
    """

    # Arrays for computing the integrals 
    xi = x[:-1]; fi = f[:-1]; hi = h[:-1]
    xj = x[1:];  fj = f[1:];  hj = h[1:]
    # include limit case where fi=fj
    ans = np.asarray(1/2 * (xj - xi) * (hi + hj)/ np.sqrt(1/2*(fi+fj)))
    # do division, keeping limit whenever fi=fj
    ans = np.divide( 2 * (xj - xi) * (hj * np.sqrt(fj) - hi * np.sqrt(fi))*(fj-fi) - 4/3 * (xj - xi) * (hj - hi) * (np.power(fj,3/2)- np.power(fi,3/2)), np.square(fj-fi),
                    out=ans,where=np.abs(fi-fj)>tol)  
    
    return np.cumsum(ans)

def normalise_map(x):
    # Compute edges
    edge_l = x[0]
    edge_r = x[-1]

    # Normalised x
    factor = 2/(edge_r - edge_l)
    x_norm = factor*(x - edge_l) - 1

    return x_norm, factor

def sin_map(x, pass_t = False):
    # Take from x along the fieldline to t
    t = 2/np.pi*np.arcsin(2*(x - x[0])/(x[-1] - x[0]) - 1)
    return t

def _gquadz_definite(af,bf,cf,ah,bh,ch,xi,xj):
    r"""
    ``gquadz_definite`` is the definite integral of
    .. math::
       \int_{x_i}^{x_j} \frac{h(x)}{\sqrt{f(x)}} \mathrm{d}x
    by assuming that 
    .. math::
        f(x) = af x^2 + bf x + cf
        h(x) = ah x^2 + bh x + ch
     Args:
        af: coefficient of x^2 in f(x)
        bf: coefficient of x in f(x)
        cf: constant in f(x)
        ah: coefficient of x^2 in h(x)
        bh: coefficient of x in h(x)
        ch: constant in h(x)
        x:  x value at which the indefinite integral is evaluated
    """
    # allow complex numbers, double precision
    af = np.complex128(af)
    bf = np.complex128(bf)
    cf = np.complex128(cf)
    ah = np.complex128(ah)
    bh = np.complex128(bh)
    ch = np.complex128(ch)
    xi = np.complex128(xi)
    xj = np.complex128(xj)
    

    term_1 = -2*np.sqrt(af)*np.sqrt(cf + xi*(af*xi + bf)) + 2*af*xi + bf
    arg_1 = np.angle(term_1)
    term_2 = -2*np.sqrt(af)*np.sqrt(cf + xj*(af*xj + bf)) + 2*af*xj + bf
    arg_2 = np.angle(term_2)
    num_log = (-8*af**2*ch + 4*af*ah*cf + 4*af*bf*bh - 3*ah*bf**2)*(-1.j*(arg_1-arg_2)-np.log(np.abs(term_1))+ np.log(np.abs(term_2)))
    log_terms = np.where(np.abs(af) < 1e-7, np.zeros(len(af)), num_log/(8*af**2.5))
    
    num_sqrt = (-(2*np.sqrt(af)*np.sqrt(cf + xi*(af*xi + bf))*(2*af*ah*xi + 4*af*bh - 3*ah*bf)) \
         + (2*np.sqrt(af)*np.sqrt(cf + xj*(af*xj + bf))*(2*af*ah*xj + 4*af*bh - 3*ah*bf)))
    alt = (16 *np.sqrt(cf + bf*xj)*(8*ah*cf**2 - 2*bf*cf*(5*bh + 2*ah*xj) + bf**2*(15*ch + xj*(5*bh + 3*ah*xj))))/(120* bf**3) - \
          (16 *np.sqrt(cf + bf*xi)*(8*ah*cf**2 - 2*bf*cf*(5*bh + 2*ah*xi) + bf**2*(15*ch + xi*(5*bh + 3*ah*xi))))/(120* bf**3)
    sqrt_terms = np.where(np.abs(af) < 1e-7, alt, num_sqrt/(8*af**2.5))

    tot = log_terms + sqrt_terms

    return np.real(tot)

def _get_abc(f1,f2,f3,x1,x2,x3):
    r"""
    ``get_abc`` finds the coefficients of a quadratic function that fits the function f(x) between three consecutive points.
    .. math::
        f(x) = ax^2 + bx + c
    Args:
        f: array of function values
        x: array of x values
    """
    # find all a,b,c, and keep it vectorized. We do not assume that x is evenly spaced.
    a = (x1*(f3-f2) + x2*(f1-f3) + x3*(f2-f1))/(x1-x2)/(x1-x3)/(x2-x3)
    b = (f2-f1)/(x2-x1) - a*(x1+x2)
    c = f1 - a*x1**2 - b*x1
    return a,b,c

def _gquadz(x, f, h):
    r"""
    ``gquadz`` calculates the definite integral of
    .. math::
        \int_a^b \frac{h(x)}{\sqrt{f(x)}} \mathrm{d}x
    by assuming that f(x) is a quadratic function between three consecutive points.
    Args:
        f: array of function values
        h: array of h(x) values
        x: array of x values
    """        
    # calculate the integral for each pair of zeros
    l_zero = x[0]
    r_zero = x[-1]
    # create input for a,b,c
    f_0 = f[0:-2]
    f_1 = f[1:-1]
    f_2 = f[2:]
    h_0 = h[0:-2]
    h_1 = h[1:-1]
    h_2 = h[2:]
    x_0 = x[0:-2]
    x_1 = x[1:-1]
    x_2 = x[2:]
    # find the coefficients of the quadratic functions
    a_f,b_f,c_f = _get_abc(f_0,f_1,f_2,x_0,x_1,x_2)
    a_h,b_h,c_h = _get_abc(h_0,h_1,h_2,x_0,x_1,x_2)

    a_f,b_f,c_f = np.append(a_f,a_f[-1]), np.append(b_f,b_f[-1]), np.append(c_f,c_f[-1])
    a_h,b_h,c_h = np.append(a_h,a_h[-1]), np.append(b_h,b_h[-1]), np.append(c_h,c_h[-1])

    # now construct x_left and x_right
    x_left  = x[:-1]
    x_right = x[1:]

    # calculate the integral
    integral = _gquadz_definite(a_f,b_f,c_f,a_h,b_h,c_h,x_left,x_right)
    # integral[np.isnan(integral)] = 0.0
    integral = np.sum(integral)

    return integral

def _cum_gquadz(x, f, h):
    r"""
    ``gquadz`` calculates the definite integral of
    .. math::
        \int_a^b \frac{h(x)}{\sqrt{f(x)}} \mathrm{d}x
    by assuming that f(x) is a quadratic function between three consecutive points.
    Args:
        f: array of function values
        h: array of h(x) values
        x: array of x values
    """        
    # calculate the integral for each pair of zeros
    l_zero = x[0]
    r_zero = x[-1]
    # create input for a,b,c
    f_0 = f[0:-2]
    f_1 = f[1:-1]
    f_2 = f[2:]
    h_0 = h[0:-2]
    h_1 = h[1:-1]
    h_2 = h[2:]
    x_0 = x[0:-2]
    x_1 = x[1:-1]
    x_2 = x[2:]
    # find the coefficients of the quadratic functions
    a_f,b_f,c_f = _get_abc(f_0,f_1,f_2,x_0,x_1,x_2)
    a_h,b_h,c_h = _get_abc(h_0,h_1,h_2,x_0,x_1,x_2)

    a_f,b_f,c_f = np.append(a_f,a_f[-1]), np.append(b_f,b_f[-1]), np.append(c_f,c_f[-1])
    a_h,b_h,c_h = np.append(a_h,a_h[-1]), np.append(b_h,b_h[-1]), np.append(c_h,c_h[-1])

    # now construct x_left and x_right
    x_left  = x[:-1]
    x_right = x[1:]

    # calculate the integral
    integral = _gquadz_definite(a_f,b_f,c_f,a_h,b_h,c_h,x_left,x_right)
    # integral[np.isnan(integral)] = 0.0
    integral = np.cumsum(integral)

    return integral

def bounce_integral_discrete(f, h, x, method = "gtrapz"):
    r"""
    ``bounce_integral`` does the bounce integral
    .. math::
       \int \frac{h(x)}{\sqrt{f(x)}} \mathrm{d}x.
    Can be done by either quad if is_func=True, or
    gtrapz if is_func=False. When is_func=True 
    both f and h need to be functions. Otherwise
    they should be arrays. sinhtanh can furthermore
    be set to either True of False to use sinhtanh
    quadrature methods (only is is_func=True).
     Args:
        f: function or arrays containing f
        h: function or arrays containing h
    """
    if method == "gtrapz":
        # Compute integral
        val = _gtrapz(x, f, h)
    elif method == "gquadz":
        # Compute integral
        val = _gquadz(x, f, h)
    else:
        if method in ["trapz", "simpsons", "midpoint"]:
            weight = weight_integration(method)
            div = np.divide(weight * h, np.sqrt(f))
            val = np.sum(div)
        elif method in ["sin_trapz", "sin_simpsons", "sin_midpoint"]:
            weight = weight_integration(method[4:])
            # t = sin_map(x)
            # val = np.sum(f)
        else:
            raise Warning("Discrete integration method not recognised!")
        
    return val

def cum_bounce_integral_discrete(f, h, x, method = "gtrapz"):
    r"""
    ``bounce_integral`` does the bounce integral
    .. math::
       \int \frac{h(x)}{\sqrt{f(x)}} \mathrm{d}x.
    Can be done by either quad if is_func=True, or
    gtrapz if is_func=False. When is_func=True 
    both f and h need to be functions. Otherwise
    they should be arrays. sinhtanh can furthermore
    be set to either True of False to use sinhtanh
    quadrature methods (only is is_func=True).
     Args:
        f: function or arrays containing f
        h: function or arrays containing h
    """
    if method == "gtrapz":
        # Compute integral
        val = _cum_gtrapz(x, f, h)
        val = np.insert(val, 0, 0.0)
    elif method == "gquadz":
        # Compute integral
        val = _cum_gquadz(x, f, h)
        val = np.insert(val, 0, 0.0)
    else:
        if method in ["trapz", "simpsons", "midpoint"]:
            weight = weight_integration(method)
            div = np.divide(weight * h, np.sqrt(f))
            val = np.sum(div)
        elif method in ["sin_trapz", "sin_simpsons", "sin_midpoint"]:
            weight = weight_integration(method[4:])
            # t = sin_map(x)
            # val = np.sum(f)
        else:
            raise Warning("Discrete integration method not recognised!")
        
    return x, val

def newton_cotes_weights_uniform(N, order, avoid_edges = False):
    """
    Compute the weights for Newton-Cotes quadrature on a uniform grid.
    
    Parameters:
        x (array-like): Grid points (uniform grid).
        order (int): Order of the Newton-Cotes rule (1 to higher orders).
        
    Returns:
        weights (array-like): The corresponding weights for the given order.
    """
   
    if order == 1:  # Trapezoidal Rule (order 1)
        if avoid_edges:
            x = np.linspace(-1, 1, N+2)
            x = x[1:-1]
        else:
            x = np.linspace(-1, 1, N)
        h = x[1] - x[0]  # uniform grid spacing
        
        w = np.zeros(N)
        w[0] = h / 2
        w[-1] = h / 2
        w[1:-1] = h
    
    elif order == 2:  # Simpson's Rule (order 2)
        # Need to make sure that N is made odd
        N -= 1 + (N % 2)
        if avoid_edges:
            x = np.linspace(-1, 1, N+2)
            x = x[1:-1]
        else:
            x = np.linspace(-1, 1, N)
        h = x[1] - x[0]  # uniform grid spacing
        
        w = np.zeros(N)
        w[0] = w[-1] = h / 3
        w[1:-1:2] = 4 * h / 3  # Odd-index points
        w[2:-1:2] = 2 * h / 3  # Even-index points
    
    elif order == 4:  # Boole's Rule (order 4)
        # Need to make sure that N is made odd
        N -= 1 + (N % 4)
        if avoid_edges:
            x = np.linspace(-1, 1, N+2)
            x = x[1:-1]
        else:
            x = np.linspace(-1, 1, N)
        h = x[1] - x[0]  # uniform grid spacing

        w = np.zeros(N)
        w[0] = w[-1] = 14 * h / 45
        w[1::2] = 64 * h / 45
        w[2::4] = 8 * h / 15
        w[4::4] = 28 * h / 45
    else:
        raise ValueError("Only orders 1, 2, and 4 are implemented for uniform grid.")
    
    return x, w
    
def chebgauss1(N):
    """
    Gauss-Chebyshev quadrature.

    Returns quadrature points xₖ and weights wₖ for the approximate evaluation
    of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).

    Parameters
    ----------
    N : int
        Number of quadrature points.

    Returns
    -------
    x, w : tuple[np.ndarray]
        Shape (N, ).
        Quadrature points and weights.

    """
    x, w = chebgauss(N)         # Weight for integration with factor 1/√1-x²
    return x, w / chebweight(x) # Renormalise for the right integral without square root

def chebgauss2(N):
    """Gauss-Chebyshev quadrature of the second kind.

    Returns quadrature points xₖ and weights wₖ for the approximate evaluation
    of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).

    Parameters
    ----------
    N : int
        Number of quadrature points.

    Returns
    -------
    x, w : tuple[np.ndarray]
        Shape (N, ).
        Quadrature points and weights.

    """
    # Adapted from
    # github.com/scipy/scipy/blob/v1.14.1/scipy/special/_orthogonal.py#L1803-L1851.
    m = int(N)
    if N < 1 or N != m:
        raise ValueError('n must be a positive integer.')
    t = np.arange(m, 0, -1) * np.pi / (m + 1)
    x = np.cos(t)
    w = np.pi * np.sin(t)**2 / (m + 1)
    
    return x, w * chebweight(x)

def leggauss_lob(N, interior_only=True):
    """
    Compute nodes and weights for Lobatto-Gauss-Legendre quadrature.

    Parameters:
        N (int): Number of nodes (including endpoints).

    Returns:
        x (numpy array): Quadrature nodes.
        w (numpy array): Quadrature weights.
    """
    N = N + 2 * bool(interior_only)
    if N < 2:
        raise ValueError("Number of nodes must be at least 2.")

    # Golub-Welsh algorithm
    n = np.arange(2, N - 1)
    x = eigh_tridiagonal(np.zeros(N - 2), np.sqrt((n**2 - 1) / (4 * n**2 - 1)), eigvals_only = True)
    c0 = np.zeros(N)
    c0[-1] = 1

    # improve (single multiplicity) roots by one application of Newton
    c = legder(c0)
    dy = legval(x=x, c=c)
    df = legval(x=x, c=legder(c))
    x -= dy / df

    w = 2 / (N * (N - 1) * legval(x=x, c=c0) ** 2)

    if not interior_only:
        x = np.hstack([-1.0, x, 1.0])
        w_end = 2 / (N * (N - 1))
        w = np.hstack([w_end, w, w_end])

    assert x.size == w.size == N - 2 * bool(interior_only)

    return x, w

def leggaus(N):
    # Call leggaus for Gauss-Legendre quadrature from np
    return leggauss(N)

def takashi(N, t_max = 3.0):
    """
    Double exponential integration method. We consider an equally spaced grid in t ∈ [-t_max, t_max],
    and return the grid and weight correspondingly.
    """
    # Choose t_max
    # x_max = 1.0
    # t_max = np.arcsinh(2 * np.arctanh(x_max) / np.pi)
    x_max = 3.05
    x = np.linspace(-x_max, x_max, N)
    w = x[1] - x[0]

    return x, w

def clenshaw_curtis(n):
    """
    Computes a Clenshaw-Curtis quadrature rule without explicit loops. 
    Adapted https://people.math.sc.edu/Burkardt/py_src/quadrule/clenshaw_curtis_compute.py.
    
    Parameters:
        n (int): The order of the rule.
    
    Returns:
        x (ndarray): The abscissas.
        w (ndarray): The weights.
    """
    if n == 1:
        # sFor n = 1, return the single point and its weight
        x = np.zeros(n)
        w = np.zeros(n)
        w[0] = 2.0
        return x, w

    # Calculate the abscissas (nodes)
    theta = np.pi * np.linspace(0, n - 1, n) / (n - 1)
    x = np.cos(theta)

    # Initialize the weights
    w = np.ones(n)

    # Vectorized weight calculation (remove the double loop)
    j = np.arange(0, (n - 1) // 2)  # j ranges from 0 to (n-1)//2 - 1
    
    # Create b: 1.0 for the special case, otherwise 2.0
    b = np.where(2 * (j + 1) == (n - 1), 1.0, 2.0)
    
    # Compute the cosine terms for all i, j
    cos_terms = np.cos(2 * (j[:, None] + 1) * theta[None, :])  # shape (j.size, n)
    
    # Calculate the denominator for each j
    denominator = 4 * j * (j + 2) + 3  # shape (j.size,)
    
    # Update the weights (for all i simultaneously)
    weight_contribution = np.sum(b[:, None] * cos_terms / denominator[:, None], axis=0)
    w -= weight_contribution

    # Normalize the weights
    w[0] /= (n - 1)
    w[1:n-1] *= 2.0 / (n - 1)
    w[n-1] /= (n - 1)

    return x, w


    return x, w

def bounce_integral_fn(f, h, x_l, x_r, N = 100, method = "quad", order = 1, mapping = "sin", scale = 3.0, trim = False):
    ##############################
    # CONSTRUCT WEIGHTS AND GRID #
    ##############################
    # All the grids here are constructed in the interval t ∈ [−1, 1]
    t, w = select_int_method_tw(method, N, order = order, trim = trim)
    
    ########################
    # SAMPLE IN REAL SPACE #
    ########################
    # We map the t ∈ [-1, 1] domains into x, the real domain of the input functions.
    # Depending on which map is chosen, proceed differently
    map, d_map = effect_map(mapping, x_l, x_r, scale)

    ###############################
    # PERFORM QUADRATURE-LIKE SUM #
    ###############################
    # Construct grid in x ∈ [x_l, x_r] domain
    x = map(t)
    # Quadrature integral on the t domain:  fun(x(t))  *   dx/dt(t)   *  w(t)
    #                                        function   map derivative  weight
    res = np.sum(h(x)/np.sqrt(f(x)) * d_map(t) * w)
    
    return res

def effect_map(mapping, x_ls, x_rs, scale):
    # We map the t ∈ [-1, 1] domains into x subdomains, the real domain of the input functions.
    # Depending on which map is chosen, proceed differently
    if mapping == "normal":
        ## Standard simple case: 
        ## simply uniform map from t ∈ [−1, 1] to x ∈ [x_l, x_r]
        # Compute map
        factor = 0.5*(x_rs - x_ls)
        map = lambda t: x_ls + factor * (t + 1)
        # The derivative of the map respect to t
        d_map = lambda t: factor + t*0
    elif mapping == "sin":
        ## Sin case: 
        ## method used by Unalmis in https://doi.org/10.48550/arXiv.2412.01724, where a sine mapping is used 
        ## to ameliorate the divergence of the integrand near bounce points. The map is from t ∈ [−1, 1] to 
        ## x ∈ [x_l, x_r] using x = xl + 0.5 (xr - xl) * (1 + sin(πt/2)).
        # Construct map
        map = lambda t: x_ls + 0.5*(x_rs - x_ls) * (1 + np.sin(0.5*np.pi*t))
        # Differentiate map
        d_map = lambda t: 0.5*(x_rs - x_ls) * 0.5*np.pi * np.cos(0.5*np.pi*t)
    elif mapping == "takashi":
        ## Double exponential method: the key of the scheme is to have a regular grid in t ∈ [-scale, scale] 
        ## and map it to ↦ x using x = 0.5 (x_r + x_l) + 0.5 (x_r - x_l) tanh(π sinh(t) / 2)
        # Compute map
        t_max = scale
        map = lambda t: 0.5*(x_rs + x_ls) + 0.5*(x_rs - x_ls) * np.tanh(0.5 * np.pi * np.sinh(t_max * t))
        # Differentiate map
        d_map = lambda t:  0.5*(x_rs - x_ls) * 0.5 * np.pi * t_max * np.cosh(t_max * t) / np.cosh(0.5 * np.pi * np.sinh(t_max * t)) ** 2
    elif mapping == "exponential_l":
        ## Single exponential method: the key of the scheme is to have a regular grid in t ∈ [-1, 1] 
        ## map it to ↦ [0, scale] and then map it to x ↦ x_r - (x_r - x_l) exp(-t)
        # Compute map
        t_max = scale
        map = lambda t: x_rs + (x_ls - x_rs) * np.exp(-0.5 * scale * (t + 1))
        # Differentiate map
        d_map = lambda t:  (x_rs - x_ls) * 0.5 * scale * np.exp(-0.5 * scale * (t + 1))
    elif mapping == "exponential_r":
        ## Single exponential method: the key of the scheme is to have a regular grid in t ∈ [-1, 1] 
        ## map it to ↦ [0, scale] and then map it to x ↦ x_l - (x_l - x_r) exp(-t), so now avoid 
        ## touching the right of the domain
        # Compute map
        t_max = scale
        map = lambda t: x_ls + (x_rs - x_ls) * np.exp(-0.5 * scale * (t + 1))
        # Differentiate map
        d_map = lambda t:  (x_ls - x_rs) * 0.5 * scale * np.exp(-0.5 * scale * (t + 1))
    else:
        raise Warning("Mapping {} not implemented!".format(mapping))
    
    return map, d_map

def select_int_method_tw(method, k, order = 1, trim = False):
    """
    Make grid points and weights for integrating in the interval t ∈ [-1, 1]
    """
    # All the grids here are constructed in the interval t ∈ [−1, 1]
    if method == "chebgauss1":
        # Gauss-Chebyshev quadrature
        t, w = chebgauss1(k)
    elif method == "chebgauss2":
        # Alternative Gauss-Chebyshev quadrature: Ulnamis https://doi.org/10.48550/arXiv.2412.01724 adapted 
        # from github.com/scipy/scipy/blob/v1.14.1/scipy/special/_orthogonal.py#L1803-L1851 
        t, w = chebgauss2(k)
    elif method == "quad":
        # Standard uniform grid quadrature schemes using Newton - Cotes method
        # order = 1 : trapezoid
        # order = 2 : simpson
        # order = 4 : boole
        t, w = newton_cotes_weights_uniform(k, order = order, avoid_edges = trim)
    elif method == "GL":
        # Gauss-Legendre quadrature
        t, w = leggauss(k)
    elif method == "clenshaw":
        # Clenshaw-Curtis quadrature
        t, w = clenshaw_curtis(k)
    else:
        raise Warning("Not implemented!")
    
    return t, w

def cum_bounce_integral_fn(f, h, x_l, x_r, x_out = None, N = 100, approach = "subdomains",
                            features = {"method": "quad", "which_sub": 0, "order": 1, "k": 4, "mapping": "sin", "scale": 3.0, "trim": False}):
    # There are different forms of proceeding.
    # SUBDOMAINS: 
    #   split the domain into subdomains defined by x_out, and integrate each of these, to them cumsum to construct the 
    #   required integrated function
    # DISCRETE:
    #   calls the discrete calculation routines
    
    if approach == "subdomains":
        #######################
        # EXTRACT INFORMATION #
        #######################
        if isinstance(features["method"], list) or isinstance(features["method"], np.ndarray):
            method = features["method"]
            which_sub = features["which_sub"]
            order = features["order"]
            k = features["k"]
            trim = features["trim"]
            mapping = features["mapping"]
            scale = features["scale"]
        else:
            method = [features["method"]]
            which_sub = [features["which_sub"]]
            order = [features["order"]]
            k = [features["k"]]
            trim = [features["trim"]]
            mapping = [features["mapping"]]
            scale = [features["scale"]]

        ###############################
        # DEFINE SUBDOMAIN EDGES IN x #
        ###############################
        ## Separation of original domain into subdomains ##
        if isinstance(x_out, np.ndarray) or isinstance(x_out, list):
            if x_l < x_out[0]:
                x_l_subdomain = np.insert(x_out[:-1], 0, x_l)
                x_r_subdomain = x_out
            elif x_l == x_out[0]:
                x_l_subdomain = x_out[:-1]
                x_r_subdomain = x_out[1:]
            else:
                raise Warning('x_l > x_out[0] which makes no sense!')
        else:
            # Make x subgrids
            x_subdomain_edges = x_l + (x_r - x_l) * np.linspace(0.0, 1.0, N)
            x_l_subdomain = x_subdomain_edges[:-1]
            x_r_subdomain = x_subdomain_edges[1:]

        ##############################
        # CONSTRUCT WEIGHTS AND GRID #
        ##############################
        # For each subdomain, we need to decide how we are going to proceed with integration. We take the approach of defining
        # each of these as t ∈ [−1, 1] which we will have to map later. But first, need to construct methods according to the
        # details provided in features. These come in lists, with the first entry representing the method in the most general
        # case; then need to check others (check in which_sub). So as not to repeat the construction every time, we accommodate
        # together all different methods possible.
        method_label = np.zeros(len(k))
        method_info = []
        integration_data = []
        for j, args in enumerate(zip(method, k, order, trim)):
            if args in method_info:
                method_label[j] = method_info.index(args)
            else:
                method_info.append(args)
                method_label[j] = len(method_info) - 1
                t, w = select_int_method_tw(*args)
                integration_data.append((t, w))
        
        ############################
        # INTEGRATE EACH SUBDOMAIN #
        ############################
        ## Each sub-grid needs to be integated ##
        N_sub = len(x_l_subdomain)
        int_subdomain = np.zeros(N_sub)

        # Extend the method indices to have the right reference to the type of integration        
        def extend_which_sub(which_sub, method_label):
            method_label_ext = np.array(N_sub * [method_label[0]], dtype="object")
            for j, element in enumerate(which_sub):
                if isinstance(element, tuple):
                    method_label_ext[element[0]:element[1]] = method_label[j]
                elif j > 0:
                    method_label_ext[element] = method_label[j]
            return method_label_ext
        
        method_label_ext = extend_which_sub(which_sub, method_label)
        map_label_ext    = extend_which_sub(which_sub, mapping)
        scale_ext        = extend_which_sub(which_sub, scale)

        # Now run through each subdomain
        for j_s, (x_ls, x_rs) in enumerate(zip(x_l_subdomain, x_r_subdomain)):
            # Select the integration according to what was specified
            (t, w) = integration_data[int(method_label_ext[j_s])]
            # Select corresponcding map
            mapping_current = map_label_ext[j_s]
            scale_current   = scale_ext[j_s]
            # Construct map
            map, d_map = effect_map(mapping_current, x_ls, x_rs, scale_current)

            ###############################
            # PERFORM QUADRATURE-LIKE SUM #
            ###############################
            # Construct grid in x ∈ [x_ls, x_rs] sub-domain
            x = map(t)
            # Quadrature integral on the t domain:  fun(x(t))  *   dx/dt(t)   *  w(t)
            #                                        function   map derivative  weight
            int_subdomain[j_s] = np.sum(h(x)/np.sqrt(f(x)) * d_map(t) * w)

        #########################################
        # PUT TOGETHER INTO INDEFINITE INTEGRAL #
        #########################################
        # Sum over the different intervals: these correspond to values at x_r_subdomain
        res = np.cumsum(int_subdomain)
        x = x_r_subdomain
        # Now need to include zero for first entry at the left edge
        res = np.insert(res, 0, 0.0)
        x   = np.insert(x, 0, x_l)

    elif approach == "discrete":
        ######################
        # DEFINE DOMAIN IN x #
        ######################
        ## Make or read domain ##
        if isinstance(x_out, np.ndarray) or isinstance(x_out, list):
            if x_l < x_out[0]:
                x_out = np.insert(x_out, 0, x_l)
            elif x_l > x_out[0]:
                raise Warning("Error with the left edge: can't compte integral before left bounce point!")
            
            if x_r > x_out[-1]:
                x = np.append(x_out, x_r)
            elif x_r == x_out[-1]:
                x = x_out
            else:
                raise Warning("Error with the right edge: can't compute integral after right bounce point!")
        else:
            # Make x
            x = x_l + (x_r - x_l) * np.linspace(0.0, 1.0, N)

        #############################
        # CALL FOR DISCRETE ROUTINE #
        #############################
        # Evaluate functions 
        h_data = h(x)
        f_data = f(x)

        # Call discrete function
        x, res = cum_bounce_integral_discrete(f_data, h_data, x, method = features["method"])

    else: 
        raise Warning("No alternative to subdomains has been implemented!")
    
    return x, res


def main():
    pass
if __name__ == "__main__":
    main()







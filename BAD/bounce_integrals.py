import numpy as np
from BAD.util_integrals import bounce_integral_discrete, cum_bounce_integral_discrete, bounce_integral_fn, cum_bounce_integral_fn

def bounce_integral(f, h, x_l = None, x_r = None, x = None, N = 100, mode = "fast"):
    """
    Computes the bounce integral of ∫ₓₗˣʳh/√f dx given functions or data arrays using a variety of 
    numerical methods.
    
    Parameters:
        f (callable or array-like): 
            The function or data array that appears in the square root in the integrand denominator. Related to the parallel velocity.
            - If callable, it is given as a function of x.
            - If array-like, it is evaluated in the domain x.
            
        h (callable or array-like): 
            The numerator function in the integrand.
            - If `f` is callable, `h` must also be callable.
            - If `f` is array-like, `h` must be array-like and match its shape.
            
        x_l (float, optional): 
            The left bounce point, lower bound of integration for callable functions. Required when `f` and `h` are callable.
            For the discrete case it is assumed that x[0] is x_l.
            
        x_r (float, optional): 
            The right bounce point, upper bound of integration for callable functions. Required when `f` and `h` are callable.
            For the discrete case it is assumed that x[-1] is x_r.
            
        x (list or numpy.ndarray, optional): 
            The array of points at which we have data for the discrete case.
            - Required when `f` and `h` are array-like.
            
        N (int, optional, default=100): 
            The number of points or resolution used in the integration (when no x provided). 
            
        mode (str, optional, default="fast"): 
            Specifies the integration method to use:
            - "fast": aimed at speed, using less computationally expensive methods (e.g., `takashi` method for function, 'gtrapz' 
            for discrete case).
            - "accurate": aimed at accuracy, uses Gauss-Legendre with sine mapping for functions and 'gquadz' for discrete case.
    
    Returns:
        res (numpy.ndarray): 
            The computed bounce integral value based on the input domain `x`.
    
    Exceptions and Warnings:
        - Raises a warning if the `mode` argument is invalid.
        - Asserts the correct types for inputs when `f` and `h` are callable.
    """

    ## Call one or other depending on whether a function or array is passed ##
    if callable(f):
        assert callable(h), "h is not a function while f is!"
        assert isinstance(x_l, float), "x_l is not provided as a float"
        assert isinstance(x_r, float), "x_r is not provided as a float"

        ##################
        # DO INTEGRATION #
        ##################
        # Interpretation for typical range of N = 50 - 300
        if mode == "accurate":
            # The most accurate appears to be Gauss-Lagendre with a sine mapping (tested for elliptic-like integrals)
            res = bounce_integral_fn(f, h, x_l, x_r, N = N, method = "GL", mapping = "sin")
        elif mode == "fast":
            # Sacrificing some accuracy (still good) but executing faster (especially at larger N), the best appears to
            # be Takshi (double-exponential) method
            res = bounce_integral_fn(f, h, x_l, x_r, N = N, method = "quad", order = 1, mapping = "takashi", scale = 4.0)
        else:
            raise Warning("Provided mode is not recognised. Only 'fast' and 'accurate' are included.")
    else:
        ##################
        # DO INTEGRATION #
        ##################
        # Interpretation for typical range of N = 50 - 300
        if mode == "fast":
            res = bounce_integral_discrete(f, h, x, method = "gtrapz")
        elif mode == "accurate":
            res = bounce_integral_discrete(f, h, x, method = "gquadz")
        else:
            raise Warning("Provided mode is not recognised. Only 'fast' & 'accurate' are included.")
        # Need to include some other form to deal with this, potentially interpolating and doing one of the others
        
    return res

def cum_bounce_integral(f, h, x_l = None, x_r = None, x = None, N = 100, mode = "fast"):
    """
    Computes the cumulative integral ∫ₓₗˣh/√f dx from the left bounce point to the provided x for a given function or data array 
    using numerical methods.
    
    Parameters:
        f (callable or array-like): 
            The function or data array that appears in the square root in the integrand denominator. Related to the parallel velocity.
            - If callable, it is given as a function of x.
            - If array-like, it is evaluated in the domain x.
            
        h (callable or array-like): 
            The numerator function in the integrand.
            - If `f` is callable, `h` must also be callable.
            - If `f` is array-like, `h` must be array-like and match its shape.
            
        x_l (float, optional): 
            The left bounce point, lower bound of integration for callable functions. Required when `f` and `h` are callable.
            For the discrete case it is assumed that x[0] is x_l.
            
        x_r (float, optional): 
            The right bounce point, upper bound of integration for callable functions. Required when `f` and `h` are callable.
            For the discrete case it is assumed that x[-1] is x_r.
            
        x (list or numpy.ndarray, optional): 
            The array of points at which the cumulative integral should be evaluated.
            - Required when `f` and `h` are callable.
            - For array-like `f` and `h`, it represents the domain corresponding to the data (and also output).
            
        N (int, optional, default=100): 
            The number of points or resolution used in the integration (or N*k if the function option is being used). 
            If x is passed, then this superseeds N in the case of running with functions.
            
        mode (str, optional, default="fast"): 
            Specifies the integration method to use:
            - "fast": faster implemented algorithm, using less computationally expensive methods (based on `gtrapz`).
            - "accurate": aimed at accuracy, uses Clenshaw-Curtis with sine mapping for functions and 'gquadz' for discrete case.
    
    Returns:
        x (numpy.ndarray): 
            The points where the cumulative integral is evaluated.
            
        res (numpy.ndarray): 
            The computed cumulative integral values at each point in `x`.
    """
    ## Call one or other depending on whether a function or array is passed ##
    if callable(f):
        assert callable(h), "h is not a function while f is!"
        assert isinstance(x_l, float), "x_l is not provided as a float"
        assert isinstance(x_r, float), "x_r is not provided as a float"
        assert isinstance(x, list) or isinstance(x, np.ndarray), "x in which to evaluate the fn is not provided"

        ##################
        # DO INTEGRATION #
        ##################
        # Interpretation for typical range of N = 50 - 300
        if mode == "accurate":
            # The most accurate appears to be the subdomain approach with the Clenshaw-Curtis algorith and a sine mapping 
            # (tested for elliptic-like integrals), which is quite expensive though in terms of time
            features = {"method": "clenshaw", "which_sub": 0, "order": 1, "k": 20, 
                        "mapping": "sin",     "scale": 3.0,   "trim": False}
            x, res = cum_bounce_integral_fn(f, h, x_l=x_l, x_r=x_r, x_out = x, N = N, approach = "subdomains", features = features)
        elif mode == "fast":
            # Sacrificing accuracy but executing much faster (especially at larger N), the best appears to
            # be gtrapz method
            x, res = cum_bounce_integral_fn(f, h, x_l, x_r, x, approach = "discrete", features = {"method": "gtrapz"})
        else:
            raise Warning("Provided mode is not recognised. Only 'fast' and 'accurate' are included.")
    else:
        ##################
        # DO INTEGRATION #
        ##################
        # Interpretation for typical range of N = 50 - 300
        if mode == "fast":
            x, res = cum_bounce_integral_discrete(f, h, x, method = "gtrapz")
        elif mode == "accurate":
            x, res = cum_bounce_integral_discrete(f, h, x, method = "gquadz")
        else:
            raise Warning("Provided mode is not recognised. Only 'fast' & 'accurate' are included.")
        # Need to include some other form to deal with this, potentially interpolating and doing one of the others
        
    return x, res



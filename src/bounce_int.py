# This file contains all subroutines used in the calculation of the Available
# Energy of trapped particles. Based on fortran implementation
import  numpy           as      np
from    numba           import  jit, njit
import  scipy


def gtrapz(xi,xj,fi,fj,hi,hj):
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
    ans = np.sum( 2 * (xj - xi) * (hj * np.sqrt(fj) - hi * np.sqrt(fi))/(fj-fi) - 4/3 * (xj - xi) * (hj - hi) * (np.power(fj,3/2)- np.power(fi,3/2))/np.square(fj - fi) )
    return ans


def find_zeros(f,x,is_func=False):
    r"""
    ``find_zeros`` finds the zeros of either a function f(x) or an array f.
    x is an array with the points on which f is evaluated for the root finding,
    if there are multiple roots between x[i] and x[i+1] it will not find all
    roots.
     Args:
        f:  either a function for which the roots will be found,
            or an array containing samples of f on the location 
            given by x_arr
        x:  array on which f is evaluated. If is_func=True,
            f will be evaluated on x and roots are found by
            finding sign flips. if is_func is false, these
            are simply the locations of the samples f.
        is_func:    if f is a function set to True, normal set to False
                    
    """
    # we store the roots in a list, as we don't know a priori how many there will be.
    # we also store the indices left of the zero root
    index_list = []
    roots_list = []
    # find all roots if is_func is True
    if is_func==True:
        c = f(x)
        s = np.sign(c)
        for i in range(len(x)-1):
            if s[i] + s[i+1] == 0: # opposite signs
                u = scipy.optimize.brentq(f, x[i], x[i+1])
                z = f(u)
                index_list.append(i)
                roots_list.append(u)
    # find all roots if is_func is False    
    if is_func==False:
        s = np.sign(f)
        for i in range(len(x)-1):
            if s[i] + s[i+1] == 0: # opposite signs
                u = x[i] - f[i] * (x[i+1] - x[i])/(f[i+1]-f[i])
                index_list.append(i)
                roots_list.append(u)

    # check if total number of roots is even.
    # edge cases with odd number of roots have NOT been implemented
    if len(roots_list) % 2 == 1:
        raise Exception("Odd number of bounce points, please adjust resolution or interpolation method.")

    # return all roots
    return index_list, roots_list


def check_first_well(f,x,index,is_func=False):
    r"""
    ``check_first_well`` checks is the first root of f(x) is the start of a well
    or the end. Returns True if it is the start, false if it is the end.
    Args:
        f: either function or array containing samples of f
        x: array with values of x
        index: the indeces left of the roots
        is_func: set to True if f is a function, otherwise set to False.
    """
    ans = True
    if is_func==True:
        xi = x[index[0]]
        xj = x[index[0]+1]
        fi = f(xi)
        fj = f(xj)
        if fi > fj:
            ans = False 
    if is_func==False:
        fi = f[index[0]]
        fj = f[index[0]+1]
        if fi > fj:
            ans = False 
    return ans


def bounce_integral(f,h,x,index,root,is_func=False):
    r"""
    ``bounce_integral`` does the bounce integral
    .. math::
       \int \frac{h(x)}{\sqrt{f(x)}} \mathrm{d}x.
    Can be done by either quad if is_func=True, or
    gtrapz if is_func=False. When is_func=True 
    both f and h need to be functions. Otherwise
    they should be arrays.
     Args:
        f: function or arrays containing f
        h: function or arrays containing h
        index: indices left of the roots
        root: the values where f(x)=0
        is_func: are h and f functions or not.
    """
    # define number of bounce wells
    num_wells = int(len(root)/2)

    # make list with integral values
    bounce_val = []

    # do integral with quadrature methods
    if is_func==True:
        # make x xoordinate periodic
        xmin = x[0]
        xmax = x[-1]
        per = lambda x: ((x - xmin)%(xmax-xmin)) + xmin
        # integrand
        integrand = lambda x : h(per(x))/np.sqrt(np.abs(f(per(x))))
        for well_idx in range(num_wells):
            l_bound  = root[2*well_idx]
            r_bound = root[2*well_idx+1]
            if l_bound > r_bound:
                r_bound = r_bound + xmax
            val, err = scipy.integrate.quad(integrand,l_bound,r_bound)
            bounce_val.append(val)
    
    # do integral with gtrapz
    if is_func==False:
        for well_idx in range(num_wells):
            l_bound = root[2*well_idx]
            r_bound = root[2*well_idx+1]
            l_idx   = index[2*well_idx]
            r_idx   = index[2*well_idx+1]
            # use linear interpolation to find h val at left crossing
            hl_cross = h[l_idx] + (l_bound - x[l_idx])/(x[l_idx+1] - x[l_idx]) * ( h[l_idx+1] - h[l_idx] )
            # use linear interpolation to find h val at right crossing
            hr_cross = h[r_idx] + (r_bound - x[r_idx])/(x[r_idx+1] - x[r_idx]) * ( h[r_idx+1] - h[r_idx] )
            # if the bounce wells are ascending, 
            # the integral can be done straightforwardly
            if l_bound < r_bound:
                # do inner integral
                xi = x[l_idx+1:r_idx]
                fi = f[l_idx+1:r_idx]
                hi = h[l_idx+1:r_idx]
                xj = x[l_idx+2:r_idx+1]
                fj = f[l_idx+2:r_idx+1]
                hj = h[l_idx+2:r_idx+1]
                inner_int = gtrapz(xi,xj,fi,fj,hi,hj)
                # do left edge 
                left_int  = gtrapz(l_bound,x[l_idx+1],0.0,f[l_idx+1],hl_cross,h[l_idx+1])
                # do right edge 
                right_int = gtrapz(x[r_idx],r_bound,f[r_idx],0.0,h[r_idx],hr_cross)
                # compute total int
                val = left_int + inner_int + right_int  
            if l_bound > r_bound:
                # do inner integral (l to end)
                xi = x[l_idx+1:-1]
                fi = f[l_idx+1:-1]
                hi = h[l_idx+1:-1]
                xj = x[l_idx+2::]
                fj = f[l_idx+2::]
                hj = h[l_idx+2::]
                inner_int = gtrapz(xi,xj,fi,fj,hi,hj)
                # do inner integral (start to r)
                xi = x[0:r_idx]
                fi = f[0:r_idx]
                hi = h[0:r_idx]
                xj = x[1:r_idx+1]
                fj = f[1:r_idx+1]
                hj = h[1:r_idx+1]
                inner_int = inner_int + gtrapz(xi,xj,fi,fj,hi,hj)
                # do left edge 
                left_int  = gtrapz(l_bound,x[l_idx+1],0.0,f[l_idx+1],hl_cross,h[l_idx+1])
                # do right edge 
                right_int = gtrapz(x[r_idx],r_bound,f[r_idx],0.0,h[r_idx],hr_cross)
                # compute total int
                val = left_int + inner_int + right_int  
            # append value to bounce vals
            bounce_val.append(val)
    return bounce_val


def bounce_integral_wrapper(f,h,x,is_func=False):
    r"""
    ``bounce_integral_wrapper`` does the bounce integral
    but wraps the root finding routine into one function.
    Can be done by either quad if is_func=True, or
    gtrapz if is_func=False. When is_func=True 
    both f and h need to be functions. Otherwise
    they should be arrays.
     Args:
        f: function or arrays containing f
        h: function or arrays containing h
        is_func: are h and f functions or not.
    """
    # if f is not a function use gtrapz
    if is_func==False:
        # if false use array for root finding
        index,root = find_zeros(f,x,is_func=False)
        # check if first well is edge, if so roll
        first_well = check_first_well(f,x,index,is_func=False)
        if first_well==False:
            index = np.roll(index,1)
            root = np.roll(root,1)
        # do bounce integral
        bounce_val = bounce_integral(f,h,x,index,root,is_func=False)
    # if is_func is true, use it for both root finding and integration
    if is_func==True: 
        index,root = find_zeros(f,x,is_func=True)
        first_well = check_first_well(f,x,index,is_func=True)
        if first_well==False:
            index = np.roll(index,1)
            root = np.roll(root,1)
        bounce_val = bounce_integral(f,h,x,index,root,is_func=True)
    return bounce_val
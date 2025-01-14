#!/usr/bin/env python3


# This file contains all subroutines used in the calculation bounce-averaged drifts
import  numpy           as      np
from    scipy.optimize  import  brentq
from    scipy.integrate import  quad

brentq_tol = 1e-20
ts_tol     = 1e-6



def _gtrapz(xi,xj,fi,fj,hi,hj):
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
    # include limit case where fi=fj
    ans = np.asarray(1/2 * (xj - xi) * (hi + hj)/ np.sqrt(1/2*(fi+fj)))
    # do division, keeping limit whenever fi=fj
    ans = np.divide( 2 * (xj - xi) * (hj * np.sqrt(fj) - hi * np.sqrt(fi))*(fj-fi) - 4/3 * (xj - xi) * (hj - hi) * (np.power(fj,3/2)- np.power(fi,3/2)), np.square(fj-fi),
                    out=ans,where=fi!=fj)       
    return np.sum(ans)


def _find_zeros(f,x,is_func=False,ignore_odd=False):
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
        ignore_odd: if False, an odd number of roots will raise an exception,
                    if True, an odd number of roots will return an empty list.
                    
    """
    # we store the roots in a list, as we don't know a priori how many there will be.
    # we also store the indices left of the zero root
    index_list = []
    roots_list = []
    # find all roots 
    if is_func==True:
        c = f(x)
        indi = np.where(c[1:]*c[0:-1] < 0.0)[0]
        for i in indi:
            u = brentq(f, x[i], x[i+1],xtol=brentq_tol)
            z = f(u)
            index_list.append(i)
            roots_list.append(u)
    # find all roots if is_func is False    
    if is_func==False:
        indi = np.where(f[1:]*f[0:-1] < 0.0)[0]
        for i in indi:
            u = x[i] - f[i] * (x[i+1] - x[i])/(f[i+1]-f[i])
            index_list.append(i)
            roots_list.append(u)

    # check if total number of roots is even.
    # edge cases with odd number of roots have NOT been implemented
    if len(roots_list) % 2 == 1:
        if ignore_odd==False:
            raise Exception("Odd number of bounce points, please adjust resolution or interpolation method.")
        if ignore_odd==True:
            print("Odd number of bounce points. Empty list returned.")

    # return all roots
    return index_list, roots_list


def _check_first_well(f,x,index,is_func=False):
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


def _bounce_integral(f,h,x,index,root,is_func=False,sinhtanh=False):
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
        index: indices left of the roots
        root: the values where f(x)=0
        is_func: are h and f functions or arrays.
        sinhtanh: use sinhtanh quadrature methods.
    """
    # define number of bounce wells
    num_wells = int(len(root)/2)

    # make list with integral values
    bounce_val = []

    # do integral with quadrature methods
    if is_func==True:
        # x coordinate for periodic boundary condition
        xmin = x[0]
        xmax = x[-1]
        if sinhtanh==False:
            # integrand
            integrand = lambda x : h(x)/np.sqrt(np.abs(f(x)))
            for well_idx in range(num_wells):
                l_bound  = root[2*well_idx]
                r_bound = root[2*well_idx+1]
                if l_bound > r_bound:
                    val_left, err   = quad(integrand,l_bound,xmax)    
                    val_right, err  = quad(integrand,xmin,r_bound)
                    val = val_left + val_right
                if l_bound < r_bound:
                    val, err  = quad(integrand,l_bound,r_bound)
                bounce_val.append(val)
        if sinhtanh==True:
            import  tanh_sinh       as      ts
            # first construct the well
            for well_idx in range(num_wells):
                l_bound  = root[2*well_idx]
                r_bound  = root[2*well_idx+1]
                # sinh-tanh needs to evaluate extremely close to the root
                # and as such the shrinking method is preferred.
                l_bound += np.sqrt(brentq_tol)
                r_bound -= np.sqrt(brentq_tol)
                # normal integral routine
                if l_bound < r_bound:
                    # map interval [0,1]->[l_bound,r_bound]
                    x_nrm     = lambda x: x * (r_bound - l_bound) + l_bound
                    # construct integrand
                    integrand_nrm = lambda x: h(x_nrm(x))/np.sqrt(np.abs(f(x_nrm(x))))
                    # integrate
                    val, _ = ts.integrate_lr(lambda x: integrand_nrm(x),
                                             lambda x: integrand_nrm(1.0-x),
                                             1.0,
                                             ts_tol)
                    val       = (val)*(r_bound - l_bound)
                # routine for edge well
                # NOT TESTED
                if l_bound > r_bound:
                    # map interval [0,1]->[l_bound,xmax]
                    x_nrm     = lambda x: x * (xmax - l_bound) + l_bound
                    # construct integrand, full function for debugging 
                    # purposes
                    def integrand(x):
                        val = h(x)/np.sqrt(np.abs(f(x)))
                        #print(val)
                        return val
                    # integrate
                    val_left, _ = ts.integrate_lr(lambda x: integrand(x_nrm(x)),
                                                  lambda x: integrand(1.0-x_nrm(x)),
                                                  1.0,
                                                  ts_tol)
                    val_left       = (val_left)*(xmax - l_bound)
                    x_nrm     = lambda x: x * (r_bound - xmin) + xmin
                    val_right, _ = ts.integrate_lr(lambda x: integrand_nrm(x),
                                                  lambda x: integrand_nrm(1.0-x),
                                                  1.0,
                                                  ts_tol)
                    val_right       = val_right*(r_bound - xmin)
                    val       = val_left + val_right
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
                inner_int = _gtrapz(xi,xj,fi,fj,hi,hj)
                # do left edge 
                left_int  = _gtrapz(l_bound,x[l_idx+1],0.0,f[l_idx+1],hl_cross,h[l_idx+1])
                # do right edge 
                right_int = _gtrapz(x[r_idx],r_bound,f[r_idx],0.0,h[r_idx],hr_cross)
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
                inner_int = _gtrapz(xi,xj,fi,fj,hi,hj)
                # do inner integral (start to r)
                xi = x[0:r_idx]
                fi = f[0:r_idx]
                hi = h[0:r_idx]
                xj = x[1:r_idx+1]
                fj = f[1:r_idx+1]
                hj = h[1:r_idx+1]
                inner_int = inner_int + _gtrapz(xi,xj,fi,fj,hi,hj)
                # do left edge 
                left_int  = _gtrapz(l_bound,x[l_idx+1],0.0,f[l_idx+1],hl_cross,h[l_idx+1])
                # do right edge 
                right_int = _gtrapz(x[r_idx],r_bound,f[r_idx],0.0,h[r_idx],hr_cross)
                # compute total int
                val = left_int + inner_int + right_int  
            # append value to bounce vals
            bounce_val.append(val)
    return bounce_val


def bounce_integral_wrapper(f,h,x,is_func=False,return_roots=False,sinhtanh=False,ignore_odd=False):
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
        index,root = _find_zeros(f,x,is_func=False,ignore_odd=ignore_odd)
        # check if first well is edge, if so roll
        first_well = _check_first_well(f,x,index,is_func=False)
        if first_well==False:
            index = np.roll(index,1)
            root = np.roll(root,1)
        # do bounce integral
        bounce_val = _bounce_integral(f,h,x,index,root,is_func=False,sinhtanh=False)
    # if is_func is true, use it for both root finding and integration
    if is_func==True: 
        index,root = _find_zeros(f,x,is_func=True,ignore_odd=ignore_odd)
        first_well = _check_first_well(f,x,index,is_func=True,)
        if first_well==False:
            index = np.roll(index,1)
            root = np.roll(root,1)
        bounce_val = _bounce_integral(f,h,x,index,root,is_func=True)
    if return_roots==False:
        return bounce_val
    if return_roots==True:
        return bounce_val, root
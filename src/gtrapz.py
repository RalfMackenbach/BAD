# This file contains all subroutines used in the calculation of the Available
# Energy of trapped particles. Based on fortran implementation
import  numpy           as      np
from    numba           import  jit
from    scipy.signal    import  find_peaks
import  scipy.integrate as      integrate
from    scipy           import  special
import  quadpy




@jit
def zero_cross_idx(y_arr):
    """
    Returns all indices left of zeros of y. Based on "Numerical Recipes in
    Fortran 90" (1996), chapter B9, p.1184. Also returns number of crossings.

    Takes as input:
    y_arr   -   array of which zero points are to be determined

    Returns:
    zero_idx, num_crossings
    """

    l_arr  = y_arr[0:-1]
    r_arr  = y_arr[1:]

    # check zero crossings, returns TRUE left of the crossing. Also, count number
    # of crossings
    mask      = l_arr*r_arr < 0
    # Return indices where mask is true, so left of crossings
    zero_idx = np.asarray(np.nonzero(mask))
    return zero_idx[0], np.sum(mask)


@jit
def inner_bounce_trapz(h_i,h_j,f_i,f_j,theta_i,theta_j):
    """
    Estimation of integral of h/sqrt(f) dtheta, where we assume h and f to be
    well approximated by their linear interpolation between theta_i and theta_j.
    Only estimate the INNER part (i.e. not accounting for the edges where f
    vanshises). These are handled seperately.
    """
    y = np.sum((-2*(np.sqrt(f_j)*(2*h_i + h_j) + np.sqrt(f_i)*(h_i + 2*h_j))*
          (theta_i - theta_j))/(3.*(f_i + f_j + 2*np.sqrt(f_i*f_j))))
    return y


@jit
def left_bounce_trapz(h_l,h_0,f_l,f_0,theta_l,theta_0):
    """
    Estimation of integral of edge integral h/sqrt(f) dtheta, from theta_l where
    f = 0, to the first theta node to its right theta_0
    """
    y = 2*(2*h_l + h_0)*(theta_0 - theta_l)/(3.*np.sqrt(f_0))
    return y


@jit
def right_bounce_trapz(h_n,h_r,f_n,f_r,theta_n,theta_r):
    """
    Estimation of integral of edge integral h/sqrt(f) dtheta, from theta_n to theta_r
    where f = 0
    """
    y = 2*(h_n + 2 * h_r)*(-theta_n + theta_r)/(3.*np.sqrt(f_n))
    return y


@jit
def bounce_wells(theta_arr,b_arr,lam_val):
    """
    ! This routine calculates the bounce points and turns them into bounce wells.
    ∫h(ϑ)/sqrt(1-lam*b(ϑ))·dϑ
    theta_arr   -   Array containing theta nodes
    h_arr       -   Array containing h values
    b_arr       -   Array containing normalized magnetic field values
    lam         -   Lambda value at which we wish to calculate quantities
    Returns two arrays and one scalar
    bounce_idx, bounce_arr, num_wells
    Each row of bounce_idx contains the two bounce indices left of the bounce points
    Each row of bounce_arr contains the two bounce points (in theta)
    Number of wells
    """
    # define function of which we need zero crossings, and retrieve crossings
    zero_arr = 1.0 - lam_val * b_arr
    zero_idx, num_cross = zero_cross_idx(zero_arr)
    # Check if even number of wells
    if (np.mod(num_cross,2)!=0):
        print('ERROR: odd number of well crossings, please adjust lambda resolution')
    # Calculate number of wells
    num_wells = int(num_cross/2)


    # Check if the first crossing is end of well
    if  ( b_arr[zero_idx[0]+1] - b_arr[zero_idx[0]] > 0 ):
        first_well_end = 1
    else:
        first_well_end = 0


    # First let's fill up the bounce_idx array
    # If well crosses periodicity we must shift the indices
    if (first_well_end == 1):
        zero_idx = np.roll(zero_idx,-1)

    # make array holding bounce well information
    bounce_idx = np.empty([num_wells,2],np.float_)
    bounce_arr = np.empty([num_wells,2],np.float_)

    # Fill up bounce array
    for  do_idx in range(0,num_wells):
      l_idx                     = zero_idx[2*do_idx]
      r_idx                     = zero_idx[2*do_idx+1]
      bounce_idx[do_idx,  0]    = l_idx
      bounce_idx[do_idx,  1]    = r_idx
      bounce_arr[do_idx,  0]    = (-(zero_arr[l_idx+1]*theta_arr[l_idx]) +
                                  zero_arr[l_idx]*theta_arr[l_idx+1])/(zero_arr[l_idx] -
                                  zero_arr[l_idx+1])
      bounce_arr[do_idx,  1]    = (-(zero_arr[r_idx+1]*theta_arr[r_idx]) +
                                  zero_arr[r_idx]*theta_arr[r_idx+1])/(zero_arr[r_idx] -
                                  zero_arr[r_idx+1])

    return bounce_idx, bounce_arr, num_wells


@jit
def bounce_average(theta_arr,h_arr,b_arr,lam):
    """
    Does the bounce averaging operation, i.e. calculates
    ∫h(ϑ)/sqrt(1-lam*b(ϑ))·dϑ
    theta_arr   -   Array containing theta nodes
    h_arr       -   Array containing h values
    b_arr       -   Array containing normalized magnetic field values
    lam         -   Lambda value at which we wish to calculate quantities
    """
    # Find the bounce wells
    bounce_idx, bounce_arr, num_wells = bounce_wells(theta_arr,b_arr,lam)
    bounce_ave = np.empty(num_wells,np.float_)
    f_arr   = 1 - lam*b_arr

    l_idx   = bounce_idx[:,0]
    r_idx   = bounce_idx[:,1]
    l_cross = bounce_arr[:,0]
    r_cross = bounce_arr[:,1]

    # check if well crosses periodicity boundary
    for  do_idx in range(0,num_wells):
        l = int(l_idx[do_idx])
        r = int(r_idx[do_idx])
        if (l_idx[do_idx]>r_idx[do_idx]):
            # Split up inner int into two parts
            # first left-to-end
            h_i     = h_arr[(l + 1):-1]
            h_j     = h_arr[(l + 2):]
            f_i     = f_arr[(l + 1):-1]
            f_j     = f_arr[(l + 2):]
            theta_i = theta_arr[(l + 1):-1]
            theta_j = theta_arr[(l + 2):]
            y_l = inner_bounce_trapz(h_i,h_j,f_i,f_j,theta_i,theta_j)
            # then start-to-right
            h_i     = h_arr[(0):(r)]
            h_j     = h_arr[(1):(r+1)]
            f_i     = f_arr[(0):(r)]
            f_j     = f_arr[(1):(r+1)]
            theta_i = theta_arr[(0):(r)]
            theta_j = theta_arr[(1):(r+1)]
            y_r = inner_bounce_trapz(h_i,h_j,f_i,f_j,theta_i,theta_j)
            inner = y_l + y_r
    # otherwise business as usual
        else:
            h_i     = h_arr[(l + 1):(r)]
            h_j     = h_arr[(l + 2):(r+1)]
            f_i     = f_arr[(l + 1):(r)]
            f_j     = f_arr[(l + 2):(r+1)]
            theta_i = theta_arr[(l + 1):(r)]
            theta_j = theta_arr[(l + 2):(r+1)]
            inner = inner_bounce_trapz(h_i,h_j,f_i,f_j,theta_i,theta_j)


        # Now do the edge integrals
        h_l = h_arr[l] + (l_cross[do_idx] -
              theta_arr[l])/(theta_arr[l+1] -
              theta_arr[l]) * ( h_arr[l+1] -
              h_arr[l] )
        left = left_bounce_trapz(h_l,h_arr[l+1],0.0,
                               f_arr[l+1],l_cross[do_idx],
                               theta_arr[l+1])
        h_r = h_arr[r] + (r_cross[do_idx] -
              theta_arr[r])/(theta_arr[r+1] -
              theta_arr[r]) * ( h_arr[r+1] -
              h_arr[r] )
        right = right_bounce_trapz(h_arr[r],h_r,f_arr[r],
                                0.0,theta_arr[r],r_cross[do_idx])

        # finally, fill in full integral!
        bounce_ave[do_idx]= left + inner + right

    return bounce_ave


@jit
def w_bounce(h_arr,b_arr,f_arr,zeta_arr,lam):
    """
    Calculate the drift frequencies and bounce time. Implemented as
    bounce time = ∫h(ζ)/sqrt(1-lam*modb(ζ))·dζ
    averaged f  = ∫f(ζ)*h(ζ)/sqrt(1-lam*modb(ζ))·dζ
    """
    # Bounce time
    denom_arr = h_arr
    denom = bounce_average(zeta_arr,denom_arr,b_arr,lam)
    # Integrated f
    numer_arr = h_arr*f_arr
    numer = bounce_average(zeta_arr,numer_arr,b_arr,lam)
    # return numer and denom
    return numer, denom




@jit
def make_per(b_arr,L2,L1,sqrtg_arr,theta_arr,Delta_theta):
    """
    Makes arrays periodic by appending first value to last, and for
    theta a small padding region of theta_last + Delta_theta is added.
    """
    # make periodic
    b_arr_p    = np.append(b_arr,b_arr[0])
    dbdx_arr_p = np.append(L2,L2[0])
    dbdy_arr_p = np.append(L1,L1[0])
    sqrtg_arr_p= np.append(sqrtg_arr,sqrtg_arr[0])
    theta_arr_p= np.append(theta_arr,theta_arr[-1]+Delta_theta)

    return b_arr_p, dbdx_arr_p, dbdy_arr_p, sqrtg_arr_p, theta_arr_p


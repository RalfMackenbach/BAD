import numpy as np
import scipy.interpolate as spi
import scipy.optimize as spo
import termcolor as tc
from BAD.util_integrals import _get_abc

# Here we define functions to extract quantities relevant for a bounce-integral given a field-line
# we assume we are given:
#   a magnetic field line B(z)
#   a function h(z) that is to be bounce-averaged
# From these, we can calculate bounce-points (given either lambda or one of the bounce-points) and do the bounce-integral

# Notation:
#   lam is lambda
#   z is the coordinate along the field-line
#   B is the magnetic field strength
#   h is the function to be bounce-averaged
#   f is the parallel energy (f = 1 - lam * B)



def check_valid_lambda(B, lam):
    # check if lambda is valid
    # lamdba in (1/Bmax, 1/Bmin)
    Bmax = np.max(B)
    Bmin = np.min(B)
    boolean = np.logical_and(lam > 1/Bmax, lam < 1/Bmin)
    return boolean



def find_zeros_linear(f, z):
    # find the zeros of an array f(z)
    sgn_f = np.sign(f)
    # exact zeros
    idc_exact = np.where(sgn_f == 0)[0]
    z_exact = z[idc_exact]
    # index of approximate zeros, disregarding the exact zeros
    idc_approx = np.where(np.abs(np.diff(sgn_f)) == 2)[0]
    z_left = z[idc_approx]
    z_right = z[idc_approx+1]
    f_left = f[idc_approx]
    f_right = f[idc_approx+1]
    z_approx = z_left - f_left*(z_right - z_left)/(f_right - f_left)

    return z_exact, z_approx, idc_exact, idc_approx



def df_exact_zero(f, z, idx_exact):
    # calculate the derivative of f at the exact zeros
    df_exact = np.zeros_like(idx_exact, dtype=float)
    for i, idx in enumerate(idx_exact):
        if idx == 0:
            df_exact[i] = f[idx + 1] - f[idx]
        elif idx == len(z) - 1:
            df_exact[i] = f[idx] - f[idx - 1]
        else:
            df_forward = f[idx + 1] - f[idx]
            df_backward = f[idx] - f[idx - 1]
            df_exact[i] = 0.5 * (df_forward + df_backward)
    return df_exact



def check_exact_extremum(f, z, idx_exact):
    # check if the exact zeros are local extrema
    extremum_flags = np.zeros_like(idx_exact, dtype=bool)
    for i, idx in enumerate(idx_exact):
        if idx == 0:
            extremum_flags[i] = (f[idx + 1] * f[idx] == 0)
        elif idx == len(z) - 1:
            extremum_flags[i] = (f[idx] * f[idx - 1] == 0)
        else:
            extremum_flags[i] = (f[idx + 1] * f[idx - 1] >= 0)
    return extremum_flags



def bounce_points_f(f, z):
    # given grids f = 1 - lam * B and z, find the bounce points for a given lambda
    # incomplete orbits will be handled by NaN boundary conditions 
    

    # find the zeros of f
    z_exact, z_approx, idc_exact, idc_approx = find_zeros_linear(f, z)

    # check if there is an extremum in the exact zeros, if so print warning message and return NaNs
    # otherwise bounce-point pairing is a difficult problem
    extremum_flags = check_exact_extremum(f, z, idc_exact)
    if np.any(extremum_flags):
        print(tc.colored('Warning: exact zeros are local extrema', 'red'))
        return np.asarray([[np.nan, np.nan]]), np.asarray([[np.nan, np.nan]])

    # save the discrete derivative of f at these points
    df_exact = df_exact_zero(f, z, idc_exact)
    df_approx = np.diff(f)[idc_approx]

    # take the sign function of the derivative
    df_exact = np.sign(df_exact)
    df_approx = np.sign(df_approx)
    
    # now sort z_exact and z_approx, and the indices and derivatives accordingly
    z_b = np.concatenate((z_exact, z_approx))
    idc = np.concatenate((idc_exact, idc_approx))
    df = np.concatenate((df_exact, df_approx))
    idx = np.argsort(z_b)
    z_b = z_b[idx]
    idc = idc[idx]
    df = df[idx]

    # we now construct bounce pairs
    # pairing rules: pairs can only be of the form df = [1,-1]
    # now loop through the pairs
    z_pairs = []
    df_pairs = []
    for i, z_val in enumerate(z_b):
        # check if there are unpaired points at the boundary
        # this would mean that the first bounce point
        # has df = -1
        if i == 0:
            if (df[i] == -1):
                z_pairs.append([np.nan, z_val])
                df_pairs.append([np.nan, df[i]])

        # similarly for the right boundary
        # if df = 1
        if i == len(z_b)-1:
            if (df[i] == 1):
                z_pairs.append([z_val, np.nan])
                df_pairs.append([df[i], np.nan])
        
        # last point is never paired with the next point
        if i == len(z_b)-1:
            continue
        
        # now check if we can pair with the next point
        df_left = df[i]
        df_right = df[i+1]
        if [df_left, df_right] == [1,-1]:
            z_pairs.append([z_val, z_b[i+1]])
            df_pairs.append([df_left, df_right])
    
    # convert to numpy arrays of shape (n_pairs, 2)
    z_pairs = np.asarray(z_pairs)
    df_pairs = np.asarray(df_pairs)
    z_pairs = z_pairs.reshape(-1,2)
    df_pairs = df_pairs.reshape(-1,2)

    return z_pairs, df_pairs



def apply_boundary_condition(f, z, z_pairs, df_pairs, bc='periodic'):
    # given a field-line f = 1 - lam * B 
    # and bounce points z_pairs, df_pairs
    
    # if all are NaN, return NaN
    if np.all(np.isnan(z_pairs)):
        return np.asarray([[np.nan, np.nan]]), np.asarray([[np.nan, np.nan]])

    if bc == 'wall':
        # if the left boundary is NaN, we set it to the first point
        if np.isnan(z_pairs[0,0]):
            z_pairs[0,0] = z[0]
            df_pairs[0,0] = 1
        # if the right boundary is NaN, we set it to the last point
        if np.isnan(z_pairs[-1,1]):
            z_pairs[-1,1] = z[-1]
            df_pairs[-1,1] = -1
    
    if bc == 'periodic':
        # check if the periodic boundary condition introduces a new extremum at an exact zero
        # only possible if the left-most or right-most z corresponds to a bounce-point
        # we check if the left-most z is a bounce-point
        if z[0] in z_pairs:
            # check if the surrounding values are both positive or both negative
            if f[1]*f[-1] >= 0:
                # if they are, a new extremum is introduced
                print(tc.colored('Warning: periodic boundary condition introduces a new extremal bounce-point at z = {}'.format(z[0]), 'red'))
                return np.asarray([[np.nan, np.nan]]), np.asarray([[np.nan, np.nan]])
        # we check if the right-most z is a bounce-point
        if z[-1] in z_pairs:
            # check if the surrounding values are both positive or both negative
            if f[0]*f[-2] >= 0:
                # if they are, a new extremum is introduced
                print(tc.colored('Warning: periodic boundary condition introduces a new extremum at z = {}'.format(z[-1]), 'red'))
                return np.asarray([[np.nan, np.nan]]), np.asarray([[np.nan, np.nan]])

        # if both boundaries are NaN, they are connected
        if np.isnan(z_pairs[0,0]) and np.isnan(z_pairs[-1,1]):
            # we set the right-most boundary equal to the left bounce-point
            z_pairs[-1,1] = z_pairs[0,1]
            df_pairs[-1,1] = df_pairs[0,1]
            # we ignore the first bounce-point
            z_pairs = z_pairs[1:]
            df_pairs = df_pairs[1:]
        # if only the left boundary is NaN, the crossing must happen in between the first and last bounce-point
        elif np.isnan(z_pairs[0,0]):
            # update the left boundary
            z_pairs[0,0] = z[0]
            df_pairs[0,0] = 1
        # if only the right boundary is NaN, the crossing must happen in between the first and last bounce-point
        elif np.isnan(z_pairs[-1,1]):
            # update the right boundary
            z_pairs[-1,1] = z[-1]
            df_pairs[-1,1] = -1

        elif bc == None:
            pass

    return np.asarray(z_pairs), np.asarray(df_pairs)



def construct_well_arr(f, h, z, z_pair):
    # given arrays f and z, and bounce points z_pairs
    # define left and right boundaries
    z_left = z_pair[0]
    z_right = z_pair[1]
    # check if z_left or z_right are NaN
    if np.any(np.isnan(z_pair)):
        # return NaN arrays
        z_grid = np.array([[np.nan]])
        f_grid = np.array([[np.nan]])
        h_grid = np.array([[np.nan]])
        return np.asarray(z_grid), np.asarray(f_grid), np.asarray(h_grid)
    
    # if z_left < z_right, we construct the inner grid
    if z_left < z_right:
        idc_inner = np.where(np.logical_and(z > z_left, z < z_right))
        z_grid = z[idc_inner]
        f_grid = f[idc_inner]
        h_grid = h[idc_inner]

        # complete the grid with the boundaries
        z_grid = [np.concatenate(([z_left], z_grid, [z_right]))]
        f_left = np.abs(np.interp(z_left, z, f))
        f_right = np.abs(np.interp(z_right, z, f))
        h_left = np.interp(z_left, z, h)
        h_right = np.interp(z_right, z, h)
        f_grid = [np.concatenate(([f_left], f_grid, [f_right]))]
        h_grid = [np.concatenate(([h_left], h_grid, [h_right]))]

    # if z_left > z_right (only possible for periodic boundary conditions)
    # we split the grid in two
    if z_left > z_right:
        idc_left = np.where(z > z_left)
        idc_right = np.where(z < z_right)
        z_left_grid = z[idc_left]
        f_left_grid = f[idc_left]
        h_left_grid = h[idc_left]
        z_right_grid = z[idc_right]
        f_right_grid = f[idc_right]
        h_right_grid = h[idc_right]
        # complete the grids with the boundaries
        f_left = np.abs(np.interp(z_left, z, f))
        f_right = np.abs(np.interp(z_right, z, f))
        h_left = np.interp(z_left, z, h)
        h_right = np.interp(z_right, z, h)
        z_left_grid = np.concatenate(([z_left], z_left_grid))
        f_left_grid = np.concatenate(([f_left], f_left_grid))
        h_left_grid = np.concatenate(([h_left], h_left_grid))
        z_right_grid = np.concatenate((z_right_grid, [z_right]))
        f_right_grid = np.concatenate((f_right_grid, [f_right]))
        h_right_grid = np.concatenate((h_right_grid, [h_right]))

        z_grid = [z_left_grid, z_right_grid]
        f_grid = [f_left_grid, f_right_grid]
        h_grid = [h_left_grid, h_right_grid]

   

    return z_grid, f_grid, h_grid



def construct_well_func(z, z_pair):
    # given z, and bounce points z_pairs
    # give the range of integration as pairs
    z_left = z_pair[0]
    z_right = z_pair[1]
    # check if z_left or z_right are NaN
    if np.any(np.isnan(z_pair)):
        # return NaN arrays
        z_grid = np.array([[np.nan]])
        f_grid = np.array([[np.nan]])
        h_grid = np.array([[np.nan]])
        return np.asarray(z_grid), np.asarray(f_grid), np.asarray(h_grid)
    
    # if z_left < z_right we're done
    if z_left < z_right:
        return [[z_left, z_right]]
    
    # if z_left > z_right, we split the grid in two
    if z_left > z_right:
        return [[z_left, np.max(z)], [np.min(z), z_right]]
        


def refine_roots(f_func,z_init,eps=1e-5):
    # refine the roots of a function f_func(z) using scipy.optimize.root_scalar
    # z_init is the initial guess for the roots
    # returns the refined roots
    tol = 1e-16
    # Compute x1 for root_scalar (so that compatible with older versions of scipy)
    p1 = z_init * (1 + eps)
    p1 += (eps if p1 >= 0 else -eps)
    res = spo.root_scalar(f_func, x0=z_init, x1 = p1, xtol=tol)
    return res.root



def make_well_given_bp(B, z, z_bp, boundary='periodic'):
    # check if B is a function
    if callable(B):
        # get B(z_bp) and B(z)
        B_bp = B(z_bp)
        B_z = B(z)
        lam = 1.0/B_bp
        f_arr = 1.0 - lam * B_z
        # find the bounce-points
        z_pairs, df_pairs = bounce_points_f(f_arr, z)
        # refine the bounce-points
        f_func = lambda z: 1.0 - lam * B(z)
        for idx, z_init in np.ndenumerate(z_pairs):
            if not np.isnan(z_init):
                z_pairs[idx] = refine_roots(f_func, z_init)
        # apply boundary conditions
        z_pairs, df_pairs = apply_boundary_condition(f_arr, z, z_pairs, df_pairs, bc=boundary)

    if not callable(B):
        # get B(z_bp) with np.interp
        B_bp = np.interp(z_bp, z, B)
        lam = 1.0/B_bp
        f_arr = 1.0 - lam * B
        # find the bounce-points
        z_pairs, df_pairs = bounce_points_f(f_arr, z)
        # apply boundary conditions
        z_pairs, df_pairs = apply_boundary_condition(f_arr, z, z_pairs, df_pairs, bc=boundary)

    # check which pair has z_bp as z_left or z_right (within a tolerance)
    tol = 1e-10
    for z_pair in z_pairs:
        if np.abs(z_pair[0] - z_bp) < tol:
            return z_pair
        if np.abs(z_pair[1] - z_bp) < tol:
            return z_pair

    return np.asarray([[np.nan, np.nan]])



def linear_bounce_wells_wrapper(f_arr, h_arr, z_arr, boundary='periodic'):
    # given an array of f_arr = 1 - lam * B(z) and z_arr
    # find the bounce-points and construct the bounce-wells
    # first find the bounce-points
    z_pairs, df_pairs = bounce_points_f(f_arr, z_arr)
    # apply boundary conditions
    z_pairs, df_pairs = apply_boundary_condition(f_arr, z_arr, z_pairs, df_pairs, bc=boundary)
    # construct the bounce-wells
    z_wells = []
    f_wells = []
    h_wells = []
    for z_pair in z_pairs:
        z_well, f_well, h_well = construct_well_arr(f_arr, h_arr, z_arr, z_pair)
        z_wells.append(z_well)
        f_wells.append(f_well)
        h_wells.append(h_well)

    return z_wells, f_wells, h_wells



def func_bounce_wells_wrapper(f_func, h_func, z, boundary='periodic'):
    # given a function f_func = 1 - lam * B(z) and z
    # find the bounce-points and construct the bounce-wells
    # first find the bounce-points
    f_arr = f_func(z)
    h_arr = h_func(z)
    z_pairs, df_pairs = bounce_points_f(f_arr, z)
    # refine the bounce-points 
    for idx, z_init in np.ndenumerate(z_pairs):
        if not np.isnan(z_init):
            z_pairs[idx] = refine_roots(f_func, z_init)
    # apply boundary conditions
    z_pairs, df_pairs = apply_boundary_condition(f_arr, z, z_pairs, df_pairs, bc=boundary)
    # construct the bounce-wells
    z_wells = []
    for z_pair in z_pairs:
        z_well = construct_well_func(z, z_pair)
        z_wells.append(z_well)


    return z_wells
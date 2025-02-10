from BAD.util_fieldline import *
from BAD.bounce_integrals import *


def bounce_int_lambda(B, h, z, lam, mode='fast', boundary_condition='periodic'):
    # check if B and h are both functions
    if callable(B) and callable(h):
        f_func = lambda z: 1 - lam * B(z)
        z_wells = func_bounce_wells_wrapper(f_func, h, z, boundary=boundary_condition)
        integrals = []
        for z_well in z_wells:
            res = 0.0
            for z_wp in z_well:
                res += bounce_integral(f_func, h, x_l=z_wp[0], x_r=z_wp[1], mode=mode)
            integrals.append(res)
        # create dictionary
        res = {}
        # to dictionary z_wells can be created by taking the first and last element of each z_well
        # i.e.
        # [[[np.float64(-0.9625507478846871), np.float64(0.9625507478846871)]], [[np.float64(5.320634559294899), np.float64(6.283185307179586)], [np.float64(-6.283185307179586), np.float64(-5.320634559294899)]]]
        # becomes
        # [[-0.9625507478846871, 0.9625507478846871], [-6.283185307179586, -5.320634559294899]]
        z_wells_dict = [[z_well[0][0], z_well[-1][1]] for z_well in z_wells]
        res['z_wells'] = z_wells_dict
        res['integrals'] = integrals
        res['lambda'] = lam

    elif not callable(B) and not callable(h):
        f = 1 - lam * B
        z_wells, f_wells, h_wells = linear_bounce_wells_wrapper(f, h, z, boundary=boundary_condition)
        integrals = []
        for z_well, f_well, h_well in zip(z_wells, f_wells, h_wells):
            res = 0.0
            for z_wp, f_wp, h_wp in zip(z_well, f_well, h_well):
                res += bounce_integral(f_wp, h_wp, x=z_wp, mode=mode)
            integrals.append(res)
        # create dictionary
        res = {}
        # to dictionary z_wells can be created by taking the first and last element of each z_well
        # i.e.
        # [[array([-0.95140366, -0.83775804, -0.41887902,  0.        ,  0.41887902, 0.83775804,  0.95140366])], [array([5.33178165, 5.44542727, 5.86430629, 6.28318531]), array([-6.28318531, -5.86430629, -5.44542727, -5.33178165])]]
        # becomes
        # [[-0.95140366, 0.95140366], [5.33178165, -5.33178165]]
        z_wells_dict = [[z_well[0][0], z_well[-1][-1]] for z_well in z_wells]
        res['z_wells'] = z_wells_dict
        res['integrals'] = integrals
        res['lambda'] = lam

    else:
        raise ValueError('B and h must be both functions or arrays')
    
    return res



def bounce_int_zbp(B, h, z, zbp, mode='fast', boundary_condition='periodic'):
    # get the bounce-well
    z_well = make_well_given_bp(B, z, zbp, boundary=boundary_condition)
    # check if B is callable
    if callable(B):
        lam = 1.0/B(zbp)
        f_func = lambda z: 1 - lam * B(z)
        # construct the well
        z_range = construct_well_func(z, z_well)
        # loop over well parts
        res = 0.0
        for z_wp in z_range:
            res += bounce_integral(f_func, h, x_l=z_wp[0], x_r=z_wp[1], mode=mode)

        z_well_dict = [z_well[0], z_well[-1]]

    else:
        lam = 1.0/np.interp(zbp, z, B)
        # construct the well #construct_well_arr(f, h, z, z_pair)
        f_arr = 1 - lam * B
        h_arr = h 
        z_well, f_well, h_well = construct_well_arr(f_arr, h_arr, z, z_well)
        # loop over well parts
        res = 0.0
        for z_wp, f_wp, h_wp in zip(z_well, f_well, h_well):
            res += bounce_integral(f_wp, h_wp, x=z_wp, mode=mode)

        z_well_dict = [z_well[0][0], z_well[-1][-1]]


    # construct dictionary
    res_dict = {}
    res_dict['z_well'] = z_well_dict
    res_dict['integral'] = res
    res_dict['lambda'] = lam

    return res_dict

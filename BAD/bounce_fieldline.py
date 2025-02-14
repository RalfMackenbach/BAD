from BAD.util_fieldline import *
from BAD.bounce_integrals import *
from scipy.interpolate import interp1d


def bounce_int_lambda(B, h, z, lam, mode='fast', boundary_condition='periodic'):
    # check if h is a list, if not make it a list
    if not isinstance(h, list):
        h = [h]
    # check if B and all h are functions
    if callable(B) and all([callable(h_i) for h_i in h]):
        f_func = lambda z: 1 - lam * B(z)
        z_wells = func_bounce_wells_wrapper(f_func, z, boundary=boundary_condition)
        # make array to store integrals (len(z_wells)xlen(h))
        integrals = np.zeros((len(z_wells), len(h)))
        for idx, _ in np.ndenumerate(integrals):
            z_well = z_wells[idx[0]]
            for z_wp in z_well:
                integrals[idx] += bounce_integral(f_func, h[idx[1]], x_l=z_wp[0], x_r=z_wp[1], mode=mode)
        # create dictionary
        res = {}
        # to dictionary z_wells can be created by taking the first and last element of each z_well
        # i.e.
        # [[[np.float64(-0.9625507478846871), np.float64(0.9625507478846871)]], [[np.float64(5.320634559294899), np.float64(6.283185307179586)], [np.float64(-6.283185307179586), np.float64(-5.320634559294899)]]]
        # becomes
        # [[-0.9625507478846871, 0.9625507478846871], [-6.283185307179586, -5.320634559294899]]
        z_wells_dict = [[z_well[0][0], z_well[-1][1]] for z_well in z_wells]
        res['z_wells'] = z_wells_dict
        res['integrals'] = np.asarray(integrals)
        res['lambda'] = lam

    # check if B and all h are all arrays
    elif isinstance(B, np.ndarray) and all([isinstance(h_i, np.ndarray) for h_i in h]):
        f = 1 - lam * B
        z_wells, f_wells, hs_wells = linear_bounce_wells_wrapper(f, h, z, boundary=boundary_condition)
        if mode == 'accurate':
            # refine the roots using quadratic interpolation
            f_quad = interp1d(z, f, kind='quadratic')
            h_quad = [interp1d(z, h_i, kind='quadratic') for h_i in h]
            # loop over bounce points
            for i, z_well in enumerate(z_wells):
                z_left = z_well[0][0]
                z_right = z_well[-1][-1]
                # refine these bounce points
                if z_left != z[0] and z_left != z[-1]:
                    z_wells[i][0][0] = refine_roots(f_quad, z_left)
                    for j, h_quad_i in enumerate(h_quad):
                        hs_wells[j][i][0][0] = h_quad_i(z_wells[i][0][0])
                if z_right != z[0] and z_right != z[-1]:
                    z_wells[i][-1][-1] = refine_roots(f_quad, z_right)
                    for j, h_quad_i in enumerate(h_quad):
                        hs_wells[j][i][-1][-1] = h_quad_i(z_wells[i][-1][-1])


            

            
                

        # make array to store integrals (len(z_wells)xlen(h))
        integrals = np.zeros((len(z_wells), len(h)))
        for idx, _ in np.ndenumerate(integrals):
            z_well = z_wells[idx[0]] 
            f_well = f_wells[idx[0]]
            h_well = hs_wells[idx[1]][idx[0]]
            for z_wp, f_wp, h_wp in zip(z_well, f_well, h_well):
                integrals[idx] += bounce_integral(f_wp, h_wp, x=z_wp, mode=mode)
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
    # check if h is a list, if not make it a list
    if not isinstance(h, list):
        h = [h]
    
    # get the bounce-well
    z_pair = make_well_given_bp(B, z, zbp, boundary=boundary_condition)

    # check if B and all h are functions
    if callable(B) and all([callable(h_i) for h_i in h]):
        lam = 1.0/B(zbp)
        f_func = lambda z: 1 - lam * B(z)
        # construct the well
        z_range = construct_well_func(z, z_pair)
        # make array to store integrals (len(z_well)xlen(h))
        integrals = np.zeros(len(h))
        for idx, h_i in enumerate(h):
            # loop over well parts
            for z_wp in z_range:
                integrals[idx] += bounce_integral(f_func, h_i, x_l=z_wp[0], x_r=z_wp[1], mode=mode)

        z_well_dict = [z_pair[0], z_pair[-1]]

    # check if B and all h are all arrays
    elif isinstance(B, np.ndarray) and all([isinstance(h_i, np.ndarray) for h_i in h]):
        lam = 1.0/np.interp(zbp, z, B)
        # construct the well
        f_arr = 1 - lam * B
        h_arr = h
        h_wells = []
        z_well, f_well, _ = construct_well_arr(f_arr, h_arr[0], z, z_pair)
        for i, h_grid in enumerate(h_arr):
            _, _, h_grid = construct_well_arr(f_arr, h_grid, z, z_pair)
            h_wells.append(h_grid)

        if mode == 'accurate':
            # refine the roots using quadratic interpolation
            f_quad = interp1d(z, f_arr, kind='quadratic')
            h_quad = [interp1d(z, h_i, kind='quadratic') for h_i in h_arr]
            # refine these bounce points
            z_left = z_well[0][0]
            z_right = z_well[-1][-1]
            if z_left != z[0] and z_left != z[-1]:
                z_well[0][0] = refine_roots(f_quad, z_left)
                for j, h_quad_i in enumerate(h_quad):
                    h_wells[j][0][0] = h_quad_i(z_well[0][0])
            if z_right != z[0] and z_right != z[-1]:
                z_well[-1][-1] = refine_roots(f_quad, z_right)
                for j, h_quad_i in enumerate(h_quad):
                    h_wells[j][-1][-1] = h_quad_i(z_well[-1][-1])

            
        # make array to store integrals (len(z_well)xlen(h))
        integrals = np.zeros(len(h))
        # loop over well parts
        for idx, h_i in enumerate(h):
            for z_wp, f_wp, h_wp in zip(z_well, f_well, h_wells[idx]):
                integrals[idx] += bounce_integral(f_wp, h_wp, x=z_wp, mode=mode)


        z_well_dict = [z_well[0][0], z_well[-1][-1]]


    # construct dictionary
    res_dict = {}
    res_dict['z_well'] = z_well_dict
    res_dict['integrals'] = integrals
    res_dict['lambda'] = lam

    return res_dict
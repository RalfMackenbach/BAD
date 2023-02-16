import sys
sys.path.append('/Users/ralfmackenbach/Documents/GitHub/Bounce-averaged-drift/BAD/src/')  
import bounce_int
import numpy as np
import time
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib        as mpl


# B hat from the square well example
def modb(l,al=1.0):
    return 1 + al * l**2

# radial derivative of modB of square well example
def dbdpsi(l,apsi=1.5,b=1.0):
    return apsi * ( l**2 - b )

# analytical bounce-time
def bounce_time(lam_val,al=1.0):
    return np.pi/np.sqrt(lam_val*al)

# analytical Delta alpha
def delta_alpha(lam_val,al=1.0,apsi=1.5,b=1.0):
    return np.pi * apsi * (lam_val + 2 * lam_val * b * al - 1) / ( 2 * al**1.5 * lam_val**0.5 )

# analytical bounce averaged drift
def drift(lam_val,al=1.0,apsi=1.5,b=1.0):
    return apsi * ( 2 * b * al * lam_val + lam_val - 1.0 ) / ( 2 * al )





# loop over res as well
res = np.rint(np.logspace(1,6,6,endpoint=True)).astype(int)+1
print(res)
gtrapz_err = np.zeros(len(res))
quad_err = np.zeros(len(res))
qquad_err = np.zeros(len(res))
gtrapz_time = np.zeros(len(res))
quad_time = np.zeros(len(res))
qquad_time = np.zeros(len(res))

# first do gtrapz method
for res_idx, res_val in enumerate(res):
    l_arr       = np.linspace(-10,10,res_val)
    # dl/dl is unity
    dldl       = np.ones_like(l_arr)
    # mod b
    modb_arr    = modb(l_arr)
    # dbdpsi 
    dbdpsi_arr  = dbdpsi(l_arr)
    # make array with lambdas and bounce-averaged quantities
    # we exclude the endpoint and starting point
    lam_arr     = np.linspace(1/np.max(modb_arr),1/np.min(modb_arr),99,endpoint=False)
    lam_arr     = np.delete(lam_arr,  0)
    gtrapz_num  = np.empty_like(lam_arr)
    gtrapz_den  = np.empty_like(lam_arr)
    gtrapz_ave  = np.empty_like(lam_arr)
    # call on function once already to make sure all is loaded
    tau_b   = bounce_int.bounce_integral_wrapper(1.0 - lam_arr[1]*modb_arr,dldl,l_arr,is_func=False)
    # now calculate for various lambda
    start_time = time.time()
    for idx, lam_val in enumerate(lam_arr):
        f = 1.0 - lam_val*modb_arr
        tau_b   = bounce_int.bounce_integral_wrapper(f,dldl,l_arr,is_func=False)
        d_alpha = bounce_int.bounce_integral_wrapper(f,-1.0*lam_val*dbdpsi_arr,l_arr,is_func=False)
        gtrapz_den[idx] = tau_b[0]
        gtrapz_num[idx] = d_alpha[0]
        gtrapz_ave[idx] = d_alpha[0]/tau_b[0]
    tot_time = time.time() - start_time
    # calculate error
    true_res = drift(lam_arr)
    gtrapz_err[res_idx] = np.average(np.abs((gtrapz_ave-true_res)/true_res))
    gtrapz_time[res_idx] = tot_time



# now use quad method
for res_idx, res_val in enumerate(res):
    l_arr       = np.linspace(-10,10,res_val)
    # dl/dl is unity
    dldl       = np.ones_like(l_arr)
    # mod b
    modb_arr    = modb(l_arr)
    # dbdpsi 
    dbdpsi_arr  = dbdpsi(l_arr)
    # make array with lambdas and bounce-averaged quantities
    # we exclude the endpoint and starting point
    lam_arr     = np.linspace(1/np.max(modb_arr),1/np.min(modb_arr),99,endpoint=False)
    lam_arr     = np.delete(lam_arr,  0)
    quad_num  = np.empty_like(lam_arr)
    quad_den  = np.empty_like(lam_arr)
    quad_ave  = np.empty_like(lam_arr)
    # make interpolated functions
    modb_interp = interp.PchipInterpolator(l_arr, modb_arr)
    dldldbdpsi_interp = interp.PchipInterpolator(l_arr, dldl*dbdpsi_arr)
    dldl_interp = interp.PchipInterpolator(l_arr, dldl)
    # now calculate for various lambda
    start_time = time.time()
    for idx, lam_val in enumerate(lam_arr):
        f = lambda x: 1.0 - lam_val*modb_interp(x)
        h_alpha = lambda x: -1.0 * lam_val * dldldbdpsi_interp(x)
        tau_b   = bounce_int.bounce_integral_wrapper(f,dldl_interp,l_arr,is_func=True)
        d_alpha = bounce_int.bounce_integral_wrapper(f,h_alpha,l_arr,is_func=True)
        quad_den[idx] = tau_b[0]
        quad_num[idx] = d_alpha[0]
        quad_ave[idx] = d_alpha[0]/tau_b[0]
    tot_time = time.time() - start_time
    # calculate error
    true_res = drift(lam_arr)
    quad_err[res_idx] = np.average(np.abs((quad_ave-true_res)/true_res))
    quad_time[res_idx] = tot_time


# now use qquad method (quadratic interpolator)
for res_idx, res_val in enumerate(res):
    l_arr       = np.linspace(-2,2,res_val)
    # dl/dl is unity
    dldl       = np.ones_like(l_arr)
    # mod b
    modb_arr    = modb(l_arr)
    # dbdpsi 
    dbdpsi_arr  = dbdpsi(l_arr)
    # make array with lambdas and bounce-averaged quantities
    # we exclude the endpoint and starting point
    lam_arr     = np.linspace(1/np.max(modb_arr),1/np.min(modb_arr),99,endpoint=False)
    lam_arr     = np.delete(lam_arr,  0)
    qquad_num  = np.empty_like(lam_arr)
    qquad_den  = np.empty_like(lam_arr)
    qquad_ave  = np.empty_like(lam_arr)
    # make interpolated functions
    kind='cubic'
    modb_interp = interp.interp1d(l_arr, modb_arr,kind=kind)
    dldldbdpsi_interp = interp.interp1d(l_arr, dldl*dbdpsi_arr,kind=kind)
    dldl_interp = interp.interp1d(l_arr, dldl,kind=kind)
    # now calculate for various lambda
    start_time = time.time()
    for idx, lam_val in enumerate(lam_arr):
        f = lambda x: 1.0 - lam_val*modb_interp(x)
        h_alpha = lambda x: -1.0 * lam_val * dldldbdpsi_interp(x)
        tau_b   = bounce_int.bounce_integral_wrapper(f,dldl_interp,l_arr,is_func=True)
        d_alpha = bounce_int.bounce_integral_wrapper(f,h_alpha,l_arr,is_func=True)
        qquad_den[idx] = tau_b[0]
        qquad_num[idx] = d_alpha[0]
        qquad_ave[idx] = d_alpha[0]/tau_b[0]
    tot_time = time.time() - start_time
    # calculate error
    true_res = drift(lam_arr)
    qquad_err[res_idx] = np.average(np.abs((qquad_ave-true_res)/true_res))
    qquad_time[res_idx] = tot_time


# Plotting parameters 
plt.close('all')

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10}

mpl.rc('font', **font)

fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(3.5, 2*2.5))


ax[0].set_aspect('auto')
ax[1].set_aspect('auto')


ax[0].loglog(res,10*gtrapz_err[-1]*(res[-1]/res)**1.5,color='black',linestyle='dotted')
ax[0].loglog(res,gtrapz_err,label='gtrapz')
ax[0].loglog(res,quad_err,label='m-quad',linestyle='dashed')
ax[0].loglog(res,qquad_err,label='c-quad',linestyle='dashdot')
ax[0].set_ylabel(r'Error')
ax[1].set_ylabel(r'Time')
ax[0].legend()
ax[1].set_xlabel(r'$\hat{L}/\Delta \hat{\ell}$')
ax[1].loglog(res,gtrapz_time,label='gtrapz')
ax[1].loglog(res,quad_time,label='m-quad',linestyle='dashed')
ax[1].loglog(res,qquad_time,label='q-quad',linestyle='dashdot')
plt.savefig('error_square_well.eps')

plt.show()
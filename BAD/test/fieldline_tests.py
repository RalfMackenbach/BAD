from BAD.util_fieldline import *
from BAD.bounce_integrals import bounce_integral
import numpy as np
import matplotlib.pyplot as plt
# enable latex rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

z_res = 1000
lam_res = 100
lam_val = 0.7



############### Define relevant functions ####################
# define magnetic field
def B(z):
    return 1.0 - 0.5 * np.cos(z) #+ 0.1*np.sin(np.sqrt(2)*z)

# define intersection 1 - lam * B = 0
def intersection(lam):
    left_intersection = -np.arccos(2 * (lam - 1)/lam)
    right_intersection = np.arccos(2 * (lam - 1)/lam)
    return left_intersection, right_intersection

# calculate the bounce points for a given lambda
def bounce_points_num(lam,B_z,z):
    bounce_points, df_vals = bounce_points_f(1 - lam * B_z, z)
    return bounce_points, df_vals

# plot bounce points for a given lambda
lam_arr = np.linspace(1/1.5, 1/0.5, lam_res+2, endpoint=True)
lam_arr = lam_arr[1:-1]

# define the z and B_z arrays
z = np.linspace(-np.pi, np.pi, z_res)
B_z = B(z)
###################################################################################


################################# check bounce points #############################
fig, ax = plt.subplots(2, 1)

for lam in lam_arr:
    bounce_points, _ = bounce_points_num(lam,B_z,z)
    B_val = 1/lam
    # plot the left bounce point as a blue >
    left_bounce = bounce_points[:,0]
    ax[0].scatter(left_bounce, B_val * np.ones_like(left_bounce), c='b', marker='>')
    # plot the right bounce point as a red <
    right_bounce = bounce_points[:,1]
    ax[0].scatter(right_bounce, B_val * np.ones_like(right_bounce), c='r', marker='<')

ax[0].set_title('Bounce points, numerical')

# same plot as above but exact
for lam in lam_arr:
    left_intersection, right_intersection = intersection(lam)
    # plot the left bounce point as a blue >
    ax[1].scatter(left_intersection, 1/lam, c='b', marker='>')
    # plot the right bounce point as a red <
    ax[1].scatter(right_intersection, 1/lam, c='r', marker='<')

ax[1].set_title('Bounce points, exact')
plt.tight_layout()

plt.show()

# plot difference as a function of lambda
diff = []
for lam in lam_arr:
    bounce_points, _ = bounce_points_num(lam,B_z,z)
    left_intersection, right_intersection = intersection(lam)
    left_bounce = bounce_points[0,0]
    right_bounce = bounce_points[0,1]
    diff.append(np.abs(left_intersection - left_bounce) + np.abs(right_intersection - right_bounce))

plt.plot(lam_arr, diff)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$|z_{\rm num} - z_{\rm anal}|$')

plt.tight_layout()

plt.show()
###################################################################################

################################# check bounce points refined #####################
fig, ax = plt.subplots(2, 1)

for lam in lam_arr:
    bounce_points, _ = bounce_points_num(lam,B_z,z)
    for idx_bp, bp in np.ndenumerate(bounce_points):
        f_z = lambda z: 1.0 - lam*B(z)
        refined_bp = refine_roots(f_z,bp)
        bounce_points[idx_bp] = refined_bp
    B_val = 1/lam
    # plot the left bounce point as a blue >
    left_bounce = bounce_points[:,0]
    ax[0].scatter(left_bounce, B_val * np.ones_like(left_bounce), c='b', marker='>')
    # plot the right bounce point as a red <
    right_bounce = bounce_points[:,1]
    ax[0].scatter(right_bounce, B_val * np.ones_like(right_bounce), c='r', marker='<')

ax[0].set_title('Bounce points, numerical refined')

# same plot as above but exact
for lam in lam_arr:
    left_intersection, right_intersection = intersection(lam)
    # plot the left bounce point as a blue >
    ax[1].scatter(left_intersection, 1/lam, c='b', marker='>')
    # plot the right bounce point as a red <
    ax[1].scatter(right_intersection, 1/lam, c='r', marker='<')

ax[1].set_title('Bounce points, exact')
plt.tight_layout()

plt.show()

# plot difference as a function of lambda
diff = []
for lam in lam_arr:
    bounce_points, _ = bounce_points_num(lam,B_z,z)
    for idx_bp, bp in np.ndenumerate(bounce_points):
        f_z = lambda z: 1.0 - lam*B(z)
        refined_bp = refine_roots(f_z,bp)
        bounce_points[idx_bp] = refined_bp
    left_intersection, right_intersection = intersection(lam)
    left_bounce = bounce_points[0,0]
    right_bounce = bounce_points[0,1]
    diff.append(np.abs(left_intersection - left_bounce) + np.abs(right_intersection - right_bounce))

plt.plot(lam_arr, diff)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$|z_{\rm num} - z_{\rm anal}|$')

plt.tight_layout()

plt.show()
###################################################################################




################################# check boundary conditions #####################
# define the z and B_z arrays
z = np.linspace(-2*np.pi, 2*np.pi, 2*z_res)
B_z = B(z)

# plot the field
plt.plot(z, B_z)
plt.title('Field')
plt.show()


# construct the bounce_well
f = 1 - lam_val * B_z
bounce_points, df_vals  = bounce_points_num(lam_val,B_z,z)
# apply boundary conditions
bounce_points, df_vals = apply_boundary_condition(f, z, bounce_points, df_vals, bc='wall')
# loop over the bounce points
for b_idx in range(bounce_points.shape[0]):
    z_bounce = bounce_points[b_idx,:]
    z_pair = z_bounce
    h = np.ones_like(B_z)
    # construct the well
    z_well, f_well, h_well = construct_well_arr(f, h, z, z_pair)
    # loop over the well parts and plot
    color = ['b', 'r', 'g', 'm', 'c']
    for z_wp, f_wp, h_wp in zip(z_well, f_well, h_well):
        plt.scatter(z_wp, f_wp, c=color[b_idx], s=1)
        plt.scatter(z_wp, h_wp, c=color[b_idx], s=1)

plt.title('Wall boundary conditions')
plt.show()



# construct the bounce_well
f = 1 - lam_val * B_z
bounce_points, df_vals  = bounce_points_num(lam_val,B_z,z)
# apply boundary conditions
bounce_points, df_vals = apply_boundary_condition(f, z, bounce_points, df_vals, bc='periodic')
# loop over the bounce points
for b_idx in range(bounce_points.shape[0]):
    z_bounce = bounce_points[b_idx,:]
    z_pair = z_bounce
    h = np.ones_like(B_z)
    # construct the well
    z_well, f_well, h_well = construct_well_arr(f, h, z, z_pair)
    # loop over the well parts and plot
    color = ['b', 'r', 'g', 'm', 'c']
    for z_wp, f_wp, h_wp in zip(z_well, f_well, h_well):
        plt.scatter(z_wp, f_wp, c=color[b_idx], s=1)
        plt.scatter(z_wp, h_wp, c=color[b_idx], s=1)


plt.title('Periodic boundary conditions')
plt.show()



############## check construct well func #################
# construct the bounce_well
f = 1 - lam_val * B_z
bounce_points, df_vals  = bounce_points_num(lam_val,B_z,z)
# apply boundary conditions
bounce_points, df_vals = apply_boundary_condition(f, z, bounce_points, df_vals, bc='periodic')
# now construct the well for each pair
for b_idx in range(bounce_points.shape[0]):
    z_bounce = bounce_points[b_idx,:]
    z_pair = z_bounce
    z_range = construct_well_func(z, z_pair)
    print(z_range)

###################################################################################



################################# make well given z_bp #####################
z_bp = np.arccos(2.0 * (lam_val - 1.0)/lam_val)
z_pair_arr = make_well_given_bp(B_z, z, -z_bp, boundary='wall')
z_pair_fun = make_well_given_bp(B, z, -z_bp, boundary='wall')
print(z_pair_arr)
print(z_pair_fun)
print(np.abs(z_pair_arr - z_pair_fun))
###################################################################################
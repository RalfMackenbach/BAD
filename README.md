# BAD
## A repo for calculating the Bounce Averaged Drifts

### Main idea
This code contains three example cases of calculating the bounce-averaged drift, namely a s square well, a large-aspect ratio circular tokamak, and NCSX. These examples use different numerical methods of calculating the drift, and the differences in runtime and accuracy can be large. It furthermore uses different boundary conditions (periodic or quasi-periodic), the bounce-averaged drifts of particles which cross the boundary may change significantly. 


### Use
One can install the code by simply running 
```
pip install -e.
```
in the main directory.

The main workhorse of this code is the `bounce_integral_wrapper(f,h,x,is_func=False,return_roots=False)` function, in the `bounce_int` module (found in `BAD`). This function solves integrals of the form:
```
I=∫h(x)/sqrt(f(x)) dx
```
where the integration domain is set by simply connected regions of `f(x)>0`. For each simply connected region (typically referred to as a bounce-well) `I` is evaluated, and the answer is returned as a list where each element of the list corresponds to the value of `I` for one simply connected region. The function can be called in two ways. 


### Method 1: gtrapz
Given arrays `f_arr`, `h_arr`, and corresponding nodes `x_arr`, the integral can be evaluated using a generalisation of the trapezoidal rule. To do so, simply write
```
I = bounce_int.bounce_integral_wrapper(f_arr,h_arr,x_arr,is_func=False,return_roots=False)
```
This mode assumes that the functions f_arr and h_arr are well approximated by piece-wise linear interpolations.


### Method 2: quad
Given functions `f(x)` and `h(x)`, and an array `x_arr` integral can be evaluated with quadrature methods using
```
I = bounce_int.bounce_integral_wrapper(f,h,x_arr,is_func=False,return_roots=False)
```
The array `x_arr` is used for root-finding: the code looks for approximate locations of roots in the array `f_arr=f(x_arr)`, which are then refined using `brentq`. If there are roots on very small scales, one should take care to choose an appropriately well resolved `x_arr`. One can also set a `kwarg` for sinh-tanh quadrature methods (`sinhtanh=True` or `False`), though this tends to be finicky (an `AssertionError` often pops up). Finally, 


### Roots
If one wishes to return the locations of the roots of `f(x)` or `f_arr` as well, one can do so by setting
```
I, roots = bounce_int.bounce_integral_wrapper(f_arr,h_arr,x_arr,is_func=False,return_roots=True)
```
The roots are returned in an array, where each consecutive pair belongs to one region of `f(x)>0`.


### Some general remarks
Typically, `is_func=False` should be used in situations where speed is preferred over accuracy (such as optimisation loops or large database scans). Conversely `is_func=True` should be used in scenarios where accuracy is preferred. 

The code is written to deal with periodic boundary conditions for the functions `f(x)` and `h(x)`. Such a boundary condition may be exceptionally poor for devices with high shear, as `h(x)=v_D.nabla_alpha` has a linear term. To use a quasi-linear boundary condition, it is recommended to simply extrapolate the domain to the next B_max so that the boundary condition is unimportant. An example of this is given in the CHM folder.

If one wishes to calculate the bounce time, the binormal drift, and the radial drift in one go, it is best to construct a function that does so manually. This is because each call of `bounce_integral_wrapper(f,h,x)` calculates the roots again given the input. The roots don't change if one only varies `h(x)`, so one can best construct a new function which calculates the roots only once. Please construct a new function by consulting `bounce_integral_wrapper(f,h,x,is_func=False,return_roots=False)` to do so - it would only require minor changes.

One needs to take care in non-omnigenous systems, as there may be wells which have only one bounce point. There is no clear way to handle such errors (as the true drift can not be reconstructed). The code handles it in two ways: it either raises an error if it encounters an odd number of bounce-points, or you can set the kwarg `ignore_odd=True`, so that it instead returns an empty list. This kwarg is included in `bounce_integral_wrapper`, see the code for if one wishes to make their own implementation.

# BAD
## A repo for calculating the Bounce Averaged Drifts


### UPDATE
Major overhaul of the code, many routines have improved quadrature methods. Legacy function wrapper can be accessed via
```
from BAD import bounce_fieldline as bounce_int
bounce_int.bounce_integral_wrapper(...)
```
All old files are in `old_src`.


### Main idea
This code calculates the bounce-averaged drifts in a magnetic confinement device, tokamak or stellarator. The drifts are calculated using the bounce-averaging integral
```
I=∫h(x)/sqrt(f(x)) dx
```
where `f(x)` is the normalised parallel energy (`vpar^2 = 1 - \lambda B`), and `h(x)` is the function to be bounce-averaged. The integral is evaluated over the regions where `f(x)>0`, which are typically referred to as bounce-wells. The code can handle unstructured and structured data, and can handle both functions and arrays. The code is written in Python, and uses the `scipy` library for some numerical integration and root-finding.


### Installation
One can install the code by simply running 
```
pip install -e.
```
in the main directory.


### Usage
For bounce-averaging given a fieldline, most function of relevance are in the `bounce_fieldline` module. Two main functions are available:
- `bounce_int_lambda(B, h, z, lam, mode='fast', boundary_condition='periodic')`
- `bounce_int_zbp(B, h, z, zbp, mode='fast', boundary_condition='periodic')`

where `B` is the magnetic field, `h` is the function to be bounce-averaged, `z` is the coordinate along the fieldline. The two functions give different methods of parameterising trapped particles with `lam` or `zbp` describing the bounce points, where `lam` is the pitch angle (can correspond to multiple magnetic wells) and `zbp` is the coordinate of the bounce point (corresponding to a single magnetic well). The `mode` can be either `fast` or `accurate`, where `fast` uses a faster but less accurate method, and `accurate` uses a more accurate method but can be slower. The `boundary_condition` can be either `'periodic'`, `'wall'`, or `None`. This sets the behaviour of trapped particles beyond the computational domain.


### Future work
- Add functions for calculating the bounce-averaged drifts given a flux-surface.

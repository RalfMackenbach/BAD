import sys
sys.path.append('/Users/ralfmackenbach/Documents/GitHub/Bounce-averaged-drift/BAD/src/')  
import mag_reader
import bounce_int
import numpy as np
import time
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib        as mpl
import scipy 
import pandas as pd




# read the gist data and plot
data_ext = mag_reader.mag_data("s7_alpha5.txt")
data_trn = mag_reader.mag_data("s7_alpha5.txt")
data_ext.plot_geometry()

#  extend domain 
data_ext.extend_domain()
data_ext.plot_geometry()

# truncate domain
data_trn.truncate_domain()
data_trn.plot_geometry()
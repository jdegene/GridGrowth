# -*- coding: utf-8 -*-

"""
The following code can be run as is and populates the examples foldern with output images
"""

import numpy as np
import matplotlib.pyplot as plt
from gridgrowth import GridBuilder, GeoGridBuilder

# SET INITIAL PARAMETERS AND CREATE INPUT ARRAYS

dimensions = (100,100)

# create test array defining strength kernels with some seeding starting points
tarr = np.zeros(((100,100)), dtype="int") + 9999
tarr[0,9] = 2
tarr[57,95] = 3
tarr[99,99] = 2
plt.imsave("./examples/init_strength_array.png", tarr)

# create name array, "naming" the kernels with a distinct ID
name_arr = np.zeros(dimensions, dtype="int") + 9999
name_arr[0,9] = 10
name_arr[57,95] = 25
name_arr[99,99] = 50
plt.imsave("./examples/init_name_array.png", name_arr)


# RUN ACTUAL EXAMPLES

# ex1) easiest case - load kernel strength array only and run in full without any additional 
# Kernel names are inherited from input strength array = 
grid = GridBuilder(tarr)
grid.iterate_forward("full")

plt.imsave("./examples/ex1_output_strength_array.png", grid.t_ar)
plt.imsave("./examples/ex1_output_distance_array.png", grid.dist_ar)
plt.imsave("./examples/ex1_output_name_array.png", grid.t_names_ar)
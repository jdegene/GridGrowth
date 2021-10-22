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
tarr = np.zeros(((100,100)), dtype="int")
tarr[0,9] = 2
tarr[57,95] = 3
tarr[99,99] = 2
plt.imsave("./examples/init_strength_array.png", tarr)

# create name array, "naming" the kernels with a distinct ID
name_arr = np.zeros(dimensions, dtype="int")
name_arr[0,9] = 10
name_arr[57,95] = 25
name_arr[99,99] = 50
plt.imsave("./examples/init_name_array.png", name_arr)


# RUN ACTUAL EXAMPLES

# ex1) easiest case - load kernel strength array only and run in full without any additional input
# Kernel names are inherited from input strength array 
grid = GridBuilder(tarr)
grid.iterate_forward("full")

plt.imsave("./examples/ex1_output_strength_array.png", grid.t_ar)
plt.imsave("./examples/ex1_output_distance_array.png", grid.dist_ar)
plt.imsave("./examples/ex1_output_name_array.png", grid.t_names_ar)


# ex2) similar to 1 - load kernel strength array and different name array. Run in full without any additional input
grid = GridBuilder(tarr, t_names_ar=name_arr)
grid.iterate_forward("full")

plt.imsave("./examples/ex2_output_strength_array.png", grid.t_ar)
plt.imsave("./examples/ex2_output_distance_array.png", grid.dist_ar)
plt.imsave("./examples/ex2_output_name_array.png", grid.t_names_ar)


# ex3) similar to 2 - load kernel strength array and different name array. Run only (arbitrary) 17 epochs without any additional input
grid = GridBuilder(tarr, t_names_ar=name_arr)
grid.iterate_forward("epoch", by_amount=17)

plt.imsave("./examples/ex3_output_strength_array.png", grid.t_ar)
plt.imsave("./examples/ex3_output_distance_array.png", grid.dist_ar)
plt.imsave("./examples/ex3_output_name_array.png", grid.t_names_ar)


# ex4) similar to 2 - load kernel strength array and different name array. Run full. 
# Every kernel get radius of 25 cells no matter their strength.
grid = GridBuilder(tarr, t_names_ar=name_arr, buffer_kernels_by=25)
grid.iterate_forward("full")

plt.imsave("./examples/ex4_output_strength_array.png", grid.t_ar)
plt.imsave("./examples/ex4_output_distance_array.png", grid.dist_ar)
plt.imsave("./examples/ex4_output_name_array.png", grid.t_names_ar)


# ex5) similar to 4 - load kernel strength array and different name array. Run full. 
# Every kernel get radius of 25 cells no matter their strength. Max growth is 55 cells.
grid = GridBuilder(tarr, t_names_ar=name_arr, buffer_kernels_by=25, max_dist=55)
grid.iterate_forward("full")

plt.imsave("./examples/ex5_output_strength_array.png", grid.t_ar)
plt.imsave("./examples/ex5_output_distance_array.png", grid.dist_ar)
plt.imsave("./examples/ex5_output_name_array.png", grid.t_names_ar)


# ex6) similar to 2 - load kernel strength array and different name array. Run full. 
# Add a 3x3 weight grid that emphasizes/push from "north" direction.
push_grid = np.array([10,10,10,0,0,0,0,0,0]).reshape(3,3)
grid = GridBuilder(tarr, t_names_ar=name_arr, weight_ar=push_grid, weight_method = "add")
grid.iterate_forward("full")

plt.imsave("./examples/ex6_output_strength_array.png", grid.t_ar)
plt.imsave("./examples/ex6_output_distance_array.png", grid.dist_ar)
plt.imsave("./examples/ex6_output_name_array.png", grid.t_names_ar)

# ex7) similar to 2 - load kernel strength array and different name array. Run full. 
# Reduce initial kernel strength by distance using a linear correction.
grid = GridBuilder(tarr, t_names_ar=name_arr, falloff_type="linear", falloff_weight=0.01)
grid.iterate_forward("full")

plt.imsave("./examples/ex7_output_strength_array.png", grid.t_ar)
plt.imsave("./examples/ex7_output_distance_array.png", grid.dist_ar)
plt.imsave("./examples/ex7_output_name_array.png", grid.t_names_ar)

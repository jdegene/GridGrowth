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

# create cost array, where higher values makes it easer to traverse the cell (lower values = higher cost)
cost_arr = np.ones(dimensions, dtype="int")
cost_arr[25:50,18:99] = 10
cost_arr[90:91,11:98] = 20
plt.imsave("./examples/init_cost_array.png", cost_arr)

# create terrain array, where each cell has an encoded numerical value representing a terrain type
# only useful when used in combination with a transition rules dictionary as a 'rule book'
terrain_arr = np.ones(dimensions, dtype="int")
terrain_arr[25:50,18:99] = 855 # random ID, imagine this to be water
terrain_arr[24:25,17:19] = 1800 # random ID, imagine this to a harbour
terrain_arr[50:51,40:42] = 1800 # random ID, imagine this to a harbour
terrain_arr[90:91,11:98] = 120 # random ID, imagine this as a road
plt.imsave("./examples/init_terrain_array.png", terrain_arr)

# transition rules dictionary, depicting which transitions are allowed
# only allow transitions from water to a harbour. From harbour allow to water and land
trans_rules_dict = {'only_allowed': {855:[855,1800], 1800:[1,855,1800]}, 
					  'limit_into': [],
					  'never_allowed': {1:855}, # never allow from land to water directly
					  'takes_precedence': {}}

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

# ex8) similar to 2 - load kernel strength array and different name array. Run full. 
# Cost array redefines how fast kernels can expand
grid = GridBuilder(tarr, t_names_ar=name_arr, cost_ar=cost_arr, cost_method = "add")
grid.iterate_forward("full")

plt.imsave("./examples/ex8_output_strength_array.png", grid.t_ar)
plt.imsave("./examples/ex8_output_distance_array.png", grid.dist_ar)
plt.imsave("./examples/ex8_output_name_array.png", grid.t_names_ar)

# ex9) similar to 8 - load kernel strength array and different name array. Run full. 
# A terrain array is given as well, limiting what transitions are allowed between terrain types
grid = GridBuilder(tarr, t_names_ar=name_arr, cost_ar=cost_arr, cost_method = "add", terrain_ar=terrain_arr, 
                 terrain_rules_dict = trans_rules_dict)
grid.iterate_forward("full")

plt.imsave("./examples/ex9_output_strength_array.png", grid.t_ar)
plt.imsave("./examples/ex9_output_distance_array.png", grid.dist_ar)
plt.imsave("./examples/ex9_output_name_array.png", grid.t_names_ar)
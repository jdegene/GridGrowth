# -*- coding: utf-8 -*-

import numpy as np
import math

def invert_array_input(in_ar, nan_value, max_value=None):
    """ Input arrays :in_ar: higher numbers mean more velocity/pressure 
        To with propagation logic, order must be inversed.
        
        Uses largest value, adds +1, then subtracts this value from all present
        and takes the absolute value
        
        -> original max value = 99. Subtract value = 100. Max value becomes
            |99-100| = 1. 
        -> e.g. orignal value 42 will become |42-100|=58
        
        Args:
            :in_ar:     original input array
            :nan_value: nan value, will be ignored and remain as is
            :max_value: global max value to use. If None, use local max value found in array
        
        Returns:
            numpy array with inverted values
            
        Raises:
            ValueError: if input is not int
    """
    
    if not np.issubdtype(in_ar.dtype, np.integer):
        raise ValueError(f"Input must be of integer type but {in_ar.dtype} was given")
    
    out_ar = np.ma.masked_equal(in_ar, nan_value)
    
    if max_value is None:
        max_value = out_ar.max()
    
    out_ar = abs(out_ar - max_value) + 1
    out_ar = out_ar.filled(nan_value)    
    
    return out_ar
    
    
def determine_center_values(in_3x3_ar, step, nan_value):
    """ Determines which values in the 3x3 grid will set the center value.
        Only values that are not :nan_value: and have no remainder when 
            divided by :step: are eligible. 
        Center value is explicitely left included to to allow cases when a value
            can be altered again after being set.
        If two or more possible output values exist, choose random.
    
    Args:
        :in_3x3_ar: 3x3 numpy array, where the center value will be defined 
                    by "winning" surrounding value
        :step: iteration step
        :nan_value: nan value, will be ignored
    
    Returns:
        None if no value can/should be assigned
            or
        A list of tuple of coordinates in 3x3 of value which can "win".
            Values in list are in descending order, first value is strongest,
            last value weakest. 
    
    Raises:
    
        
    in_3x3_ar = np.array([[0,0,0],[1,2,3],[5,9,1]])
    """
        
    null_ar = np.where( (in_3x3_ar == nan_value), np.nan, in_3x3_ar)
    
    # set values to nan if remainder exist when dividing by step
    null_ar = np.where(np.mod(step, null_ar, where=~np.isnan(null_ar)) == 0, 
                       null_ar, 
                       np.nan)
    
    # if all values are nan, return None
    if np.all(np.isnan(null_ar)):
        return
    
    coord_list= []
    
    # fetch min values, set min values to nan until all of array is nan
    while not np.all(np.isnan(null_ar)):
        
        min_value = np.nanmin(null_ar)
        
        # returns 2 arrays, each array represents one axis
        coordinates = np.where(null_ar == min_value)
        
        # append all coordinates for min value to coords list. This affects cases
        # where two cells have identical values
        for coord in range(len(coordinates[0])):
            out_coord = (coordinates[0][coord], coordinates[1][coord])
            coord_list.append(out_coord)
            null_ar[out_coord] = np.nan
    
    return coord_list
    

def create_3x3_weights(direction_list=[], weights=1, neutral_value=0):
    """ Creates a 3x3 matrix of the direction to push into. 
        
        Imagine this as giving a wind direction. If wind is blowing North, upper
            cells should populated with a higher chance from the South. 
            -> Direction gives the wind direction BUT the weight is increased on the 
            opposite side
            
        Direction is list of numbers of winddirection with range (0-8)
        0 1 2
        3 4 5
        6 7 8
        
        Args:
            :direction_list: integer or list of min. 1 integer value to describe wind direction                            
            :weights:        integer or list of integers to set relative strength
                                if list, must be the same len as :direction_list:
            :neutral_value:  value non affected cells will be assigned to. Probably 0 if
                             weights are "added" and 1 if weights are "multiplied"
        
        Returns:
            2D ndarray with weights, boosting the side the "wind" is coming from
        
        Raises:
            ValueError if direction_list is empty, of wrong type or if weights don't match direction_list
    """
    
    if type(direction_list) is not list:
        direction_list = list(direction_list)
    
    if len(direction_list) == 0:
        raise ValueError("Direction list cannot be empty")
    elif len(direction_list) > 9:
        raise ValueError(f"Direction list can only have a maximum of 9 values but {len(direction_list)} were given")
    
    if any([i not in range(9) for i in direction_list]):
        raise ValueError("direction_list can only contain integer values 0 - 8")
    
    if type(weights) is list and len(weights) != len(direction_list):
        raise ValueError("weights given as list, but length does not match direction_list")
        
    # dictionary to map input direction to opposite side index in 3x3 grid
    loc_swap_dict = {0:8,1:7,2:6,3:5,4:4,5:3,6:2,7:1,8:0}
    
    # init array with neutral value
    ar = np.zeros(9, dtype=int) + neutral_value
    
    # set weights depening of input weights were list or int
    if type(weights) is list:
        for i, loc in enumerate(direction_list):
            ar[loc_swap_dict[loc]] = ar[loc_swap_dict[loc]] + weights[i]
    
    else:
        for loc in direction_list:
            ar[loc_swap_dict[loc]] = ar[loc_swap_dict[loc]] + weights
    
    # reshape to 2D and return
    ar2D = ar.reshape((3,3))
    
    return ar2D
    

def get_possible_max_value(in_ar, nan_val, in_cost_ar=None, in_weight_ar=None, 
                           cost_method="add", weight_method="add"):
    """ Calculate the maximum value possbile when all weights are combined.
        Necessary to correctly scale all values for steps
        
        Args:
            :in_ar:         initial array with initial central kernels
            :in_cost_ar:    cost array
            :in_weight_ar:  weights array
            :cost_method:   method used to apply costs
            :weight_method: method used to apply weights
        
        Returns:
            Integer maximum possible value
        
        Raises:
            ValueError if cost_method or weight_method are not "add" or "multiply"
    """
    in_ar = in_ar.astype(float).copy()
    in_ar[in_ar==nan_val] = np.nan
    
    max_in = np.nanmax(in_ar)
    max_cost = 0 if in_cost_ar is None else np.nanmax(in_cost_ar)
    max_weight = 0 if in_weight_ar is None else np.nanmax(in_weight_ar)
    
    if cost_method == "add":
        if weight_method == "add":
            max_val = max_in + max_cost + max_weight
        elif weight_method == "mutliply":
            max_val = max_in + max_cost * max_weight
        else:
            raise ValueError(f"weight_method must be add or multiply but was {weight_method}")
    
    elif cost_method == "multiply":
        if weight_method == "add":
            max_val = max_in * max_cost + max_weight
        elif weight_method == "mutliply":
            max_val = max_in * max_cost * max_weight
        else:
            raise ValueError(f"weight_method must be add or multiply but was {weight_method}")
    
    else:
        raise ValueError(f"cost_method must be add or multiply but was {cost_method}")
    
    return int(max_val)


def build_transition_rules_full(trans_rules_dict=None, terrain_ar=None):
    """ 
    Takes handwritten small dictionary of rules as input and generates complete dictionary of
    all possible transitions and their validity
    
    Args:
        :trans_rules_dict: small dictionary of 'only_allowed', 'never_allowed', 'takes_precedence'
        :terrain_ar:       terrain_array with all valid terrain values
    
    Returns:
        Long dictionary with all possible terrain transitions and a flag 0,1,2 for each
        None if input was None
    
    Raises:
        ValueError when mutually exclusive rules exist in :trans_rules_dict:    
    
    terrain_ar = np.random.randint(4, size=dimensions)
    trans_rules_dict = {'only_allowed': {2: 3, 1:0},
                     'limit_into': [0],
                     'never_allowed': {3: [2, 1]},
                     'takes_precedence': {0: 1}}
    """
    
    # return None if either no dictionary or no terrain_ar was given
    if trans_rules_dict is None or terrain_ar is None:
        return
    
    # if keys are missing, add empty keys
    for key in ['never_allowed', 'only_allowed', 'takes_precedence', 'limit_into']:
        if key not in trans_rules_dict.keys():
            trans_rules_dict[key] = {}
    
    # test if 'never_allowed' rules clashes with 'only_allowed' or 'takes_precedence'
    for na in trans_rules_dict['never_allowed']:
        if na in trans_rules_dict['only_allowed'] or na in trans_rules_dict['takes_precedence']:
            raise ValueError("Rules are clashing. Cannot have same terrain rule as 'never_allowed'"                              
                             "and 'only_allowed' or 'takes_precedence'" )
    
    out_dict = {}
    
    for from_terrain in np.unique(terrain_ar):
        out_dict[from_terrain] = {}
        
        # if value in only_allowed, set all others from terrain to 0 and the one to 1
        if from_terrain in trans_rules_dict['only_allowed']:   
            for to_terrain in np.unique(terrain_ar):
                out_dict[from_terrain][to_terrain] = 0
            
            set_list = trans_rules_dict['only_allowed'][from_terrain]
            set_list = [ set_list ] if type(set_list) is not list else set_list
            for to_terrain in set_list: # trans_rules_dict['only_allowed'][from_terrain]
                out_dict[from_terrain][to_terrain] = 1
        
        # else set all values to 1 and then overwrite 'never_allowed' and 'takes_precedence'
        else:
            for to_terrain in np.unique(terrain_ar):
                if to_terrain in trans_rules_dict['limit_into']:
                    out_dict[from_terrain][to_terrain] = 0
                else:
                    out_dict[from_terrain][to_terrain] = 1
                        
            # check if never_allowed exist. Can be single value or list of values. If single value
            # convert to list
            if from_terrain in trans_rules_dict['never_allowed'].keys():
                set_list = trans_rules_dict['never_allowed'][from_terrain]
                set_list = [ set_list ] if type(set_list) is not list else set_list
                for to_terrain in set_list:
                    out_dict[from_terrain][to_terrain] = 0
            
            # check if "takes_precedence" exist and overwrite
            if from_terrain in trans_rules_dict['takes_precedence'].keys():
                set_list = trans_rules_dict['takes_precedence'][from_terrain]
                set_list = [ set_list ] if type(set_list) is not list else set_list
                for to_terrain in set_list:
                    out_dict[from_terrain][to_terrain] = 2
        
    return out_dict

def get_3x3_array(in_array, in_coords, nan_value):
    """ Get smaller 3x3 array from larger array. Takes care of edge cases (literally)
        where normal slicing would not yield a 3x3 array when center coord is on the edge
    
    Args:
        :in_array:  array to slice 3x3 array from
        :in_coords: coords from large array
        :nan_value: used to pad small array on the edges
    
    Returns:
        3x3 ndArray
    
    Raises:
        
    """
    
    # if not an edge case, simply slice and return
    if ( (in_coords[0] > 0 and in_coords[0] < in_array.shape[0]) and 
         (in_coords[1] > 0 and in_coords[1] < in_array.shape[1])   ):
        
        out_arr = in_array[in_coords[0]-1:in_coords[0]+2, 
                           in_coords[1]-1:in_coords[1]+2]
        
        return out_arr
    
    # else get what is available from large array then pad according to location
    else:
        out_arr = in_array[max(0, in_coords[0]-1) : min(in_array.shape[0], in_coords[0]+2), 
                           max(0, in_coords[1]-1) : min(in_array.shape[0], in_coords[1]+2)]
        
        # top row but not corners
        if in_coords[0] == 0 and in_coords[1] not in (0, in_array.shape[1]-1):
            out_arr = np.pad(out_arr, pad_width=[(1,0),(0,0)], mode='constant', constant_values=nan_value)
        # right side but not corners
        elif in_coords[0] not in (0, in_array.shape[0]-1) and in_coords[1] == in_array.shape[1]-1:
            out_arr = np.pad(out_arr, pad_width=[(0,0),(0,1)], mode='constant', constant_values=nan_value)    
        # lower side but not corners
        elif in_coords[0] == in_array.shape[0]-1 and in_coords[1] not in (0, in_array.shape[1]-1):
            out_arr = np.pad(out_arr, pad_width=[(0,1),(0,0)], mode='constant', constant_values=nan_value)  
        # left side but not corners
        elif in_coords[0] not in (0, in_array.shape[0]-1) and in_coords[1] == 0:
            out_arr = np.pad(out_arr, pad_width=[(0,0),(1,0)], mode='constant', constant_values=nan_value) 
        
        # top left corner, pad left and top
        elif in_coords[0] == 0 and in_coords[1] == 0:
            out_arr = np.pad(out_arr, pad_width=[(1,0),(1,0)], mode='constant', constant_values=nan_value)
        # top right corner, pad right and top
        elif in_coords[0] == 0 and in_coords[1] == in_array.shape[1]-1:
            out_arr = np.pad(out_arr, pad_width=[(1,0),(0,1)], mode='constant', constant_values=nan_value)
        # lower left corner, pad left and below
        elif in_coords[0] == in_array.shape[0]-1 and in_coords[1] == 0:
            out_arr = np.pad(out_arr, pad_width=[(0,1),(1,0)], mode='constant', constant_values=nan_value)
        # lower right corner, pad right and below
        elif in_coords[0] == in_array.shape[0]-1 and in_coords[1] == in_array.shape[1]-1:
            out_arr = np.pad(out_arr, pad_width=[(0,1),(0,1)], mode='constant', constant_values=nan_value)
        
        return out_arr

def transition_rules_outcome(full_rules_dict=None, 
                             strength_3x3_ar=None,
                             terrain_ar=None, 
                             pos_coords_list=[], 
                             glob_center_coord=None):
    """ 
    Checks if transition in full_rules_dict is 0 (not allowed), 1 (allowed) or 2 (give precedence)
    0 transitions are never returned
    1 transitions only if no 2 exists
    2 transitions always take precedence
    If several 1s or severl 2s exist, the one with greater strength is chosen (or simply first
        one if strength is identical). Greater strength equals lower number here.
    
    Args:
        :full_rules_dict:       dictionary with ALL possible transition rules, 
                                    output of build_transition_rules_full()
        :pos_coords_list:       list of coords from 3x3 grid to check
        :glob_center_coord:     global center coord to be set by values from pos_coords_list
        :terrain_ar:            full terrain array
    
    Returns:
        Tuple of 3x3 coordinates to set center value with
        None if no valid value could be found    
    
    Raises:
        
    strength_3x3_ar = np.array( [[ 2,0,0 ],[0,0,0],[0,7,0]] )
    """
    
    if full_rules_dict is None or terrain_ar is None:
        return pos_coords_list[0]
    
    # get small terrain
    #terrain_3x3_ar = terrain_ar[glob_center_coord[0]-1:glob_center_coord[0]+2, 
    #                            glob_center_coord[1]-1:glob_center_coord[1]+2]
    
    terrain_3x3_ar = get_3x3_array(terrain_ar, glob_center_coord, 9999)
    
    center_terrain_code = terrain_ar[glob_center_coord]
    
    # get precedence list, return coords if exact 1 pair exists, or determine which to choose if >1
    # if no precedence, repeat with normal allowed cells
    precedence_list = []
    precedence_weights_list = []
    allowed_list = []
    allowed_weights_list = []
    
    for coords in pos_coords_list:
        from_terrain_code = terrain_3x3_ar[coords[0],coords[1]]
        if full_rules_dict[from_terrain_code][center_terrain_code] == 2:
            precedence_list.append(coords)
            precedence_weights_list.append( strength_3x3_ar[coords] )
        elif full_rules_dict[from_terrain_code][center_terrain_code] == 1:
            allowed_list.append(coords)
            allowed_weights_list.append( strength_3x3_ar[coords] )
            
    if len(precedence_list) == 1:
        return precedence_list[0]
    elif len(precedence_list) > 1:
        min_val_index = precedence_weights_list.index(min(precedence_weights_list))
        return precedence_list[min_val_index]
    elif len(allowed_list) == 1:
        return allowed_list[0]
    elif len(allowed_list) > 1:
        min_val_index = allowed_weights_list.index(min(allowed_weights_list))
        return allowed_list[min_val_index]
    else:
        return
        

def local_coord_to_global(in_coord, center_coord):
    """ Converts a coordinate from a 3x3 grid into coordinate from large grid 
    
    Args:
        :in_coord:      tuple of local coordinates to convert to global
        :center:coord:  center coordinate of 3x3 grid cell in global coordinate system
    
    Returns:
        Tuple of coordinates in global system
    
    Raises:
        
    """
    
    new_coord_0 = center_coord[0]  + in_coord[0]-1
    new_coord_1 = center_coord[1]  + in_coord[1]-1
    
    return (new_coord_0, new_coord_1)

def falloff(func_type, dist, value, func_weight=None, correct_by=1):
    """ If called, will reduce an input :val: depending on its distance from original kernel
    
    Args:
        :func_type: function name by which to reduce input value.
                    Can be linear, exp, log
        :func_weight: optional value if func_type accepts one 
                     (eg 2 for linear if every 1 step in distance reduces value by 2)
        :value: original value
        :dist: int distance from original kernel
    
    Reutrns:
        int value after falloff. Minium value returned is 0
    
    Raises:
        InputError if unknown func_type was called
    """
    
    if func_type == "linear":
        if func_weight is None:
            func_weight = 1
        ret_val = value - dist * (func_weight/correct_by)
    
    elif func_type == "exp":
        if func_weight is None:
            func_weight = 2
        ret_val = value - dist ** (func_weight/correct_by)
    
    elif func_type == "log":
        if func_weight is None:
            func_weight = math.e
        ret_val = value - math.log(dist, (func_weight/correct_by))
    
    else:
        raise ValueError(f"Falloff function must be linear, exp or log but was {func_type}")
    
    return max(0, int(ret_val) )
    

def buffer_kernels(in_ar, org_ar, nan_value, in_name_ar, in_dist_ar, by=None,
                   falloff_type=None, falloff_weight=None, correct_by=1):
    """ Add a buffer around initial kernels to force immediate vicinty to belong to kernels
    
    Args:
        :by:            number of pixels to buffer in each directions
        :in_ar:         input array to write buffers to
        :org_ar:        array that tracks original kernel strength
        :nan_value:     kernels are all values that are not nan_value
        :in_name_ar:    input name array, names of initial kernel will be updated 
        :in_dist_ar:    distance from initial kernel will be updated
        :correct_by:    value to correct falloff_weight by. Is the same as gcd if value optimization is enabled
                        or 1 by default (= no correction). This is necessary to scale falloff_weight along optimized values
    
    Return:
        Updated in_ar, in_name_ar and in_dist_ar
    
    Raises:
        
    
    by = 20
    dims = (100,100)
    in_ar = np.zeros(dimensions, dtype="int")
    in_ar[0,9] = 5
    in_ar[7,3] = 10
    in_ar[99,99] = 25
    org_ar = in_ar.copy()
    in_dist_ar = np.zeros(dims, dtype="int")
    in_name_ar = in_ar.copy()
    falloff_type="linear"
    
    import matplotlib.pyplot as plt
    plt.imshow(t_ras.distance_ar, interpolation=None)
    plt.show()
    """
    
    if by is None:
        return in_ar, org_ar, in_name_ar, in_dist_ar
    
    # set values to copy or process will perpetuate itself
    out_ar = in_ar.copy() 

    with np.nditer(in_ar, flags=['multi_index']) as it:
        for cell in it:                       
            if cell != nan_value:
                coords = it.multi_index
                
                kernel_strength = in_ar[coords]
                name = in_name_ar[coords]
                
                # define coordinates around center to check for distance to kernel
                # account for edges
                min_x = max(0, coords[0]-by)
                max_x = min(in_ar.shape[0]-1, coords[0]+by)
                min_y = max(0, coords[1]-by)
                max_y = min(in_ar.shape[1]-1, coords[1]+by)
                
                for x in range(min_x, max_x+1):
                    for y in range(min_y, max_y+1):
                        
                        dist = int(math.sqrt( (coords[0]-x)**2 + (coords [1]-y)**2  ))
                        
                        # only set if distance within set limit
                        if dist <= by:
                            #only set values if no other value has been set in place
                            # OR if value already present, set value with smaller dist
                            if org_ar[x,y] == nan_value or dist < in_dist_ar[x,y]:
                                if falloff_type is None:
                                    out_ar[x,y] = kernel_strength
                                else:
                                    out_ar[x,y] = falloff(falloff_type, 
                                                          dist,
                                                          kernel_strength,
                                                          func_weight=falloff_weight,
                                                          correct_by=correct_by)
                                org_ar[x,y] = kernel_strength
                                in_name_ar[x,y] = name
                                in_dist_ar[x,y] = dist
    
    return out_ar, org_ar, in_name_ar, in_dist_ar
                            

                                


class GridBuilder():
    """ Main class to handle initiation of the grid and stepping through the epochs """    
    
    def __init__(self, t_ar, t_names_ar=None, cost_ar=None, terrain_ar=None, weight_ar=None,
                 terrain_rules_dict = None, nan_value=None, weight_method = "add", cost_method = "add", 
                 buffer_kernels_by=None, falloff_type=None, falloff_weight=None, 
                 optimize_input=False):
        
        # traverse array that gives the initial seeds. Should be of type non-negative int
        self.t_ar = t_ar        
        self.shape = t_ar.shape
        
        # keep an array in parallel that retains the original t_ar value for all cells
        # comes in handy when a falloff function was used an neighbour cells "lose" the information
        # of original strength
        self.org_t_ar = t_ar.copy()
        
        # name array with an ID in place of seeds from t_ar. Where in t_ar many seeds
        # can have the same value, this name array can be used to track which was original seeding point
        # If None, use t_ar values as names
        if t_names_ar is None:
            self.t_names_ar = t_ar.copy()
        else:
            self.t_names_ar = t_names_ar
        
        # a cost array is a numerical value of "how fast" a cell can be traversed.
        # Higher values means faster. Can be None = no terrain influence
        # Must have same shape as t_ar
        if cost_ar is not None:
            self.cost_ar = cost_ar
        else:
            self.cost_ar = np.ones(self.shape, dtype='int')
            #self.cost_method = "multiply" 
        
        # a terrain array can be used to disallow transitions between certain cell types
        # Transition rules come in terrain_rules_dict. Can be None = all transitions allowed
        # Must have same shape as t_ar
        self.terrain_ar = terrain_ar
        
        self.terrain_rules_dict = terrain_rules_dict
        if self.terrain_rules_dict is not None:
            self.full_transition_rules_dict = build_transition_rules_full(trans_rules_dict=self.terrain_rules_dict, 
                                                             terrain_ar=self.terrain_ar)
        else:
            self.full_transition_rules_dict = None
        
        # 3x3 weight array to simulate a push factor from one side, imagine a wind effect
        # can be None
        self.weight_ar = weight_ar

        # distance array that will keep track how far a pixel is away from original seed
        self.dist_ar = np.zeros(self.shape, dtype="int")
        
        # if nan_value was not given, use value which occurs most in cost_array
        if nan_value is None:
            self.nan_value = np.bincount(self.t_ar.reshape(-1)).argmax()
        else:
            self.nan_value = nan_value
        
        # optimize t_ar by reducing strength values so their relative difference remains the same
        # but their absolute value is reduced. This reduces computation time
        gcd = 1
        self.optimize_input = optimize_input
        if self.optimize_input == True:
            uq_values_ar = np.unique(self.t_ar)
            self.org_values_ar = np.delete(uq_values_ar, np.where(uq_values_ar== self.nan_value))
            gcd = np.gcd.reduce(self.org_values_ar) # greatest common divisor to divide original values by
            self.opt_values_ar = (self.org_values_ar / gcd).astype(int)
        
            for i, replace_val in enumerate(self.org_values_ar) :
                self.t_ar = np.where(self.t_ar == replace_val, self.opt_values_ar[i], self.t_ar)
        
        self.weight_method = weight_method
        self.cost_method = cost_method 
        self.buffer_kernels_by = buffer_kernels_by
        
        self.falloff_type = falloff_type
        self.falloff_weight = falloff_weight 
        
        self.step = 1 # steps are rolled back to 1 when max value was reached
        self.iterations = 0 # total iterations, incremented each step
        self.epoch = 0 # is incremented each time step is rolled over
        
        self.total_max_value = get_possible_max_value(self.t_ar, 
                                                      self.nan_value,
                                                      self.cost_ar, 
                                                      self.weight_ar, 
                                                      cost_method = self.cost_method, 
                                                      weight_method = self.weight_method)
    
        # set a buffer around kernels they get regardless of their strength
        if self.buffer_kernels_by:
            self.t_ar, self.org_t_ar, self.t_names_ar, self.dist_ar = buffer_kernels(self.t_ar, 
                                                                                        self.org_t_ar, 
                                                                                        self.nan_value, 
                                                                                        self.t_names_ar, 
                                                                                        self.dist_ar, 
                                                                                        by=self.buffer_kernels_by,
                                                                                        falloff_type=self.falloff_type,
                                                                                        falloff_weight=self.falloff_weight,
                                                                                        correct_by = gcd)
        
    def iterate_forward(self, how='full', by_amount=1):
        """ Progresses the raster by x steps 
            
        Args:
            :how:        'full'  runs until finished (ignores by_amount)
                         'step'  runs :by_amount: steps
                         'epoch' runs :by_amount: epochs
            :by_amount:  value by how much :how: will be progressed
        
        Returns:
            Nothing
        
        Raises:
            ValueError if by_amount is not of type int
            
        """
        
        if type(by_amount) is not int:
            raise ValueError(f"by_amount must be of type int but was {type(by_amount)}")
        
        init_step = self.step
        init_epoch = self.epoch
        
        while True:

            # make copy of input array. If this is still the same array at the end of the loop
            # process is finished and loop can be left
            old_arr = self.t_ar.copy()
            
            # set values in set_arr, then at the end overwrite t_ar with set_arr
            # dont write to array you are iterating over
            set_arr = self.t_ar.copy()
            
            array_it = np.nditer(self.t_ar, flags=['multi_index'])
            for cell in array_it:
                center_coords = array_it.multi_index
                
                # skip cells already assigned to value
                if ((self.t_ar[center_coords] != self.nan_value) and (self.t_names_ar[center_coords] )
                    or
                    (self.cost_ar[center_coords] == 0)
                    ):
                    continue
                                    
                three_by_three_arr = get_3x3_array(self.t_ar, center_coords, self.nan_value)
                
                if len(np.unique(three_by_three_arr)) == 1:
                    continue
                
                three_by_three_marr = np.ma.masked_equal(three_by_three_arr, self.nan_value)
                
                # add the weight of the center value of the cost array
                if self.cost_ar is not None:
                    if self.cost_method == "add":
                        three_by_three_marr = three_by_three_marr + self.cost_ar[center_coords]
                    elif self.cost_method == "mulitply":
                        three_by_three_marr = three_by_three_marr * self.cost_ar[center_coords]
                
                # add weight array values if any were given
                if self.weight_ar is not None:
                    if self.weight_method == "add":
                        three_by_three_marr = three_by_three_marr + self.weight_ar
                    elif self.weight_method == "multiply":
                        three_by_three_marr = three_by_three_marr * self.weight_ar
                
                # invert small array so large values become small to adhere to step logic
                inv_three_by_three_ar = invert_array_input(three_by_three_marr, 
                                                           self.nan_value, 
                                                           max_value=self.total_max_value)
                
                # get a list of all cells that could set the center value
                pos_coords_list = determine_center_values(inv_three_by_three_ar, self.step, self.nan_value)
                if pos_coords_list is not None:
                    
                    # check which of possbile many eligible cells is allowed by terrain rules to move to center coord
                    inherit_from_coord = transition_rules_outcome(full_rules_dict=self.full_transition_rules_dict, 
                                             strength_3x3_ar=inv_three_by_three_ar,
                                             terrain_ar=self.terrain_ar, 
                                             pos_coords_list=pos_coords_list, 
                                             glob_center_coord=center_coords)
                    
                    if inherit_from_coord is None:
                        continue
                    
                    # convert local 3x3 coordinate into global system
                    inherit_from_coord_glob = local_coord_to_global( inherit_from_coord, center_coords)
                    
                    # set values
                    if self.falloff_type is None:
                        set_arr[center_coords] = self.t_ar[inherit_from_coord_glob]
                    else:
                        set_arr[center_coords] = falloff(self.falloff_type, 
                                                          self.dist_ar[inherit_from_coord_glob],
                                                          self.t_ar[inherit_from_coord_glob],
                                                          func_weight=self.falloff_weight)
                    
                    self.dist_ar[center_coords] = self.dist_ar[inherit_from_coord_glob]+1
                    self.org_t_ar[center_coords] = self.org_t_ar[inherit_from_coord_glob]
                    self.t_names_ar[center_coords] = self.t_names_ar[inherit_from_coord_glob]
                    
                else:
                    continue
            
            self.t_ar = set_arr.copy()
            
            # increase step, reduce step to 1 when max possible value has been reached
            self.step = self.step+1 if self.step <= self.total_max_value else 1
            self.iterations += 1
            if self.step == 1:
                self.epoch += 1
            
            if how == 'full':
                if np.array_equal(old_arr, self.t_ar) and self.step == self.total_max_value:
                    break
            elif how == 'step':
                if (self.step - init_step) == by_amount:
                    break
            elif how == 'epoch':
                if (self.epoch - init_epoch) == by_amount:
                    break            
            else:
                raise ValueError(f"'how' must be full, step or epoch but {how} was given")
            
            if self.step % 10 == 0:
                print(f"\rStep: {self.step}, Iteration: {self.iterations}, Epoch: {self.epoch}", end='')
                

class GeoGridBuilder(GridBuilder):
    """ Inherits from main class but can handle initiation and output of georeferenced rasters. 
        All inputs are expected to be GeoTiffs if used (= no mixing between arrays & geotiffs)
        OSGEO / GDAL libraries must be installed to use this.
    """      
    
    def __init__(self, t_ar_tiff_fp, t_names_tiff_fp=None, cost_tiff_fp=None, terrain_tiff_fp=None, weight_tiff_fp=None,
                 terrain_rules_dict=None, nan_value=None, weight_method="add", cost_method="add", 
                 buffer_kernels_by=None, falloff_type=None, falloff_weight=None):
        
        # load geodata handling functions. Put in try/except to only load once even if method is called multiple times
        try: 
            geotiff_to_array
        except NameError:
            from geo_data_handler import geotiff_to_array
        
        # init paths
        self.t_ar_tiff_fp = t_ar_tiff_fp
        self.t_names_tiff_fp = t_names_tiff_fp 
        self.cost_tiff_fp = cost_tiff_fp
        self.terrain_tiff_fp = terrain_tiff_fp
        self.weight_tiff_fp = weight_tiff_fp
        
        # convert geotiffs to arrays
        t_ar = geotiff_to_array(t_ar_tiff_fp)
        t_names_ar = geotiff_to_array(t_names_tiff_fp) if t_names_tiff_fp is not None else None
        cost_ar = geotiff_to_array(cost_tiff_fp) if cost_tiff_fp is not None else None
        terrain_ar = geotiff_to_array(terrain_tiff_fp) if terrain_tiff_fp is not None else None
        weight_ar = geotiff_to_array(weight_tiff_fp) if weight_tiff_fp is not None else None
        
        # pass arguments to base class
        super().__init__(t_ar, t_names_ar=t_names_ar, cost_ar=cost_ar, terrain_ar=terrain_ar, weight_ar=weight_ar,
                 terrain_rules_dict=terrain_rules_dict, nan_value=nan_value, weight_method=weight_method, cost_method=cost_method, 
                 buffer_kernels_by=buffer_kernels_by, falloff_type=falloff_type, falloff_weight=falloff_weight)
    
    def save_to_geotiff(self, save_fol, file_name=None, save_strength_file=True):
        """ Save the current state as a geotiff 
        
        Args:
            :save_fol:          folder to save output to
            :file_name:         specify a filename (without file type ending like .tif)
                                can be None -> output will be simply named "main.tif"
            :save_strength_file: True: tries to save the output of the strength file if an input path was given
                                    uses file_name and appends _strength_file to it
        
        Returns:
            Nothing
        
        Raises:            
        """
        
        try: 
            array_to_raster
        except NameError:
            from geo_data_handler import array_to_raster

        
        if file_name is None:
            file_name = "main"
            strength_file_name = "main_strength_file"
        else:
            # if filename was given with file ending, replace by single file endin
            file_name = file_name.replace(".tif", "")
            strength_file_name  = file_name + "_strength_file"
        
        main_outpath = save_fol + file_name + ".tif"
        strength_file_outpath = save_fol + strength_file_name + ".tif"
           
        array_to_raster(self.t_ar_tiff_fp, self.t_names_ar, main_outpath)
        
        if save_strength_file is True:
            strength_file_outpath = save_fol + strength_file_name + ".tif"
            array_to_raster(self.t_ar_tiff_fp, self.t_ar, strength_file_outpath )
 

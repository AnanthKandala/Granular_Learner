import numpy as np
from helper_funcs import Wrapper

def Total_energy_func(inner_positions, boundary_positions, cut_offs, boundary_cutoffs, Ly, epsilon, alpha):
    '''Calculates the total potential energy of the system
    inner_positions_cyl [1D np.array]: positions of the inner spheres. positions must be between [0, Lx] x [0, Ly]
    boundary_positions [np.array]: positions of the boundary spheres (fixed)
    cut_offs [np.array]: cut_offs[i] is the cut_off distance between the ith inner sphere combination
    boundary_cutoffs [np.array]: boundary_cutoffs[i] is the cut_off distance between the ith inner and outer sphere combination
    epsilon [float]: energy scale
    alpha [float]: exponent in the potential energy function
    Ly [float]: length of the periodic direction'''

    def Energy_func(r_diff, cut_offs, epsilon, alpha):
            '''Calculates the energy of the system'''
            return (epsilon/alpha)*np.sum((1 - r_diff / cut_offs) ** alpha)

    inner_positions = inner_positions.reshape(-1, 2) #reshape the inner positions
    inner_positions[:, 1] = Wrapper(inner_positions[:, 1], Ly) #periodic boundary conditions in the y direction
    #calculate the bulk energy:
    uti = np.triu_indices(len(inner_positions), k=1) #obtain unique indices of the inner sphere combinations
    r_diff = inner_positions[uti[0]] - inner_positions[uti[1]] #calculate the difference in positions
    r_diff[:, 1] = r_diff[:, 1] + Ly*(r_diff[:, 1] < -Ly/2) - Ly*(r_diff[:, 1] > Ly/2) #periodic boundary conditions in the y direction
    r_diff = np.linalg.norm(r_diff, axis=1) #calculate the distance between the inner spheres
    bulk_energy = Shortrange_energy(r_diff, cut_offs, Energy_func, epsilon, alpha) #calculate the bulk energy

    #calculate the boundary energy:
    r_diff = inner_positions[:, np.newaxis, :] - boundary_positions[np.newaxis, :, :] #calculate the difference in positions
    r_diff[:, :, 1] = r_diff[:, :, 1] + Ly*(r_diff[:, :, 1] < -Ly/2) - Ly*(r_diff[:, :, 1] > Ly/2) #periodic boundary conditions in the y direction
    r_diff = np.linalg.norm(r_diff, axis=2) #calculate the distance between the inner and outer spheres
    boundary_energy = Shortrange_energy(r_diff, boundary_cutoffs, Energy_func, epsilon, alpha) #calculate the boundary energy
    total_energy = bulk_energy + boundary_energy #calculate the total energy
    print(f'boundary_energy = {boundary_energy}, bulk_energy = {bulk_energy}, total_energy = {total_energy}')
    return total_energy


def Shortrange_energy(r_diff, cut_offs, energy_func, epsilon, alpha):
    neighbors = np.where(r_diff < cut_offs) #find the neighbors
    energy = energy_func(r_diff[neighbors], cut_offs[neighbors], epsilon, alpha) #calculate the energy
    return energy

    

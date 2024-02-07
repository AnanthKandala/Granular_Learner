import numpy as np
from helper_funcs import Wrapper

def  Inner_forces(inner_positions, Ly, cut_offs, return_mags=False, epsilon = 1, alpha = 2):
    '''Calculates the forces between the inner spheres
    args:
        inner_positions [Nx2 np.array]: positions of the inner spheres
        Ly [float]: length of the periodic direction
        cut_offs [list/1d np.array]: cut_offs[i] is the cut_off distance between the ith inner sphere combination
        epsilon [float]: energy scale
        alpha [float]: exponent in the potential energy function
        return_mags [bool]: if True, returns the force magnitudes instead of the force vectors, (option used for visualization)
    returns:
        forces [N'x 4 np.array]: force[k] = [i, j, fx, fy] where [fx,fy] is force on the ith inner sphere due to the jth inner sphere and N' is the number of neighbors. If return_mags is True, force[k] = [i, j, f] where f is the force magnitude
    '''
    inner_positions = inner_positions.reshape(-1, 2) #reshape the inner positions
    uti = np.triu_indices(len(inner_positions), k=1) #obtain unique indices of the inner sphere combinations
    r_diff_vec = inner_positions[uti[0]] - inner_positions[uti[1]] #calculate the difference in positions
    r_diff_vec[:,1] = r_diff_vec[:,1] - Ly*np.round(r_diff_vec[:,1]/Ly) #periodic boundary conditions in the y direction
    r_diff = np.linalg.norm(r_diff_vec, axis = 1)
    neighbors = np.where(r_diff < cut_offs)
    r_diff = r_diff[neighbors]; r_diff_vec = r_diff_vec[neighbors]; cut_offs = cut_offs[neighbors] #selecting only the neighbors
    force_magnitudes = (epsilon/cut_offs)*(1 - r_diff/cut_offs)**(alpha-1)
    if return_mags:
        return_force_mags = np.column_stack((uti[0][neighbors], uti[1][neighbors], force_magnitudes))
        return return_force_mags, r_diff_vec
    else:
        force_directions = r_diff_vec/r_diff[:, None]
        forces = force_magnitudes[:, None]*force_directions
        return np.column_stack((uti[0][neighbors], uti[1][neighbors], forces))


def Boundary_forces(inner_positions, boundary_positions, Ly, boundary_cutoffs, return_mags=False, epsilon=1, alpha=2):
    '''Calculates the forces between the inner spheres and the boundary spheres
    args:
        inner_positions [Nx2 np.array]: positions of the inner spheres
        boundary_positions [Mx2 np.array]: positions of the boundary spheres
        Ly [float]: length of the periodic direction
        boundary_cutoffs [2d np.array of shape (N, M)]: boundary_cutoffs[i, j] is the cut_off distance between the ith inner sphere and the jth epsilon [float]: energy scale
        alpha [float]: exponent in the potential energy function
        return_mags [bool]: if True, returns the force magnitudes instead of the force vectors (option used for visualization)
    returns:
        forces [N'x4 np.array]: force[i] = [n, m, fx, fy] where [fx,fy] is force on the nth inner sphere due to the mth boundary sphere, N' is the number of neighbors. If return_mags is True, force[i] = [n, m, f] where f is the force magnitude
    '''
    r_diff_vec = inner_positions[:, np.newaxis, :] - boundary_positions[np.newaxis, :, :] #calculate the difference in positions between inner and boundary spheres
    r_diff_vec[:,1] = r_diff_vec[:,1] - Ly*np.round(r_diff_vec[:,1]/Ly) #calculate the shortest distance in the y direction
    r_diff = np.linalg.norm(r_diff_vec, axis = 2) #calculate the distance between the inner and boundary spheres
    neighbors = np.where(r_diff < boundary_cutoffs) #selecting the neighbors
    r_diff = r_diff[neighbors]; r_diff_vec = r_diff_vec[neighbors]; boundary_cutoffs = boundary_cutoffs[neighbors] #selecting only the neighbors
    #r_diff, boundary_cutoffs are now 1d arrays with len=num of neighbors| r_diff_vec is a 2d array 
    force_magnitudes = (epsilon/boundary_cutoffs)*(1 - r_diff/ boundary_cutoffs)**(alpha-1)
    if return_mags:
        return_force_mags = np.column_stack((neighbors[0], neighbors[1], force_magnitudes))
        return return_force_mags, r_diff_vec
    else:
        force_directions = r_diff_vec/r_diff[:, None]
        forces = force_magnitudes[:, None]*force_directions
        return np.column_stack((neighbors[0], neighbors[1], forces)) #add the indices of the sphere pairs to the forces
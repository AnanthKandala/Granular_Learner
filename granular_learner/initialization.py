import numpy as np


def Initialize_boundary_spheres(obj):
    '''Initializes the boundary spheres'''
    distance_between_spheres = 2*obj.boundary_sphere_radius + obj.boundary_sphere_gap
    input_node_pos = np.array([[0, a] for a in np.arange(0, obj.Ly, distance_between_spheres)])
    output_node_pos = np.array([[obj.Lx, a] for a in np.arange(0, obj.Ly, distance_between_spheres)])
    boundary_spheres_locs = np.vstack((input_node_pos, output_node_pos))
    boundary_sphere_radii = obj.boundary_sphere_radius*np.ones(len(boundary_spheres_locs))
    obj.boundary_spheres = np.column_stack((boundary_spheres_locs, boundary_sphere_radii))


def Initialize_inner_spheres(obj, n_spheres, distribution=[0.4, 0.6]):
    '''Selects random points on a uniform grid inside the box to place the spheres.
        Calculates the cutoffs for the short range interaction between the spheres based on the radii.
    n_spheres [int]: number of spheres of each kind
    distribution [np.array]: probability distribution of the spheres'''
    assert len(obj.inner_radii) == len(distribution), "The number of radii and the number of probabilities must be the same"
    assert np.sum(distribution) == 1, "The probabilities must sum to 1"

    grid_size =  0.9*(obj.Lx*obj.Ly/(len(obj.inner_radii)*n_spheres))**0.5 #initialize the grid size
    start = 0.9*obj.boundary_sphere_radius #place the inner spheres at some distance away from the boundary spheres
    interior_grid = np.mgrid[start:obj.Lx:grid_size, start:obj.Ly:grid_size].reshape(2,-1).T #generate the grid
    assert len(interior_grid) >= len(obj.inner_radii)*n_spheres, "The number of inner spheres is too large. Adjust the box params or the sphere proximity" #check if the number of inner spheres is too large for the box size
    selected_radii = np.random.choice(obj.inner_radii, size=len(obj.inner_radii)*n_spheres, p=distribution) #select the radii

    uti = np.triu_indices(len(obj.inner_radii)*n_spheres, k=1) #obtain unique indices of the inner sphere combinations
    obj.inner_cutoffs = selected_radii[uti[0]] + selected_radii[uti[1]] #calculate the cutoffs

    selected_indices = np.random.choice(range(len(interior_grid)), size=len(obj.inner_radii)*n_spheres, replace=False) #sample points from the grid
    inner_sphere_locs = interior_grid[selected_indices] #obtain the locations of the inner spheres
    obj.inner_spheres = np.column_stack((inner_sphere_locs, selected_radii)) #combine the locations and the radii
        


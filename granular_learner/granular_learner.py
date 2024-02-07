import numpy as np
from initialization import Initialize_boundary_spheres, Initialize_inner_spheres
from energy_func import Total_energy_func
from box_plotter import Plot_box, Force_chain_viz
from helper_funcs import Wrapper
from force_funcs import Inner_forces, Boundary_forces
from pprint import pprint

class Granular_learner:
    def __init__(self, Lx, Ly, radii, boundary_sphere_radius, boundary_sphere_gap, epsilon=1, alpha=2):
        '''Initializes the granular learner
        Lx [float]: length of the box in the x direction
        Ly [float]: length of the box in the y direction
        radii [np.array/list]: radii of the inner spheres 
        boundary_sphere_radius [float]: radius of the boundary spheres
        boundary_sphere_gap [float]: spacing between the boundary spheres
        epsilon [float]: coefficient in the energy function
        alpha [float]: exponent in the energy function'''
        self.Lx = Lx
        self.Ly = Ly
        self.inner_radii = np.unique(radii)
        self.boundary_sphere_radius = boundary_sphere_radius
        self.boundary_sphere_gap = boundary_sphere_gap
        self.epsilon = epsilon
        self.alpha = alpha
        self.boundary_spheres = None
        self.inner_spheres = None
        self.inner_cutoffs = None
        self.boundary_cutoffs = None
        self.total_energy = None
        
    def initialize_(self, n_spheres, distribution=[0.4, 0.6]):
        '''Initializes the granular learner
        n_spheres [int]: number of spheres of each kind
        distribution [np.array]: probability distribution of the spheres'''
        Initialize_boundary_spheres(self)
        Initialize_inner_spheres(self, n_spheres, distribution)
        self.calc_boundary_cutoffs_() #calculate the boundary cutoffs

    def calc_boundary_cutoffs_(self):
        '''Calculates the cutoffs for the short range interaction between the inner and boundary spheres'''
        self.boundary_cutoffs = self.inner_spheres[:, -1][:, None] + self.boundary_spheres[:,-1][None, :]

    def learner_energy_(self):
        '''Calculates the total energy of the learner at its current state'''
        self.total_energy = Total_energy_func(self.inner_spheres[:, :-1].flatten(), self.boundary_spheres[:, :-1], self.inner_cutoffs, self.boundary_cutoffs, self.Ly, self.epsilon, self.alpha)
        return self.total_energy

    def plot_box_(self, title, outimage):
        '''Plots the box
        title [str]: title of the plot
        outimage [str]: name of the output image'''
        Plot_box(self, title, outimage)
    
    def plot_force_chain_(self, title, outimage, inner_plot=(False,False),boundary_plot=(True, True)):
        '''Plots the force chain
        title [str]: title of the plot
        outimage [str]: name of the output image'''
        inner_force_mags, inner_r_diffs = Inner_forces(self.inner_spheres[:,:-1], self.Ly, self.inner_cutoffs, return_mags=True)
        boundary_force_mags, boundary_r_diffs = Boundary_forces(self.inner_spheres[:,:-1], self.boundary_spheres[:,:-1], self.Ly, self.boundary_cutoffs, return_mags=True)
        with np.printoptions(threshold=np.inf):
            print(inner_force_mags)
        Force_chain_viz(self, title, outimage, inner_force_mags, inner_r_diffs, boundary_force_mags, boundary_r_diffs, inner_plot,boundary_plot)


    def update_inner_spheres_(self, new_positions):
        '''Updates the inner spheres with the optimized positions
        optimized_positions [np.array]: optimized positions of the inner spheres'''
        positions = new_positions.reshape(-1, 2)
        positions[:, 1] = Wrapper(positions[:, 1], self.Ly)
        self.inner_spheres[:, :-1] = positions
        



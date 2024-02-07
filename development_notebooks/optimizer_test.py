import sys
path_granular = '/orange/physics-dept/an.kandala/Biophysics_projeccts/granular_learner/granular_learner_ananth/granular_learner'
#append to PATH
sys.path.append(path_granular)
from granular_learner import Granular_learner
from scipy.optimize import minimize
from functools import partial
from energy_func import Total_energy_func
import pickle



#INITIALIZATION
epsilon = 4
r1 = 1
r2 = 1.4*r1
Lx = 30
Ly = 20
inner_radii = [r1, r2]
# inner_radii = [r1]
# prob = [1]
prob = [0.5, 0.5]
#place input boundary spheres:
boundary_sphere_radius = 2
boundary_sphere_gap = 0 # gap between boundary spheres 
n_spheres = 100
learner = Granular_learner(Lx, Ly, inner_radii, boundary_sphere_radius, boundary_sphere_gap,epsilon)
learner.initialize_(n_spheres, prob)

#CALCULATE THE ENERGY
starting_energy = learner.learner_energy_()
print(f'starting_energy = {starting_energy}')

#Plot the spheres
title = f'Energy function test, E = {starting_energy}'
outimage = 'init.png'
inner_plot = (True, True)
boundary_plot = (True, True)
learner.plot_force_chain_(title, outimage, inner_plot, boundary_plot)

#function to be passed to the scipy optimizer
obj_function = partial(Total_energy_func, boundary_positions=learner.boundary_spheres[:,:-1], cut_offs=learner.inner_cutoffs, boundary_cutoffs=learner.boundary_cutoffs, Ly=learner.Ly, epsilon=learner.epsilon,  alpha=learner.alpha)

result = minimize(obj_function, learner.inner_spheres[:,:-1].flatten(), method='L-BFGS-B', tol=1e-8)
optimized_positions = result.x
#CALCULATE THE ENERGY
learner.update_inner_spheres_(optimized_positions)
file_path = 'test.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(learner, file)

total_energy = learner.learner_energy_()
print(f'starting_energy = {starting_energy}')
print(f'total_energy = {total_energy}')

#Plot the spheres
title = f'Energy function test, E = {total_energy}'
outimage = 'final.png'
learner.plot_force_chain_(title, outimage, inner_plot, boundary_plot)
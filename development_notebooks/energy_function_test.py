import sys
path_granular = '/orange/physics-dept/an.kandala/Biophysics_projeccts/granular_learner/granular_learner_ananth/granular_learner'
#append to PATH
sys.path.append(path_granular)
from granular_learner import Granular_learner

#INITIALIZATION
r1 = 1
r2 = 1.4*r1
Lx = 30
Ly = 15
inner_radii = [r1, r2]
prob = [0.4, 0.6]
#place input boundary spheres:
boundary_sphere_radius = 1
boundary_sphere_gap = 0 # gap between boundary spheres 
n_spheres = 50
learner = Granular_learner(Lx, Ly, inner_radii, boundary_sphere_radius, boundary_sphere_gap)
learner.initialize_(n_spheres, prob)

#CALCULATE THE ENERGY
total_energy = learner.learner_energy_()
print(f'total_energy = {total_energy}')

#Plot the spheres
title = f'Energy function test, E = {total_energy}'
outimage = 'test.png'
learner.plot_box_(title, outimage)



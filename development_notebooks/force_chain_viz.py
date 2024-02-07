import sys
path_granular = '/orange/physics-dept/an.kandala/Biophysics_projeccts/granular_learner/granular_learner_ananth/granular_learner'
#append to PATH
sys.path.append(path_granular)
from granular_learner import Granular_learner
from scipy.optimize import minimize
from functools import partial
from energy_func import Total_energy_func
import pickle


inner_plot = (True, True)
boundary_plot = (True, True)
file_path = 'test.pkl'

with open(file_path, 'rb') as file:
    learner = pickle.load(file)

total_energy = learner.learner_energy_()

#Plot the spheres
title = f'Energy function test, E = {total_energy}'
outimage = 'f_chain.png'
learner.plot_force_chain_(title, outimage, inner_plot, boundary_plot)
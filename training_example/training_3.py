#!/usr/bin/env python
import numpy as np 
from scipy.optimize import minimize
import sys
import copy
# Periodically drives a particular bsphere with const amplitude
#    1) takes a system at equilibrium 
#    2) changes the radius of bsphere and waits for system to equlibriate
#    3) resets the radius and wait for system to equlibriate
#    4) Saves the final energy and forces acting on the bspheres
#    5) repeat
def create_spheres(Lx, Ly, radius, radius_ratio, boundary_sphere_radius):
    points = generatePointsOnGrid(Lx, Ly, radius, boundary_sphere_radius)
    sphere_pos = np.array(points)
    sphere_radius = np.array([radius if np.random.uniform(0, 1) < 0.5 else radius_ratio * radius for _ in points])
    return sphere_pos, sphere_radius

def create_boundary_spheres(boundary_sphere_gap, Lx, Ly, boundary_sphere_radius):
    y_values = np.arange(boundary_sphere_radius, Ly, boundary_sphere_gap)
    input_bsphere_pos = np.array([[0, y] for y in y_values])
    output_bsphere_pos = np.array([[Lx, y] for y in y_values])
    input_bsphere_radius = output_bsphere_radius = np.full(len(y_values), boundary_sphere_radius)
    return input_bsphere_pos, input_bsphere_radius, output_bsphere_pos, output_bsphere_radius

def generatePointsOnGrid(Lx, Ly, radius,boundary_sphere_radius):
    spacing = 0.85*(radius + 1.4 * radius)  # Grid spacing
    points = []

    x_points = np.arange(boundary_sphere_radius, Lx-boundary_sphere_radius, spacing)
    y_points = np.arange(spacing, Ly, spacing)

    for x in x_points:
        for y in y_points:
            points.append([x, y])

    return points

def interaction_potential(r, sigma):
    """
    Interaction potential function.
    r: Distance between particles
    sigma: sum of radii of two spheres
    epsilon, alpha: Parameters of the potential
    """
    epsilon = 1.0
    alpha = 2
    with np.errstate(divide='ignore', invalid='ignore'):  # Ignore division by zero and invalid errors
        potential = np.where(r < sigma, (epsilon / alpha) * (1 - r / sigma)**alpha, 0)
    return potential

def tot_energy(sphere_pos_flat, Ly,input_bsphere_pos, output_bsphere_pos, sigma_matrix,num_input_output):

    sphere_pos = sphere_pos_flat.reshape(-1, 2)
    all_sphere_pos = np.vstack((input_bsphere_pos, output_bsphere_pos, sphere_pos))

    # Compute distance matrix using broadcasting   
    diff = all_sphere_pos[:, np.newaxis, :] - all_sphere_pos[np.newaxis, :, :] 
    # Apply periodic boundary conditions in y-direction
    diff[:, :, 1] -= np.round(diff[:, :, 1] / Ly) * Ly 
    dist_matrix = np.linalg.norm(diff, axis=-1)
    # Apply interaction potential function
    interaction_matrix = interaction_potential(dist_matrix, sigma_matrix)
    # Exclude self interactions and interactions among boundary spheres
    # Set the interactions within boundary spheres to zero
    interaction_matrix[:num_input_output, :num_input_output] = 0
    # Use only upper triangular part to avoid double counting
    upper_tri_matrix = np.triu(interaction_matrix, k=1)
    # Sum up the interaction energies
    total_energy = np.sum(upper_tri_matrix)
    return total_energy



def extract_sphere_data(sphere_dict):
    positions = np.array([s.position_ for s in sphere_dict.values()])
    radii = np.array([s.radius_ for s in sphere_dict.values()])
    return positions, radii

def create_bounds(Lx, sphere_radius):
    bounds = []
    for radius in sphere_radius:
        # x-coordinate bounds considering the radius
        x_min = radius
        x_max = Lx - radius
        bounds.extend([(x_min, x_max), (None, None)])  # No bounds for y-coordinate due to PBC
    return bounds


def update_sphere_positions(result,Ly) :

    optimized_positions = result.x
    optimized_positions_2d = optimized_positions.reshape(-1, 2)
    optimized_positions_2d[:,1] = optimized_positions_2d[:, 1] % Ly # apply PBC to output
    
    return optimized_positions_2d
  
def change_OBsphere_radius_by_percent(percent_increase,sigma_matrix,output_bsphere_radius,bsphere_id,NIBspheres):
   
    sigma_matrix_nudged = copy.deepcopy(sigma_matrix)
    output_bsphere_radius_nudged = copy.deepcopy(output_bsphere_radius)

    radius_change = output_bsphere_radius[bsphere_id]*(percent_increase / 100)

    output_bsphere_radius_nudged[bsphere_id] += radius_change
    sigma_matrix_index = NIBspheres + bsphere_id
    sigma_matrix_nudged[sigma_matrix_index, :] += radius_change
    sigma_matrix_nudged[:, sigma_matrix_index] += radius_change

    return sigma_matrix_nudged,output_bsphere_radius_nudged

def reset_radius(bsphere_dict, bsphere_id, original_radius):
    bsphere_dict[bsphere_id].setRadius(original_radius)

def dict2arr(dic) :
    return np.array([np.array(dic[k]) for k in dic.keys() ])

def calculate_sigma_matrix(input_bsphere_radius, output_bsphere_radius, sphere_radius) :

    # A2) Define inputs for energy minimization
    all_sphere_radii = np.hstack((input_bsphere_radius, output_bsphere_radius, sphere_radius))
  #  print('all_sphere_radii=',all_sphere_radii)
    # Compute sigma matrix (sum of radii for each pair)
    return all_sphere_radii[:, np.newaxis] + all_sphere_radii[np.newaxis, :]

def callback_with_grad(xi):
    global global_counter    
    if global_counter['iter'] % 1 == 0:  # Compute every 10th iteration
        current_energy = tot_energy(xi, Ly, input_bsphere_pos, output_bsphere_pos, sigma_matrix, num_input_output)
        grad = np.zeros_like(xi)
        epsilon = 1e-8  # Small perturbation for gradient approximation
        for i in range(len(xi)):
            xi_perturbed = np.copy(xi)
            xi_perturbed[i] += epsilon
            energy_perturbed = tot_energy(xi_perturbed, Ly, input_bsphere_pos, output_bsphere_pos, sigma_matrix, num_input_output)
            grad[i] = (energy_perturbed - current_energy) / epsilon
        grad_norm = np.linalg.norm(grad)
        print(f"Iteration {global_counter['iter']}: Energy = {current_energy}, Gradient Norm = {grad_norm}")
    global_counter['iter'] += 1


if __name__ == "__main__":

    args=sys.argv
    amplitude=float(args[1])  # must be multiple of delta_Lx
    process_number=int(args[2])
    total_time = 100

    # PARAMETERS      
    alpha=2
    epsilon = 1
    radius_ratio = 1.4
    Lx,Ly = 100,50
    radius=3
    boundary_sphere_radius = 4
    boundary_sphere_gap = 8 # gap between boundary spheres 
   
    

    # A) INITIALISE SPHERES
    # A1) create spheres
    sphere_pos, sphere_radius = create_spheres(Lx, Ly, radius, radius_ratio, boundary_sphere_radius)
    input_bsphere_pos, input_bsphere_radius, output_bsphere_pos, output_bsphere_radius = create_boundary_spheres(boundary_sphere_gap, Lx, Ly, boundary_sphere_radius)
    Nspheres = len(sphere_radius)
    NIBspheres= len(input_bsphere_radius)
    NOBspheres= len(output_bsphere_radius)
    # Compute number of input and output spheres
    num_input_output = NIBspheres + NOBspheres
    # Define input variable to energy function # scipy.minimize needs a flatten variable array as input
    
    sigma_matrix = calculate_sigma_matrix(input_bsphere_radius, output_bsphere_radius, sphere_radius) 
    
    
    # create dictionary of nudged output bsphere pos
    delta_Lx = 1
    no_steps = amplitude/delta_Lx
    output_bsphere_pos_nudged_dict = {}
    for Delta_Lx in np.arange(0,amplitude,delta_Lx) :
         output_bsphere_pos_nudged_dict[Delta_Lx] = output_bsphere_pos - np.column_stack( (Delta_Lx*np.ones(NOBspheres),np.zeros(NOBspheres)) )
         print('output_bsphere_pos_nudged_dict[',Delta_Lx,']=',output_bsphere_pos_nudged_dict[Delta_Lx])
   
    perturb_arr = np.hstack((np.arange(0,amplitude,delta_Lx),np.arange(amplitude-2*delta_Lx,0,-delta_Lx)))
    # print('sigma_matrix=',sigma_matrix[:,:(NIBspheres+NOBspheres)],sigma_matrix[:(NIBspheres+NOBspheres),:])
    print('norm=',np.arange(0,amplitude,delta_Lx),'flip=',np.arange(amplitude-2*delta_Lx,0,-delta_Lx))
    # training details
    print('perturb_arr=',perturb_arr)
   
    # Save system info
    f_initial_param = open("config_details"+str(process_number)+".npy","wb")
    np.save(f_initial_param,sphere_radius) # save initial positions
    np.save(f_initial_param,input_bsphere_radius)
    np.save(f_initial_param,output_bsphere_radius)
    np.save(f_initial_param,sigma_matrix)
    f_initial_param.close()
    
    
    
    f_position = open("sphere_pos"+str(process_number)+".npy","wb")

    for t in np.arange(total_time)  : 

    
        # B) TRAINING 
 
        
        # Step 1: Compress the system by an amount delta_Lx and let it relax        
        # Step 2: Do this till the system is compressed by Delta_Lx= amplitude
        # Step 3: revert the system back to original state , again in steps of delta_Lx
        
        for Delta_Lx in perturb_arr :

            print('t=',t,'Delta_Lx=',Delta_Lx)
            sphere_pos_flat = np.array(sphere_pos).flatten() # variable
            bounds = create_bounds(Lx-Delta_Lx, sphere_radius) # Bounds for minimizer, make sure the movable spheres stay within [0,Lx]

            result = minimize(tot_energy, sphere_pos_flat, args=(Ly, input_bsphere_pos,output_bsphere_pos_nudged_dict[Delta_Lx], sigma_matrix, num_input_output), method='L-BFGS-B', bounds=bounds, tol=1e-8)#, callback=callback_with_grad)
            sphere_pos = update_sphere_positions(result,Ly) # reset sphere positions in dict after minimization        
            np.save(f_position,sphere_pos) # save intermediate positions

        
    f_position.close()


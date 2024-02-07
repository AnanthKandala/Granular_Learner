import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import training_3 as tr3
import os
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

from matplotlib import rc,rcParams
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import copy

def total_force_on_OBspheres_in_x(NIBspheres,NOBspheres,inner_positions, boundary_positions, Ly, boundary_cutoffs):
    forces = Boundary_forces(inner_positions, boundary_positions, Ly, boundary_cutoffs)
    
    print('forces=',forces)
    
    forces_for_all_bsphere = forces[forces[:, 1] > NIBspheres-1] # choose forces on OBspheres 

 

    print('forces_for_bsphere=',forces_for_all_bsphere,'forces_for_bsphere[:, 2:]=',forces_for_all_bsphere[:, 2:],"np.sum((forces_for_bsphere[:, 2:]),axis=0)=",np.sum((forces_for_all_bsphere[:, 2:]),axis=0))

    force_on_bsphere_index = np.sum((forces_for_all_bsphere[:, 2:]),axis=0)

    coordination_no = len(forces_for_all_bsphere[:, 2:])
    
    total_force_in_x = -force_on_bsphere_index[0] if coordination_no > 0 else 0 # minus as we want force on the bsphere
    
    return total_force_in_x, coordination_no

def bsphere_energy(sphere_pos, boundary_pos, boundary_radius, Ly, sigma_matrix,bsphere_index):
    all_sphere_pos = np.vstack((boundary_pos, sphere_pos))
    print(sphere_pos, boundary_pos, boundary_radius, Ly, sigma_matrix,bsphere_index)

    # Compute distance matrix with periodic boundary conditions
    diff = all_sphere_pos[:, np.newaxis, :] - all_sphere_pos[np.newaxis, :, :]
    diff[:, :, 1] = np.minimum(np.abs(diff[:, :, 1]), Ly - np.abs(diff[:, :, 1]))
    dist_matrix = np.linalg.norm(diff, axis=-1)

    # Apply interaction potential function
    interaction_matrix = tr3.interaction_potential(dist_matrix, sigma_matrix)

    # Exclude self interactions and interactions among boundary spheres
    num_boundary_spheres = len(boundary_radius)
    interaction_matrix[:num_boundary_spheres, :num_boundary_spheres] = 0

    # Calculate the energy specifically for boundary spheres
    boundary_energy = np.sum(interaction_matrix[:num_boundary_spheres], axis=1)
    print('boundary_energy=',boundary_energy)
    return boundary_energy[bsphere_index]


def Wrapper(input_array, d):
    '''Applied periodic boundary conditions to the input_array by wrapping it around the box of size d'''
    return input_array - np.round(input_array/d)*d + 0.5*d

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
        forces [N'x 4 np.array]: force[k] = [i, j, x, y] where [x,y] is force on the ith inner sphere due to the jth inner sphere and N' is the number of neighbors. If return_mags is True, force[k] = [i, j, f] where f is the force magnitude
    '''
    inner_positions = inner_positions.reshape(-1, 2) #reshape the inner positions
    inner_positions[:, 1] = Wrapper(inner_positions[:, 1], Ly) #periodic boundary conditions in the y direction
    uti = np.triu_indices(len(inner_positions), k=1) #obtain unique indices of the inner sphere combinations
    r_diff_vec = inner_positions[uti[0]] - inner_positions[uti[1]] #calculate the difference in positions
    r_diff_vec[:, 1] = r_diff_vec[:, 1] + Ly*(r_diff_vec[:, 1] < -Ly/2) - Ly*(r_diff_vec[:, 1] > Ly/2) #shortest distance in the y direction
    r_diff_vec_return = r_diff_vec.copy()
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
        forces [N'x4 np.array]: force[i] = [n, m, x, y] where [x,y] is force on the nth inner sphere due to the mth boundary sphere, N' is the number of neighbors. If return_mags is True, force[i] = [n, m, f] where f is the force magnitude
    '''
    r_diff_vec = inner_positions[:, np.newaxis, :] - boundary_positions[np.newaxis, :, :] #calculate the difference in positions between inner and boundary spheres
    #r_diff_vec.shape = (inner_spheres, boundary_spheres, 2)
    r_diff_vec[:, :, 1] = r_diff_vec[:, :, 1] + Ly*(r_diff_vec[:, :, 1] < -Ly/2) - Ly*(r_diff_vec[:, :, 1] > Ly/2) #calculate the shortest distance in the y direction
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



def Force_chain_viz(Lx,Ly,inner_spheres,boundary_spheres,inner_cutoffs,title, outimage, inner_force_mags, inner_r_diffs, boundary_force_mags, boundary_r_diffs, inner_plot=(True,True),boundary_plot=(True, True)):
    '''Plots the force chains in the box
    args:
        Lx,Ly: dimentions
        title [str]: title of the plot
        outimage [str]: name of the image
        inner_plot [tuple]: (plot inner sphere outlines, plot inner sphere indices)
        boundary_plot [tuple]: (plot boundary sphere outlines, plot boundary sphere indices)
        inner_force_mags [np.array], inner_r_diffs [np.array]: non-zero inner force magnitudes (output of Inner_forces func)
        boundary_force_mags [np.array], boundary_r_diffs [np.array]: non-zero boundary force magnitudes (output of Boundary_forces func)
        '''
    def plot_force(ax, sphere_1, sphere_2, color, lw):
        '''Plots the force vector'''
        [x1, y1, r1] = sphere_1; [x2, y2, r2] = sphere_2
        ax.plot([x1, x2], [y1, y2], color=color, lw=1)
        if y1 + r1 - Ly >0 or y2+r2 - Ly > 0: #one of the end is sticking out the top of the box
            ax.plot([x1, x2], [y1 - Ly, y2 - Ly], color=color, lw=lw)
        if y1-r1 <0 or y2-r2 < 0: #one of the end is sticking out the bottom of the box
            ax.plot([x1, x2], [y1 + Ly, y2 + Ly], color=color, lw=lw)

    fig, ax = plt.subplots(dpi=300)
    fig.suptitle(title)
    ax.set_xticks([]) #remove axes
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0-0.001*Lx, Lx+0.001*Lx])
    ax.set_ylim([0, Ly])

    #plot the boundary rectangle
    rectangle = Rectangle((0, 0), Lx, Ly, edgecolor='k', facecolor='none')
    ax.axis('off')
    ax.add_patch(rectangle)
    # color = {r: to_rgba(f"C{i}") for i, r in enumerate(np.unique(inner_spheres[:, 2]))}
    color = {r: 'k' for i, r in enumerate(np.unique(inner_spheres[:, 2]))}
    r0 = np.max(inner_cutoffs)
    if np.any(inner_plot): #plot inner spheres or their numbers
        for j, [x,y,r] in enumerate(inner_spheres):
            if inner_plot[0]: #plot the inner spheres outlines
                ax.add_patch(Circle([x,y], r,  fill=None, edgecolor=color[r], alpha=0.5))
                if y + r0 > Ly:
                    # ax.text(x, y-Ly, j, horizontalalignment='center', verticalalignment='center', fontsize=6) #plot the sphere number at the center
                    ax.add_patch(Circle([x, y-Ly], r,  fill=None, edgecolor=color[r], linestyle='--', alpha=0.5))
                if y - r0 < 0:
                    # ax.text(x, y+Ly, j, horizontalalignment='center', verticalalignment='center', fontsize=6) #plot the sphere number at the center
                    ax.add_patch(Circle([x, y+Ly], r,  fill=None, edgecolor=color[r], linestyle='--', alpha=0.5))
            if inner_plot[1]: #plot the inner sphere indices
                ax.text(x, y, j, horizontalalignment='center', verticalalignment='center', fontsize=6, zorder=100) #plot the sphere number at the center
    if np.any(boundary_plot):
        for j, [x,y,r] in enumerate(boundary_spheres):
            if boundary_plot[0]: #plot boundary sphere outlines
                ax.add_patch(Circle([x,y], r,  fill=None, edgecolor='k'))
                if y+r>Ly:
                    ax.add_patch(Circle([x,y-Ly], r,  fill=None, edgecolor='k', linestyle='--'))
                if y-r<0:
                    ax.add_patch(Circle([x,y+Ly], r,  fill=None, edgecolor='k', linestyle='--'))
            if boundary_plot[1]: #plot boundary sphere indices
                ax.text(x, y, j, horizontalalignment='center', verticalalignment='center', fontsize=6) #plot the sphere number at the center

    cmap = mpl.colormaps.get_cmap('hot_r')
    forces = np.append(inner_force_mags[:,-1], boundary_force_mags[:,-1])
    vmin = np.min(forces); vmax = np.max(forces)
    normalize = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap,  norm=normalize)#np.max(force_magnitudes)))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size='5%', pad=0.1)
    fig.colorbar(sm, cax=cax)
    #plotting the inner forces
    for ind, row in enumerate(inner_force_mags):
        [i, j, f] = row
        [x1, y1, r1] = inner_spheres[int(i)]
        [dx, dy] = inner_r_diffs[ind]
        [x2, y2] = [x1-dx, y1-dy]
        r2 = inner_spheres[int(j)][-1]
        sphere_1 = [x1, y1, r1]
        sphere_2 = [x2, y2, r2]
        color = cmap(normalize(f))
        # color = 'k'
        lw = 1
        plot_force(ax, sphere_1, sphere_2, color, lw)

    #plotting the boundary forces
    for ind, row in enumerate(boundary_force_mags):
        [i, j, f] = row
        [x1, y1, r1] = inner_spheres[int(i)]
        [dx, dy] = boundary_r_diffs[ind]
        [x2, y2] = [x1-dx, y1-dy]
        r2 = boundary_spheres[int(j)][-1]
        sphere_1 = [x1, y1, r1]
        sphere_2 = [x2, y2, r2]
        color = cmap(normalize(f))
        # color = 'r'
        lw = 1
        plot_force(ax, sphere_1, sphere_2, color, lw)    
    # plt.tight_layout()
    fig.savefig(outimage, dpi=300)
    plt.close(fig)
    return 

def cutoffs_maker(sigma_matrix,NOBspheres,NIBspheres) :
        inner = sigma_matrix[(NOBspheres+NIBspheres):,(NOBspheres+NIBspheres):]
        uti = np.triu_indices_from(inner, k=1) # Get the indices for the upper triangle, excluding the diagonal (k=1)
        return inner[uti]  # Flatten the matrix to get the cutoffs for each unique inner sphere pair

def plot(initial_spheres, input_bspheres, output_bspheres, Lx, Ly):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))  # Create 1 row, 2 columns for subplots

    # Function to plot spheres
    def plot_spheres(ax, spheres, edge_color, marker_color):
        for sphere in spheres.values():

            x, y = sphere.position_
            radius = sphere.radius_

            # Draw each sphere
            circle = plt.Circle((sphere.position_[0], sphere.position_[1]), sphere.radius_, fill=False, edgecolor=edge_color, linewidth=1)
            ax.add_patch(circle)

            if y - radius < 0:
                ghost_y = y + Ly
                ghost_circle = plt.Circle((x, ghost_y), radius, fill=False, edgecolor='black', linestyle='--', linewidth=1)
                ax.add_patch(ghost_circle)

            if y + radius > Ly:
                ghost_y = y - Ly
                ghost_circle = plt.Circle((x, ghost_y), radius, fill=False, edgecolor='black', linestyle='--', linewidth=1)
                ax.add_patch(ghost_circle)


            # Plot the center of the sphere
            ax.plot(sphere.position_[0], sphere.position_[1], 'o', color=marker_color, markersize=3)
            # Label each sphere with its ID
            ax.text(sphere.position_[0], sphere.position_[1], str(sphere.id_), fontsize=8, ha='center')

    # Plot initial spheres on the first subplot (ax1)
    plot_spheres(ax, initial_spheres, edge_color='blue', marker_color='red')
    ax.set_title('Initial Positions')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal', adjustable='box')


   
    # Draw the boxes for both subplots
    box = Rectangle((0, 0), Lx, Ly, fill=False)
    ax.add_patch(box)

    # Plot the boundary spheres
    for bsphere in list(input_bspheres.values())+list(output_bspheres.values()):
        circle = plt.Circle((bsphere.position_[0], bsphere.position_[1]), bsphere.radius_, fill=False, edgecolor='green', linewidth=1)
        ax.add_patch(circle)
        ax.plot(bsphere.position_[0], bsphere.position_[1], 'o', color='black', markersize=3)


    plt.tight_layout()
    plt.show()
   
args=sys.argv
amplitude= float(args[1])
run_number = int(args[2])
# Number of runs and percentage changes
num_runs = 50
total_time = 100

alpha=2
epsilon = 1
radius_ratio = 1.4
Lx,Ly = 100,50
radius=3
boundary_sphere_radius = 4
boundary_sphere_gap = 8 # gap between boundary spheres 
   
t_arr=np.arange(total_time+1)   

sphere_pos_old, sphere_radius = tr3.create_spheres(Lx, Ly, radius, radius_ratio, boundary_sphere_radius)
input_bsphere_pos, input_bsphere_radius_old, output_bsphere_pos, output_bsphere_radius_old = tr3.create_boundary_spheres(boundary_sphere_gap, Lx, Ly, boundary_sphere_radius)



f_initial_param = open("config_details"+str(run_number)+".npy","rb")
sphere_radius = np.load(f_initial_param) # save initial positions
input_bsphere_radius = np.load(f_initial_param)
output_bsphere_radius = np.load(f_initial_param)

sigma_matrix = np.load(f_initial_param)

f_initial_param.close()


Nspheres = len(sphere_radius)
NIBspheres= len(input_bsphere_radius)
NOBspheres= len(output_bsphere_radius)
# Compute number of input and output spheres
num_input_output = NIBspheres + NOBspheres


boundary_cutoffs = np.transpose(sigma_matrix)[(NOBspheres+NIBspheres):,:(NOBspheres+NIBspheres)]

cutoffs = cutoffs_maker(sigma_matrix,NOBspheres,NIBspheres)

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


# Directory path you want to create
directory = "./run="+str(run_number)+"_dt=_["+str(0)+","+str(total_time)+"]"

# Check if the directory already exists
if not os.path.exists(directory):
# Create the directory
    os.makedirs(directory)
    print(f"Directory '{directory}' was created.")
else:
    print(f"Directory '{directory}' already exists.")
 
f_position = open("sphere_pos"+str(run_number)+".npy","rb")
        
for t in t_arr :

        
    print('amplitude=',amplitude,'run_number=',run_number,'t=',t)

    
#    print('inner_positions=',inner_positions)

    step_no = 0
    for Delta_Lx in perturb_arr :

                inner_positions = np.load(f_position) # load the initial position
            # print energies, forces, coordination number in fig title
                sphere_pos_flat = np.array(inner_positions).flatten() # variable
         
                energy_total = tr3.tot_energy(sphere_pos_flat, Ly,input_bsphere_pos, output_bsphere_pos_nudged_dict[Delta_Lx], sigma_matrix,num_input_output)
                
                total_force_magnitude,coord = total_force_on_OBspheres_in_x(NIBspheres,NOBspheres,inner_positions, np.vstack((input_bsphere_pos, output_bsphere_pos_nudged_dict[Delta_Lx])), Ly, boundary_cutoffs)
                
                plot_title = 'A='+str(amplitude)+', dx='+str(Delta_Lx)+', t='+str(t)+', step='+str(step_no)+', energy='+str(np.round(energy_total,3))+', fx='+str(np.round(total_force_magnitude,3))+', z_av='+str(np.round((coord/NOBspheres),3))

                print('t=',t,'Delta_Lx=',Delta_Lx,'en=',energy_total,'force=',total_force_magnitude )
                print('inner_positions=',inner_positions)
               
            # Convert to ananths convention for force calculations
                inner_force_mags, inner_r_diffs = Inner_forces(inner_positions, Ly, cutoffs, return_mags=True, epsilon=epsilon, alpha=alpha)
       
                boundary_force_mags, boundary_r_diffs = Boundary_forces(inner_positions, np.vstack((input_bsphere_pos,output_bsphere_pos_nudged_dict[Delta_Lx])), Ly, boundary_cutoffs,return_mags=True)

          
            # Convert to ananths convention for plotting
                inner_spheres = np.column_stack((inner_positions,sphere_radius))
                boundary_spheres = np.vstack((  np.column_stack((input_bsphere_pos ,input_bsphere_radius)),np.column_stack((output_bsphere_pos_nudged_dict[Delta_Lx],output_bsphere_radius))))   
        
                t_padded = str(t).zfill(3)  # Adjust the number of zeros based on the expected range of 't'
                step_no_padded = str(step_no).zfill(3)  # Adjust the number of zeros based on the expected range of 'step_no'
                delta_lx_padded = str(Delta_Lx).zfill(3)  # Adjust the number of zeros based on the expected range of 'Delta_Lx'

                filename = f"{directory}/t{t_padded}_step{step_no_padded}_DeltaLx{delta_lx_padded}.png"
                Force_chain_viz(Lx, Ly, inner_spheres, boundary_spheres, cutoffs, plot_title, filename, inner_force_mags, inner_r_diffs, boundary_force_mags, boundary_r_diffs)

                step_no = step_no + 1
           
f_position.close()



    
# Now you can use flattened_cut_offs as the input to your Inner_forces function





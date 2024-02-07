import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
# turn on for latex rendering   
# rc('text', usetex=True)
# rc('font', weight='bold')
# custom_preamble = {
#     "text.latex.preamble":
#         r"\usepackage{amsmath,amssymb}" # for the align, center,... environment
#         ,
#     }
# plt.rcParams.update(custom_preamble)
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle, Circle
import numpy as np
from force_funcs import Inner_forces, Boundary_forces


def Plot_box(obj, title, outimage):
    '''plots the spheres
        inner_spheres [3xN np.array]: positions of the inner spheres with their radii as the third column
        boundary_spheres [3xN np.array]: positions of the boundary spheres with their radii as the third column'''
    
    fig, ax = plt.subplots(dpi=100)
    ax.set_xticks([]) #remove axes
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, obj.Lx)
    ax.set_ylim(0, obj.Ly)
    ax.set_title(title)
    #plot boundary spheres
    for j, [x,y,r] in enumerate(obj.boundary_spheres):
        ax.add_patch(Circle([x,y], r,  fill=None, edgecolor='k'))
        if x==0:
            x = x*(1+0.1)
        else:
            x = x*(1-0.1)
        ax.text(x, y, j, horizontalalignment='center', verticalalignment='center', fontsize=6) #plot the sphere number at the center
    #plot interior spheres
    color = {r: to_rgba(f"C{i}") for i, r in enumerate(np.unique(obj.inner_spheres[:, 2]))} #generate colors
    for j, [x,y,r] in enumerate(obj.inner_spheres):
        #plot the sphere number at the center
        ax.text(x, y, j, horizontalalignment='center', verticalalignment='center', fontsize=6)
        ax.add_patch(Circle([x,y], r,  fill=None, edgecolor=color[r], alpha=0.5))
        if y + r > obj.Ly:
            ax.add_patch(Circle([x, y-obj.Ly], r,  fill=None, edgecolor=color[r], linestyle='--', alpha=0.5))
        if y - r < 0:
            ax.add_patch(Circle([x, y+obj.Ly], r,  fill=None, edgecolor=color[r], linestyle='--', alpha=0.5))

    fig.savefig(outimage, dpi=200)
    plt.close(fig)


def Force_chain_viz(obj, title, outimage, inner_force_mags, inner_r_diffs, boundary_force_mags, boundary_r_diffs, inner_plot=(False,False),boundary_plot=(True, True)):
    '''Plots the force chains in the box
    args:
        obj: granular learner object
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
        if y1 + r1 - obj.Ly >0 or y2+r2 - obj.Ly > 0: #one of the end is sticking out the top of the box
            ax.plot([x1, x2], [y1 - obj.Ly, y2 - obj.Ly], color=color, lw=lw)
        if y1-r1 <0 or y2-r2 < 0: #one of the end is sticking out the bottom of the box
            ax.plot([x1, x2], [y1 + obj.Ly, y2 + obj.Ly], color=color, lw=lw)

    fig, ax = plt.subplots(dpi=300)
    ax.set_xticks([]) #remove axes
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0-0.001*obj.Lx, obj.Lx+0.001*obj.Lx])
    ax.set_ylim([0, obj.Ly])

    #plot the boundary rectangle
    rectangle = Rectangle((0, 0), obj.Lx, obj.Ly, edgecolor='k', facecolor='none')
    ax.axis('off')
    ax.add_patch(rectangle)
    # color = {r: to_rgba(f"C{i}") for i, r in enumerate(np.unique(obj.inner_spheres[:, 2]))}
    color = {r: 'k' for i, r in enumerate(np.unique(obj.inner_spheres[:, 2]))}
    r0 = np.max(obj.inner_cutoffs)
    if np.any(inner_plot): #plot inner spheres or their numbers
        for j, [x,y,r] in enumerate(obj.inner_spheres):
            if inner_plot[0]: #plot the inner spheres outlines
                ax.add_patch(Circle([x,y], r,  fill=None, edgecolor=color[r], alpha=0.5))
                if y + r0 > obj.Ly:
                    # ax.text(x, y-obj.Ly, j, horizontalalignment='center', verticalalignment='center', fontsize=6) #plot the sphere number at the center
                    ax.add_patch(Circle([x, y-obj.Ly], r,  fill=None, edgecolor=color[r], linestyle='--', alpha=0.5))
                if y - r0 < 0:
                    # ax.text(x, y+obj.Ly, j, horizontalalignment='center', verticalalignment='center', fontsize=6) #plot the sphere number at the center
                    ax.add_patch(Circle([x, y+obj.Ly], r,  fill=None, edgecolor=color[r], linestyle='--', alpha=0.5))
            if inner_plot[1]: #plot the inner sphere indices
                ax.text(x, y, j, horizontalalignment='center', verticalalignment='center', fontsize=6, zorder=100) #plot the sphere number at the center
    if np.any(boundary_plot):
        for j, [x,y,r] in enumerate(obj.boundary_spheres):
            if boundary_plot[0]: #plot boundary sphere outlines
                ax.add_patch(Circle([x,y], r,  fill=None, edgecolor='k'))
                if y+r>obj.Ly:
                    ax.add_patch(Circle([x,y-obj.Ly], r,  fill=None, edgecolor='k', linestyle='--'))
                if y-r<0:
                    ax.add_patch(Circle([x,y+obj.Ly], r,  fill=None, edgecolor='k', linestyle='--'))
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
        [x1, y1, r1] = obj.inner_spheres[int(i)]
        [dx, dy] = inner_r_diffs[ind]
        [x2, y2] = [x1-dx, y1-dy]
        r2 = obj.inner_spheres[int(j)][-1]
        sphere_1 = [x1, y1, r1]
        sphere_2 = [x2, y2, r2]
        color = cmap(normalize(f))
        # color = 'k'
        lw = 1
        plot_force(ax, sphere_1, sphere_2, color, lw)

    #plotting the boundary forces
    for ind, row in enumerate(boundary_force_mags):
        [i, j, f] = row
        [x1, y1, r1] = obj.inner_spheres[int(i)]
        [dx, dy] = boundary_r_diffs[ind]
        [x2, y2] = [x1-dx, y1-dy]
        r2 = obj.boundary_spheres[int(j)][-1]
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
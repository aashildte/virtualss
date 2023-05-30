"""

Åshild Telle / University of Washington, Simula Research Laboratory / 2023–2022

Implementation of boundary conditions ++ for different passive deformation modes.

TODO:
    - Maybe deformations in the normal direction might make sense if we
      consider a cross-section in this direction?

"""

import dolfin as df
from mpi4py import MPI
import numpy as np

from virtualss.deformation_functions import *


def define_boundary_conditions(deformation_type, fixed_sides, mesh, V):
    
    top_dim = mesh.topology().dim()    
    define_bnd_fun = get_deformation_function_from_keywords(deformation_type, fixed_sides, top_dim)

    dimensions = get_mesh_dimensions(mesh)
    boundary_markers, ds = set_boundary_markers(mesh, dimensions)

    L = dimensions[0][1] - dimensions[0][0]  # TODO make generic

    bcs, bc_fun = define_bnd_fun(L, V, boundary_markers)

    return bcs, bc_fun


def get_deformation_function_from_keywords(deformation_type, fixed_sides, topological_dimensions):
    """

    These functions are all defined in deformation_functions.

    """

    fun_overview = {"stretch_ff" : {
                        "noslip" : {
                            2 : stretch_ff_nonslip_2D,
                            3 : stretch_ff_nonslip_3D,
                            },
                        "fixed_base" : {
                            2 : stretch_ff_fixed_base_2D,
                            3 : stretch_ff_fixed_base_3D,
                            },
                        },
                    }

    return fun_overview[deformation_type][fixed_sides][topological_dimensions]



def get_mesh_dimensions(mesh):

    dim = mesh.topology().dim()

    coords = mesh.coordinates()[:]

    xcoords = coords[:, 0]
    ycoords = coords[:, 1]

    xmin = min(xcoords)
    xmax = max(xcoords)
    ymin = min(ycoords)
    ymax = min(ycoords)

    length = xmax - xmin
    width = ymax - ymin
    
    if dim > 2:
        zcoords = coords[:, 2]
        
        zmin = min(zcoords)
        zmax = max(zcoords)
        height = zmax - zmin
        
        print(f"Domain length={length}, " + f"width={width}, " + f"height={height}")
        dimensions = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]

    else:    
        print(f"Domain length={length}, " + f"width={width}")
        dimensions = [[xmin, xmax], [ymin, ymax]]

    return dimensions


def set_boundary_markers(mesh, dimensions):
    # define subdomains
    dim = mesh.topology().dim()

    boundaries = {
        "xmin": {"subdomain": Wall(0, "min", dimensions), "idt": 1},
        "xmax": {"subdomain": Wall(0, "max", dimensions), "idt": 2},
        "ymin": {"subdomain": Wall(1, "min", dimensions), "idt": 3},
        "ymax": {"subdomain": Wall(1, "max", dimensions), "idt": 4},
    }

    if dim > 2:
        boundaries["zmin"] = {"subdomain": Wall(2, "min", dimensions), "idt": 5}
        boundaries["zmax"] = {"subdomain": Wall(2, "max", dimensions), "idt": 6}

    # Mark boundary subdomains

    boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)

    for bnd_pair in boundaries.items():
        bnd = bnd_pair[1]
        bnd["subdomain"].mark(boundary_markers, bnd["idt"])
    
    # Redefine boundary measure
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

    return boundaries, ds



class Wall(df.SubDomain):
    """

    Subdomain class; extracts coordinates for the six walls. Assumes
    all boundaryes are aligned with the x, y and z axes.

    Params:
        index: 0, 1 or 2 for x, y or z
        minmax: 'min' or 'max'; for smallest and largest values for
            chosen dimension
        dimensions: 3 x 2 array giving dimensions of the domain,
            logically following the same flow as index and minmax

    """

    def __init__(self, index, minmax, dimensions):
        super().__init__()

        assert minmax in ["min", "max"], "Error: Let minmax be 'min' or 'max'."

        # extract coordinate for min or max in the direction we're working in
        index_coord = dimensions[index][0 if minmax == "min" else 1]

        self.index, self.index_coord = index, index_coord

    def inside(self, x, on_boundary):
        index, index_coord = self.index, self.index_coord

        return df.near(x[index], index_coord, eps=1e-10) and on_boundary

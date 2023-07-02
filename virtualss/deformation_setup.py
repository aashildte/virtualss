"""

Åshild Telle / University of Washington / 2023–2022

Functions related to marking walls, getting relevant coordinates, etc. from the mesh.

"""

import dolfin as df
from mpi4py import MPI
import numpy as np


def get_mesh_dimensions(mesh):
    mpi_comm = mesh.mpi_comm()
    
    dim = mesh.topology().dim()

    coords = mesh.coordinates()[:]

    xcoords = coords[:, 0]
    ycoords = coords[:, 1]

    xmin = mpi_comm.allreduce(min(xcoords), op=MPI.MIN)
    xmax = mpi_comm.allreduce(max(xcoords), op=MPI.MAX)
    
    ymin = mpi_comm.allreduce(min(ycoords), op=MPI.MIN)
    ymax = mpi_comm.allreduce(max(ycoords), op=MPI.MAX)

    length = xmax - xmin
    width = ymax - ymin

    if dim > 2:
        zcoords = coords[:, 2]

        zmin = mpi_comm.allreduce(min(zcoords), op=MPI.MIN)
        zmax = mpi_comm.allreduce(max(zcoords), op=MPI.MAX)
        height = zmax - zmin

        print(f"Domain length={length}, " + f"width={width}, " + f"height={height}")
        dimensions = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]

    else:
        print(f"Domain length={length}, " + f"width={width}")
        dimensions = [[xmin, xmax], [ymin, ymax]]

    return dimensions


def get_corner_coords(mesh):
    mpi_comm = mesh.mpi_comm()
    
    dim = mesh.topology().dim()

    coords = mesh.coordinates()[:]

    xcoords = coords[:, 0]
    ycoords = coords[:, 1]

    xmin = mpi_comm.allreduce(min(xcoords), op=MPI.MIN)
    ymin = mpi_comm.allreduce(min(ycoords), op=MPI.MIN)

    corner_coords = [xmin, ymin]
    
    if dim > 2:
        zcoords = coords[:, 2]
    
        zmin = mpi_comm.allreduce(min(zcoords), op=MPI.MIN)
        corner_coords.append(zmin)

    return corner_coords


def get_length(mesh):
    mpi_comm = mesh.mpi_comm()
    coords = mesh.coordinates()[:]

    xcoords = coords[:, 0]

    xmin = mpi_comm.allreduce(min(xcoords), op=MPI.MIN)
    xmax = mpi_comm.allreduce(max(xcoords), op=MPI.MAX)

    return xmax - xmin


def get_width(mesh):
    mpi_comm = mesh.mpi_comm()
    coords = mesh.coordinates()[:]

    ycoords = coords[:, 1]

    ymin = mpi_comm.allreduce(min(ycoords), op=MPI.MIN)
    ymax = mpi_comm.allreduce(max(ycoords), op=MPI.MAX)

    return ymax - ymin


def get_height(mesh):
    mpi_comm = mesh.mpi_comm()
    coords = mesh.coordinates()[:]

    zcoords = coords[:, 2]

    zmin = mpi_comm.allreduce(min(zcoords), op=MPI.MIN)
    zmax = mpi_comm.allreduce(max(zcoords), op=MPI.MAX)

    return zmax - zmin


def get_boundary_markers(mesh):
    dimensions = get_mesh_dimensions(mesh)
    
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

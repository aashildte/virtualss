"""

Boundary conditions for compression/stretch along the x direction in a
Cartesian coordinate system.

"""


import dolfin as df

from virtualss.deformation_setup import (
    get_corner_coords,
    get_length,
)

def stretch_xx_fixed_sides(V, boundary_markers):
    """

    Defines boundary conditions equivalent to stretch with fixed areas
    on both sides of the domain by caling auxillary similar functions for
    2D and 3D cases respectively.

    Params:
        V - Fucntion space for the displacement function.
        boundary_markers - dictionary with subspaces and wall identities
            for all four sides of the presumably rectangular mersh.

    """

    mesh = V.mesh()
    top_dim = mesh.geometric_dimension()

    if top_dim == 2:
        return _stretch_xx_fixed_sides_2D(V, boundary_markers, mesh)
    elif top_dim == 3:
        return _stretch_xx_fixed_sides_3D(V, boundary_markers, mesh)
    else:
        raise NotImplementedError()


def _stretch_xx_fixed_sides_2D(V, boundary_markers, mesh):

    const = df.Constant([0, 0])

    length = get_length(mesh)
    bcsfun = df.Expression(("(k*L, 0)", 0), L=L, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, xmin),
        df.DirichletBC(V, bcsfun, xmax),
    ]

    return bcs, bcsfun


def _stretch_xx_fixed_sides_3D(V, boundary_markers, mesh):
    const = df.Constant([0, 0, 0])

    length = get_length(mesh)
    bcsfun = df.Expression(("k*L", 0, 0), L=length, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, xmin),
        df.DirichletBC(V, bcsfun, xmax),
    ]
    return bcs, bcsfun

def stretch_xx_xcomp(V, boundary_markers, mesh):
    """

    Defines boundary conditions equivalent to stretch with fixed areas
    on both sides of the domain by caling auxillary similar functions for
    2D and 3D cases respectively.

    Params:
        V - Fucntion space for the displacement function.
        boundary_markers - dictionary with subspaces and wall identities
            for all four sides of the presumably rectangular mersh.
        mesh - the mesh itself  #TODO do we need this?

    """

    top_dim = mesh.get_topological_dimension()

    if top_dim == 2:
        return _stretch_xx_xcomp_2D(V, boundary_markers, mesh)
    elif top_dim == 3:
        return _stretch_xx_xcomp_3D(V, boundary_markers, mesh)
    else:
        raise NotImplementedError()


def _stretch_xx_xcomp_2D(V, boundary_markers, mesh):
    # find corner point, which we will fix completely

    pt = get_corner_coords(mesh)
    cb = lambda x, on_boundary: df.near(x[0], pt[0]) and df.near(x[1], pt[1])

    # and sides, which we will fix in one component

    xmin = boundary_markers["xmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]
    length = get_length(mesh)

    bcsfun = df.Expression("k*L", L=length, k=0, degree=1)

    bcs = [
        df.DirichletBC(V, df.Constant([0, 0]), cb, "pointwise"),
        df.DirichletBC(V.sub(0), df.Constant(0), xmin),
        df.DirichletBC(V.sub(0), bcsfun, xmax),
    ]

    return bcs, bcsfun



def _stretch_xx_xcomp_3D(V, boundary_markers, mesh):

    # find corner point, which we will fix completely

    pt = get_corner_coords(mesh)
    cb = (
        lambda x, on_boundary: df.near(x[0], pt[0])
        and df.near(x[1], pt[1])
        and df.near(x[2], pt[2])
    )

    # and sides, which we will fix in one component

    xmin = boundary_markers["xmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]

    length = get_length(mesh)
    bcsfun = df.Expression("k*L", L=length, k=0, degree=1)

    bcs = [
        df.DirichletBC(V, df.Constant([0, 0, 0]), cb, "pointwise"),
        df.DirichletBC(V.sub(0), df.Constant(0), xmin),
        df.DirichletBC(V.sub(0), bcsfun, xmax),
    ]

    return bcs, bcsfun



def stretch_xx_load(F, v, boundary_markers, mesh, ds):
    facet_norm = df.FacetNormal(mesh)

    xmin_idt = boundary_markers["xmin"]["idt"]
    xmax_idt = boundary_markers["xmax"]["idt"]

    # one function, applied symmetrically

    pressure_fun = df.Expression("-k", k=0, degree=1)
    ext_pressure_min = pressure_fun * df.inner(v, df.det(F) * df.inv(F) * facet_norm) * ds(xmax_idt)
    ext_pressure_max = pressure_fun * df.inner(v, df.det(F) * df.inv(F) * facet_norm) * ds(xmin_idt)

    return [ext_pressure_min, ext_pressure_max], pressure_fun

"""

Boundary conditions for compression/stretch along the y direction in a
Cartesian coordinate system.

"""


import dolfin as df

from virtualss.deformation_setup import (
    get_corner_coords,
    get_width,
)
from virtualss.load import external_pressure_term


def stretch_yy_fixed_sides(V, boundary_markers):
    """

    Defines boundary conditions equivalent to stretch with fixed areas
    on both sides of the domain. The "ymax" side will be fixed to zero
    completely while the "ymax" side will be assigned values via the
    returned bcsfun in the x component while keeping y = z = 0.

    Args:
        V - Fucntion space for the displacement function.
        boundary_markers - dictionary with subspaces and wall identities
            for all four sides of the presumably cubical/rectangular mersh.

    Returns:
        bcs - list of DirichletBC instances
        bcsfun - function defining the behavior of the "ymax" side only

    """

    mesh = V.mesh()
    top_dim = mesh.geometric_dimension()

    if top_dim == 2:
        return _stretch_yy_fixed_sides_2D(V, boundary_markers, mesh)
    elif top_dim == 3:
        return _stretch_yy_fixed_sides_3D(V, boundary_markers, mesh)
    else:
        raise NotImplementedError()


def _stretch_yy_fixed_sides_2D(V, boundary_markers, mesh):

    const = df.Constant([0, 0])

    width = get_width(mesh)
    bcsfun = df.Expression((0, "k*L"), L=width, k=0, degree=1)

    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, ymin),
        df.DirichletBC(V, bcsfun, ymax),
    ]

    return bcs, bcsfun


def _stretch_yy_fixed_sides_3D(V, boundary_markers, mesh):
    const = df.Constant([0, 0, 0])

    width = get_width(mesh)
    bcsfun = df.Expression((0, "k*L", 0), L=width, k=0, degree=1)

    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, ymin),
        df.DirichletBC(V, bcsfun, ymax),
    ]
    return bcs, bcsfun


def stretch_yy_ycomp(V, boundary_markers):
    """

    Defines boundary conditions equivalent to stretch with fixed x comp.
    while allowing for free movement in the other directions. The "ymin" side
    will be kept at x = 0; the lower left corner will be fixed at x = y = z = 0,
    and the "ymax" side will be assigned to a fixed value as determined by the
    returned bcsfun function.

    Args:
        V - Fucntion space for the displacement function.
        boundary_markers - dictionary with subspaces and wall identities
            for all four sides of the presumably cubical/rectangular mersh.
    
    Returns:
        bcs - list of DirichletBC instances
        bcsfun - function defining the behavior of the right side only

    """

    mesh = V.mesh()
    top_dim = mesh.geometric_dimension()

    if top_dim == 2:
        return _stretch_yy_ycomp_2D(V, boundary_markers, mesh)
    elif top_dim == 3:
        return _stretch_yy_ycomp_3D(V, boundary_markers, mesh)
    else:
        raise NotImplementedError()


def _stretch_yy_ycomp_2D(V, boundary_markers, mesh):
    # find corner point, which we will fix completely

    pt = get_corner_coords(mesh)
    cb = lambda x, on_boundary: df.near(x[0], pt[0]) and df.near(x[1], pt[1])

    # and sides, which we will fix in one component

    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]
    width = get_width(mesh)

    bcsfun = df.Expression("k*L", L=width, k=0, degree=1)

    bcs = [
        df.DirichletBC(V, df.Constant([0, 0]), cb, "pointwise"),
        df.DirichletBC(V.sub(1), df.Constant(0), ymin),
        df.DirichletBC(V.sub(1), bcsfun, ymax),
    ]

    return bcs, bcsfun


def _stretch_yy_ycomp_3D(V, boundary_markers, mesh):
    # find corner point, which we will fix completely

    pt = get_corner_coords(mesh)
    cb = (
        lambda x, on_boundary: df.near(x[0], pt[0])
        and df.near(x[1], pt[1])
        and df.near(x[2], pt[2])
    )

    # and sides, which we will fix in one component

    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    width = get_width(mesh)
    bcsfun = df.Expression("k*L", L=width, k=0, degree=1)

    bcs = [
        df.DirichletBC(V, df.Constant([0, 0, 0]), cb, "pointwise"),
        df.DirichletBC(V.sub(1), df.Constant(0), ymin),
        df.DirichletBC(V.sub(1), bcsfun, ymax),
    ]

    return bcs, bcsfun


def stretch_yy_load(F, v, mesh, boundary_markers, ds):
    ymin_idt = boundary_markers["ymin"]["idt"]
    ymax_idt = boundary_markers["ymax"]["idt"]

    # one function, applied symmetrically

    ext_pressure_fun = df.Expression("-k", k=0, degree=1)
    ext_pressure_min = external_pressure_term(
        ext_pressure_fun, F, v, mesh, ds(ymin_idt)
    )
    ext_pressure_max = external_pressure_term(
        ext_pressure_fun, F, v, mesh, ds(ymax_idt)
    )

    return [ext_pressure_min, ext_pressure_max], ext_pressure_fun

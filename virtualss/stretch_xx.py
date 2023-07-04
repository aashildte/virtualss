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
    on both sides of the domain. The left side will be fixed to zero
    completely while the right side will be assigned values via the
    returned bcsfun in the x component while keeping y = z = 0.

    Params:
        V - Fucntion space for the displacement function.
        boundary_markers - dictionary with subspaces and wall identities
            for all four sides of the presumably rectangular mersh.

    Returns:
        bcs - list of DirichletBC instances
        bcsfun - function defining the behavior of the right side only

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


def stretch_xx_xcomp(V, boundary_markers):
    """

    Defines boundary conditions equivalent to stretch with fixed x comp.
    while allowing for free movement in the other directions. The left side
    will be kept at x = 0; the lower left corner will be fixed at x = y = z = 0,
    and the right side will be assigned to a fixed value as determined by the
    returned bcsfun function.

    Params:
        V - Fucntion space for the displacement function.
        boundary_markers - dictionary with subspaces and wall identities
            for all four sides of the presumably rectangular mersh.

    """

    mesh = V.mesh()
    top_dim = mesh.geometric_dimension()

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


def stretch_xx_load(F, v, mesh, boundary_markers, ds):
    facet_norm = df.FacetNormal(mesh)

    xmin_idt = boundary_markers["xmin"]["idt"]
    xmax_idt = boundary_markers["xmax"]["idt"]

    # one function, applied symmetrically

    ext_pressure_fun = df.Expression("-k", k=0, degree=1)
    ext_pressure_min = (
        ext_pressure_fun
        * df.inner(v, df.det(F) * df.inv(F) * facet_norm)
        * ds(xmax_idt)
    )
    ext_pressure_max = (
        ext_pressure_fun
        * df.inner(v, df.det(F) * df.inv(F) * facet_norm)
        * ds(xmin_idt)
    )

    return [ext_pressure_min, ext_pressure_max], ext_pressure_fun

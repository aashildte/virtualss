"""

Boundary conditions for compression/stretch along the z direction in a
Cartesian coordinate system.

"""

import dolfin as df

from virtualss.deformation_setup import (
    get_corner_coords,
    get_width,
)
from virtualss.load import external_pressure_term


def stretch_zz_fixed_sides(V, boundary_markers):
    """

    Defines boundary conditions equivalent to stretch with fixed areas
    on both sides of the domain. The "zmax" side will be fixed to zero
    completely while the "zmax" side will be assigned values via the
    returned bcsfun in the x component while keeping y = z = 0.

    Args:
        V - Fucntion space for the displacement function.
        boundary_markers - dictionary with subspaces and wall identities
            for all four sides of the presumably cubical/rectangular mersh.

    Returns:
        bcs - list of DirichletBC instances
        bcsfun - function defining the behavior of the "zmax" side only

    """

    mesh = V.mesh()
    top_dim = mesh.geometric_dimension()

    assert top_dim == 3, "Error: Stretch in the z direction only makes sense in 3D"

    const = df.Constant([0, 0, 0])

    width = get_width(mesh)
    bcsfun = df.Expression((0, 0, "k*L"), L=width, k=0, degree=1)

    zmin = boundary_markers["zmin"]["subdomain"]
    zmax = boundary_markers["zmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, zmin),
        df.DirichletBC(V, bcsfun, zmax),
    ]
    return bcs, bcsfun


def stretch_zz_zcomp(V, boundary_markers):
    """

    Defines boundary conditions equivalent to stretch with fixed x comp.
    while allowing for free movement in the other directions. The "zmin" side
    will be kept at x = 0; the lower left corner will be fixed at x = y = z = 0,
    and the "zmax" side will be assigned to a fixed value as determined by the
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
    
    assert top_dim == 3, "Error: Stretch in the z direction only makes sense in 3D"

    # find corner point, which we will fix completely

    pt = get_corner_coords(mesh)
    cb = (
        lambda x, on_boundary: df.near(x[0], pt[0])
        and df.near(x[1], pt[1])
        and df.near(x[2], pt[2])
    )

    # and sides, which we will fix in one component

    zmin = boundary_markers["zmin"]["subdomain"]
    zmax = boundary_markers["zmax"]["subdomain"]

    width = get_width(mesh)
    bcsfun = df.Expression("k*L", L=width, k=0, degree=1)

    bcs = [
        df.DirichletBC(V, df.Constant([0, 0, 0]), cb, "pointwise"),
        df.DirichletBC(V.sub(2), df.Constant(0), zmin),
        df.DirichletBC(V.sub(2), bcsfun, zmax),
    ]

    return bcs, bcsfun


def stretch_zz_load(F, v, mesh, boundary_markers, ds):
    zmin_idt = boundary_markers["zmin"]["idt"]
    zmax_idt = boundary_markers["zmax"]["idt"]

    # one function, applied symmetrically

    ext_pressure_fun = df.Expression("-k", k=0, degree=1)
    ext_pressure_min = external_pressure_term(
        ext_pressure_fun, F, v, mesh, ds(zmin_idt)
    )
    ext_pressure_max = external_pressure_term(
        ext_pressure_fun, F, v, mesh, ds(zmax_idt)
    )

    return [ext_pressure_min, ext_pressure_max], ext_pressure_fun

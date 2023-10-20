import dolfin as df
from virtualss.deformation_setup import get_corner_coords, get_length, get_width, get_height


def shear_zx_fixed_sides(V, boundary_markers):
    """

    Defines boundary conditions equivalent to stretch with fixed areas
    on both sides of the domain. The "zmin" side will be fixed to zero
    completely while the "zmax" side will be assigned values via the
    returned bcsfun in the z component while keeping x = y = 0.

    Args:
        V - Fucntion space for the displacement function.
        boundary_markers - dictionary with subspaces and wall identities
            for all four sides of the presumably cubical/rectangular mesh.

    Returns:
        bcs - list of DirichletBC instances
        bcsfun - function defining the behavior of the "zmax" side only

    """

    mesh = V.mesh()
    top_dim = mesh.geometric_dimension()

    if top_dim == 2:
        raise NotImplementedError("Error: zx shear not implemented in 2D.")
    elif top_dim == 3:
        return _shear_zx_fixed_sides_3D(V, boundary_markers, mesh)
    else:
        raise NotImplementedError()


def _shear_zx_fixed_sides_3D(V, boundary_markers, mesh):
    const = df.Constant([0, 0, 0])

    length = get_length(mesh)
    bcsfun = df.Expression(("k*L", 0, 0), L=length, k=0, degree=1)

    zmin = boundary_markers["zmin"]["subdomain"]
    zmax = boundary_markers["zmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, zmin),
        df.DirichletBC(V, bcsfun, zmax),
    ]
    return bcs, bcsfun


def shear_zx_comp(V, boundary_markers):
    """

    Defines boundary conditions equivalent to stretch with fixed x comp.
    while allowing for free movement in the other directions. The "zmin" side
    will be kept at z = 0; the lower left corner will be fixed at x = y = z = 0,
    and the "zmax" side will be assigned to a fixed value in the x component
    with value as determined by the returned bcsfun function.

    Args:
        V - Fucntion space for the displacement function.
        boundary_markers - dictionary with subspaces and wall identities
            for all four sides of the presumably cubical/rectangular mesh.

    Returns:
        bcs - list of DirichletBC instances
        bcsfun - function defining the behavior of the "zmax" side only

    """

    mesh = V.mesh()
    top_dim = mesh.geometric_dimension()

    if top_dim == 2:
        raise NotImplementedError("Error: zx shear not implemented in 2D.")
    elif top_dim == 3:
        return _shear_zx_comp_3D(V, boundary_markers, mesh)
    else:
        raise NotImplementedError()


def _shear_zx_comp_3D(V, boundary_markers, mesh):
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

    length = get_length(mesh)
    bcsfun = df.Expression("k*L", L=length, k=0, degree=1)

    bcs = [
        df.DirichletBC(V, df.Constant([0, 0, 0]), cb, "pointwise"),
        df.DirichletBC(V.sub(0), df.Constant(0), zmin),
        df.DirichletBC(V.sub(0), bcsfun, zmax),
    ]

    return bcs, bcsfun


def shear_zx_load(F, v, mesh, boundary_markers, ds):

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

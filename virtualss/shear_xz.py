import dolfin as df
from virtualss.deformation_setup import get_corner_coords, get_length, get_width, get_height


def simple_shear_xz(V, boundary_markers):
    """

    Defines boundary conditions equivalent to stretch with fixed areas
    on both sides of the domain. The "xmin" side will be fixed to zero
    completely while the "xmax" side will be assigned values via the
    returned bcsfun in the z component while keeping x = y = 0.

    Args:
        V - Fucntion space for the displacement function.
        boundary_markers - dictionary with subspaces and wall identities
            for all four sides of the presumably cubical/rectangular mesh.

    Returns:
        bcs - list of DirichletBC instances
        bcsfun - function defining the behavior of the "xmax" side only

    """

    mesh = V.mesh()
    top_dim = mesh.geometric_dimension()

    if top_dim == 2:
        raise NotImplementedError("Error: xz shear not implemented in 2D.")
    elif top_dim == 3:
        return _simple_shear_xz_3D(V, boundary_markers, mesh)
    else:
        raise NotImplementedError()


def _simple_shear_xz_3D(V, boundary_markers, mesh):
    const = df.Constant([0, 0, 0])

    length = get_length(mesh)
    bcsfun = df.Expression((0, 0, "k*L"), L=length, k=0, degree=1)

    xmin = boundary_markers["xmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, xmin),
        df.DirichletBC(V, bcsfun, xmax),
    ]
    return bcs, bcsfun


def shear_xz_comp(V, boundary_markers):
    """

    Defines boundary conditions equivalent to stretch with fixed x comp.
    while allowing for free movement in the other directions. The "xmin" side
    will be kept at x = 0; the lower left corner will be fixed at x = y = z = 0,
    and the "xmax" side will be assigned to a fixed value in the z component
    with value as determined by the returned bcsfun function.

    Args:
        V - Fucntion space for the displacement function.
        boundary_markers - dictionary with subspaces and wall identities
            for all four sides of the presumably cubical/rectangular mesh.

    Returns:
        bcs - list of DirichletBC instances
        bcsfun - function defining the behavior of the "xmax" side only

    """

    mesh = V.mesh()
    top_dim = mesh.geometric_dimension()

    if top_dim == 2:
        raise NotImplementedError("Error: xz shear not implemented in 2D.")
    elif top_dim == 3:
        return _shear_xz_comp_3D(V, boundary_markers, mesh)
    else:
        raise NotImplementedError()


def _shear_xz_comp_3D(V, boundary_markers, mesh):
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
        df.DirichletBC(V.sub(2), bcsfun, xmax),
    ]

    return bcs, bcsfun

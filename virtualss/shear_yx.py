import dolfin as df

from virtualss.deformation_setup import get_corner_coords, get_length, get_width, get_height


def shear_yx_fixed_sides(V, boundary_markers):
    """

    Defines boundary conditions equivalent to stretch with fixed areas
    on both sides of the domain. The "ymin" side will be fixed to zero
    completely while the "ymax" side will be assigned values via the
    returned bcsfun in the x component while keeping y = z = 0.

    Args:
        V - Fucntion space for the displacement function.
        boundary_markers - dictionary with subspaces and wall identities
            for all four sides of the presumably cubical/rectangular mesh.

    Returns:
        bcs - list of DirichletBC instances
        bcsfun - function defining the behavior of the "ymax" side only

    """

    mesh = V.mesh()
    top_dim = mesh.geometric_dimension()

    if top_dim == 2:
        return _shear_yx_fixed_sides_2D(V, boundary_markers, mesh)
    elif top_dim == 3:
        return _shear_yx_fixed_sides_3D(V, boundary_markers, mesh)
    else:
        raise NotImplementedError()


def _shear_yx_fixed_sides_2D(V, boundary_markers, mesh):
    const = df.Constant([0, 0])
    
    width = get_width(mesh)
    bcsfun = df.Expression(("k*W", 0), W=width, k=0, degree=1)

    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, ymin),
        df.DirichletBC(V, bcsfun, ymax),
    ]
    return bcs, bcsfun


def _shear_yx_fixed_sides_3D(V, boundary_markers, mesh):
    const = df.Constant([0, 0, 0])

    width = get_width(mesh)
    bcsfun = df.Expression(("k*W", 0, 0), W=width, k=0, degree=1)

    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, ymin),
        df.DirichletBC(V, bcsfun, ymax),
    ]
    return bcs, bcsfun

def XX_shear_yx_fixed_sides_3D(V, boundary_markers, mesh):
    const = df.Constant([0, 0, 0])

    width = get_width(mesh)
    bcsfun = df.Expression(("k*W", 0, 0), W=width, k=0, degree=2)

    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, bcsfun, ymin),
        df.DirichletBC(V, bcsfun, ymax),
    ]
    return bcs, bcsfun


def shear_yx_comp(V, boundary_markers):
    """

    Defines boundary conditions equivalent to stretch with fixed y comp.
    while allowing for free movement in the other directions. The "ymin" side
    will be kept at x = 0; the lower left corner will be fixed at x = y = z = 0,
    and the "ymax" side will be assigned to a fixed value in the x component
    with value as determined by the returned bcsfun function.

    Args:
        V - Fucntion space for the displacement function.
        boundary_markers - dictionary with subspaces and wall identities
            for all four sides of the presumably cubical/rectangular mesh.

    Returns:
        bcs - list of DirichletBC instances
        bcsfun - function defining the behavior of the "ymax" side only

    """

    mesh = V.mesh()
    top_dim = mesh.geometric_dimension()

    if top_dim == 2:
        return _shear_yx_comp_2D(V, boundary_markers, mesh)
    elif top_dim == 3:
        return _shear_yx_comp_3D(V, boundary_markers, mesh)
    else:
        raise NotImplementedError()


def _shear_yx_comp_2D(V, boundary_markers, mesh):
    # find corner point, which we will fix completely

    pt = get_corner_coords(mesh)
    cb = lambda x, on_boundary: df.near(x[0], pt[0]) and df.near(x[1], pt[1])

    # and sides, which we will fix in one component

    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]
    
    width = get_width(mesh)
    bcsfun = df.Expression("k*W", W=width, k=0, degree=1)

    bcs = [
        df.DirichletBC(V, df.Constant([0, 0]), cb, "pointwise"),
        df.DirichletBC(V.sub(0), df.Constant(0), ymin),
        df.DirichletBC(V.sub(0), bcsfun, ymax),
    ]

    return bcs, bcsfun


def _shear_yx_comp_3D(V, boundary_markers, mesh):
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

    length = get_width(mesh)
    bcsfun = df.Expression("k*W", W=width, k=0, degree=1)

    bcs = [
        df.DirichletBC(V, df.Constant([0, 0, 0]), cb, "pointwise"),
        df.DirichletBC(V.sub(0), df.Constant(0), ymin),
        df.DirichletBC(V.sub(0), bcsfun, ymax),
    ]

    return bcs, bcsfun

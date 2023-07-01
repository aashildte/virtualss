import dolfin as df

from virtualss.deformation_setup import get_corner_coords

def stretch_ff_fixed_base_2D(L, V, boundary_markers):
    const = df.Constant([0, 0])
    bcsfun = df.Expression(("k*L", 0), L=L, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, xmin),
        df.DirichletBC(V, bcsfun, xmax),
    ]
    return bcs, bcsfun


def stretch_ff_componentwise_2D(L, V, boundary_markers):
    bcsfun = df.Expression("k*L", L=L, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    ymin = boundary_markers["ymin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V.sub(0), df.Constant(0.0), xmin),
        df.DirichletBC(V.sub(1), df.Constant(0.0), ymin),
        df.DirichletBC(V.sub(0), bcsfun, xmax),
    ]

    return bcs, bcsfun


def stretch_ff_xcomp_2D(L, V, boundary_markers, mesh):
    # find corner point, which we will fix completely

    pt = get_corner_coords(mesh)
    cb = lambda x, on_boundary: df.near(x[0], pt[0]) and df.near(x[1], pt[1])

    # and sides, which we will fix in one component

    xmin = boundary_markers["xmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]
    
    bcsfun = df.Expression("k*L", L=L, k=0, degree=1)

    bcs = [
        df.DirichletBC(V, df.Constant([0, 0]), cb, "pointwise"),
        df.DirichletBC(V.sub(0), df.Constant(0), xmin),
        df.DirichletBC(V.sub(0), bcsfun, xmax),
    ]

    return bcs, bcsfun


def shear_fs_fixed_base_2D(L, V, boundary_markers):
    const = df.Constant([0, 0])
    bcsfun = df.Expression((0, "k*L"), L=L, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, xmin),
        df.DirichletBC(V, bcsfun, xmax),
    ]
    return bcs, bcsfun


def shear_sf_fixed_base_2D(L, V, boundary_markers):
    const = df.Constant([0, 0])
    bcsfun = df.Expression(("k*L", 0), L=L, k=0, degree=2)

    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, ymin),
        df.DirichletBC(V, bcsfun, ymax),
    ]
    return bcs, bcsfun


def stretch_ss_fixed_base_2D(L, V, boundary_markers):
    const = df.Constant([0, 0])
    bcsfun = df.Expression((0, "k*L"), L=L, k=0, degree=2)

    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, ymin),
        df.DirichletBC(V, bcsfun, ymax),
    ]
    return bcs, bcsfun


def stretch_ss_componentwise_2D(L, V, boundary_markers):
    bcsfun = df.Expression("k*L", L=L, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    bcs = [
        df.DirichletBC(V.sub(0), df.Constant(0.0), xmin),
        df.DirichletBC(V.sub(1), df.Constant(0.0), ymin),
        df.DirichletBC(V.sub(0), bcsfun, ymax),
    ]
    return bcs, bcsfun

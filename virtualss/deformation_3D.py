import dolfin as df

from virtualss.deformation_setup import get_corner_coords

def stretch_ff_fixed_base_3D(L, V, boundary_markers):
    const = df.Constant([0, 0, 0])
    bcsfun = df.Expression(("k*L", 0, 0), L=L, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, xmin),
        df.DirichletBC(V, bcsfun, xmax),
    ]
    return bcs, bcsfun


def stretch_ff_componentwise_3D(L, V, boundary_markers):
    bcsfun = df.Expression("k*L", L=L, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    ymin = boundary_markers["ymin"]["subdomain"]
    zmin = boundary_markers["zmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V.sub(0), df.Constant(0.0), xmin),
        df.DirichletBC(V.sub(1), df.Constant(0.0), ymin),
        df.DirichletBC(V.sub(2), df.Constant(0.0), zmin),
        df.DirichletBC(V.sub(0), bcsfun, xmax),
    ]
    return bcs, bcsfun
    

def stretch_ff_xcomp_3D(L, V, boundary_markers, mesh):

    # find corner point, which we will fix completely

    pt = get_corner_coords(mesh)
    cb = lambda x, on_boundary: df.near(x[0], pt[0]) and df.near(x[1], pt[1]) and df.near(x[2], pt[2])

    # and sides, which we will fix in one component
    
    xmin = boundary_markers["xmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]
    
    bcsfun = df.Expression("k*L", L=L, k=0, degree=1)

    bcs = [
        df.DirichletBC(V, df.Constant([0, 0, 0]), cb, "pointwise"),
        df.DirichletBC(V.sub(0), df.Constant(0), xmin),
        df.DirichletBC(V.sub(0), bcsfun, xmax),
    ]

    return bcs, bcsfun


def stretch_ff_load_3D(state, test_state, F, mesh, boundary_markers, ds):
    
    u, p, r = df.split(state)
    v, q, s = df.split(test_state)

    # external prerssure term
    N = df.FacetNormal(mesh)

    xmin_idt = boundary_markers["xmin"]["idt"]
    xmax_idt = boundary_markers["xmax"]["idt"]

    pressure_fun = df.Expression("-k", k=0, degree=2)
    Gext1 = pressure_fun * df.inner(v, df.det(F) * df.inv(F) * N) * ds(xmax_idt)
    Gext2 = pressure_fun * df.inner(v, df.det(F) * df.inv(F) * N) * ds(xmin_idt)

    Gext = Gext1 + Gext2

    # rigid motion term
    
    position = df.SpatialCoordinate(mesh)

    RM = [
        df.Constant((1, 0, 0)),
        df.Constant((0, 1, 0)),
        df.Constant((0, 0, 1)),
        df.cross(position, df.Constant((1, 0, 0))),
        df.cross(position, df.Constant((0, 1, 0))),
        df.cross(position, df.Constant((0, 0, 1))),
    ]
    Pi = sum(df.dot(u, zi) * r[i] * df.dx for i, zi in enumerate(RM))

    rm = df.derivative(Pi, state, test_state)

    return Gext, pressure_fun, rm


def shear_fs_fixed_base_3D(L, V, boundary_markers):
    const = df.Constant([0, 0, 0])
    bcsfun = df.Expression((0, "k*L", 0), L=L, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, xmin),
        df.DirichletBC(V, bcsfun, xmax),
    ]
    return bcs, bcsfun


def shear_fn_fixed_base_3D(L, V, boundary_markers):
    const = df.Constant([0, 0, 0])
    bcsfun = df.Expression((0, 0, "k*L"), L=L, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, xmin),
        df.DirichletBC(V, bcsfun, xmax),
    ]
    return bcs, bcsfun


def shear_sf_fixed_base_3D(L, V, boundary_markers):
    const = df.Constant([0, 0, 0])
    bcsfun = df.Expression(("k*L", 0, 0), L=L, k=0, degree=2)

    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, ymin),
        df.DirichletBC(V, bcsfun, ymax),
    ]
    return bcs, bcsfun


def stretch_ss_fixed_base_3D(L, V, boundary_markers):
    const = df.Constant([0, 0, 0])
    bcsfun = df.Expression((0, "k*L", 0), L=L, k=0, degree=2)

    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, ymin),
        df.DirichletBC(V, bcsfun, ymax),
    ]
    return bcs, bcsfun


def stretch_ss_componentwise_3D(L, V, boundary_markers):
    bcsfun = df.Expression("k*L", L=L, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    ymin = boundary_markers["ymin"]["subdomain"]
    zmin = boundary_markers["zmin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    bcs = [
        df.DirichletBC(V.sub(0), df.Constant(0.0), xmin),
        df.DirichletBC(V.sub(1), df.Constant(0.0), ymin),
        df.DirichletBC(V.sub(2), df.Constant(0.0), zmin),
        df.DirichletBC(V.sub(0), bcsfun, ymax),
    ]
    return bcs, bcsfun


def shear_sn_fixed_base_3D(L, V, boundary_markers):
    const = df.Constant([0, 0, 0])
    bcsfun = df.Expression((0, 0, "k*L"), L=L, k=0, degree=2)

    ymin = boundary_markers["ymin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, ymin),
        df.DirichletBC(V, bcsfun, ymax),
    ]
    return bcs, bcsfun


def shear_nf_fixed_base_3D(L, V, boundary_markers):
    const = df.Constant([0, 0, 0])
    bcsfun = df.Expression(("k*L", 0, 0), L=L, k=0, degree=2)

    zmin = boundary_markers["zmin"]["subdomain"]
    zmax = boundary_markers["zmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, xmin),
        df.DirichletBC(V, bcsfun, xmax),
    ]
    return bcs, bcsfun


def shear_ns_fixed_base_3D(L, V, boundary_markers):
    const = df.Constant([0, 0, 0])
    bcsfun = df.Expression((0, "k*L", 0), L=L, k=0, degree=2)

    zmin = boundary_markers["zmin"]["subdomain"]
    zmax = boundary_markers["zmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, zmin),
        df.DirichletBC(V, bcsfun, zmax),
    ]
    return bcs, bcsfun


def stretch_nn_fixed_base_3D(L, V, boundary_markers):
    const = df.Constant([0, 0, 0])
    bcsfun = df.Expression((0, 0, "k*L"), L=L, k=0, degree=2)

    zmin = boundary_markers["zmin"]["subdomain"]
    zmax = boundary_markers["zmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V, const, zmin),
        df.DirichletBC(V, bcsfun, zmax),
    ]
    return bcs, bcsfun


def stretch_nn_componentwise_3D(L, V, boundary_markers):
    bcsfun = df.Expression("k*L", L=L, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    ymin = boundary_markers["ymin"]["subdomain"]
    zmin = boundary_markers["zmin"]["subdomain"]
    zmax = boundary_markers["zmax"]["subdomain"]

    bcs = [
        df.DirichletBC(V.sub(0), df.Constant(0.0), xmin),
        df.DirichletBC(V.sub(1), df.Constant(0.0), ymin),
        df.DirichletBC(V.sub(2), df.Constant(0.0), zmin),
        df.DirichletBC(V.sub(0), bcsfun, zmax),
    ]
    return bcs, bcsfun

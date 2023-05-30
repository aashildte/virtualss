
import dolfin as df


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


def stretch_ff_noslip_3D(L, V, boundary_markers):
    bcsfun = df.Expression("k*L", L=L, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    ymin = boundary_markers["ymin"]["subdomain"]
    zmin = boundary_markers["zmin"]["subdomain"]
    xmax = boundary_markers["xmax"]["subdomain"]

    bcs = [
            df.DirichletBC(V.sub(0), df.Constant(0.), xmin),
            df.DirichletBC(V.sub(1), df.Constant(0.), ymin),
            df.DirichletBC(V.sub(2), df.Constant(0.), zmin),
            df.DirichletBC(V.sub(0), bcsfun, xmax),
    ]
    return bcs, bcsfun


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


def stretch_ss_noslip_3D(L, V, boundary_markers):
    bcsfun = df.Expression("k*L", L=L, k=0, degree=2)

    xmin = boundary_markers["xmin"]["subdomain"]
    ymin = boundary_markers["ymin"]["subdomain"]
    zmin = boundary_markers["zmin"]["subdomain"]
    ymax = boundary_markers["ymax"]["subdomain"]

    bcs = [
            df.DirichletBC(V.sub(0), df.Constant(0.), xmin),
            df.DirichletBC(V.sub(1), df.Constant(0.), ymin),
            df.DirichletBC(V.sub(2), df.Constant(0.), zmin),
            df.DirichletBC(V.sub(0), bcsfun, ymax),
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


def stretch_nn_noslip_3D(L, V, boundary_markers):
    bcsfun = df.Expression("k*L", L=L, k=0, degree=2)

    print(boundary_markers)

    xmin = boundary_markers["xmin"]["subdomain"]
    ymin = boundary_markers["ymin"]["subdomain"]
    zmin = boundary_markers["zmin"]["subdomain"]
    zmax = boundary_markers["zmax"]["subdomain"]

    bcs = [
            df.DirichletBC(V.sub(0), df.Constant(0.), xmin),
            df.DirichletBC(V.sub(1), df.Constant(0.), ymin),
            df.DirichletBC(V.sub(2), df.Constant(0.), zmin),
            df.DirichletBC(V.sub(0), bcsfun, zmax),
    ]
    return bcs, bcsfun


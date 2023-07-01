
import dolfin as df
from mpi4py import MPI


def evaluate_normal_load(F, P, mesh, ds, wall_idt):
    normal_vector = df.FacetNormal(mesh)

    return evaluate_load(F, P, mesh, ds, wall_idt, normal_vector)

def evaluate_shear_load(F, P, mesh, ds, wall_idt):
    
    top_dim = mesh.topology().dim()

    unit_vectors = {
            1: {
                2 : df.as_vector([-1.0, 0.0]),
                3 : df.as_vector([-1.0, 0.0, 0.0]),
                },
            2: {
                2 : df.as_vector([1.0, 0.0]),
                3 : df.as_vector([1.0, 0.0, 0.0]),
                },
            3: {
                2 : df.as_vector([0.0, -1.0]),
                3 : df.as_vector([0.0, -1.0, 0.0]),
                },
            4: {
                2 : df.as_vector([0.0, 1.0]),
                3 : df.as_vector([0.0, 1.0, 0.0]),
                },
            5: {
                3 : df.as_vector([0.0, 0.0, -1.0]),
                },
            6: {
                3 : df.as_vector([0.0, 0.0, 1.0]),
                }
            }

    unit_vector = unit_vectors[wall_idt][top_dim]

    return evaluate_load(F, P, mesh, ds, wall_idt, unit_vector)


def evaluate_load(F, P, mesh, ds, wall_idt, unit_vector):
    normal_vector = df.FacetNormal(mesh)

    load = df.inner(P * normal_vector, unit_vector)
    total_load = df.assemble(load * ds(wall_idt))
    area = df.assemble(         # = total length in 2D
        df.det(F)
        * df.inner(df.inv(F).T * unit_vector, unit_vector)
        * ds(wall_idt)
    )

    return total_load/area


def _evaluate_ds(F, f, mesh, ds, wall_idt):
    normal_vector = df.FacetNormal(mesh)

    area = df.assemble(         # = total length in 2D
        df.det(F)
        * df.inner(df.inv(F).T * normal_vector, normal_vector)
        * ds(wall_idt)
    )
    return df.assemble(f*ds(wall_idt)) / area




def evaluate_stretch(u, mesh, ds):
    """

    Taken as average displacement at the "xmax" wall minus the "xmin" wall,
    divided by original domain length.

    """
    
    top_dim = mesh.topology().dim()
    if top_dim == 2:
        dir_vector = df.as_vector([1., 0.])
    elif top_dim == 3:
        dir_vector = df.as_vector([1., 0., 0.])
    else:
        raise NotImplementedError()

    d = len(u)
    I = df.Identity(d)
    F = df.variable(I + df.grad(u))

    xcomp = df.inner(u, dir_vector)
    disp_min = df.assemble(xcomp*ds(1))     # TODO generalize
    disp_max = df.assemble(xcomp*ds(2))

    mpi_comm = mesh.mpi_comm()
    coords = mesh.coordinates()[:]

    xcoords = coords[:, 0]
    xmin = mpi_comm.allreduce(min(xcoords), op=MPI.MIN)
    xmax = mpi_comm.allreduce(max(xcoords), op=MPI.MAX)
    length = xmax - xmin

    relative_shortening = (disp_max - disp_min)/length

    return relative_shortening


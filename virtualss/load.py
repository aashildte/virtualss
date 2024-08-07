
import dolfin as df
from mpi4py import MPI


def external_pressure_term(external_pressure_fun, F, v, mesh, ds, dir_vector=None):

    if dir_vector is None:      # assume facet normal
        dir_vector = df.FacetNormal(mesh)

    return external_pressure_fun* df.inner(v, df.det(F) * df.inv(F) * dir_vector) * ds


def _get_normal_vector(wall_idt, dim):
    # this doesn't work with the projection, but the point is to resemble this operation:
    # normal_vector = df.FacetNormal(mesh)
    
    normal_vectors = {
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
                },
            }

    return normal_vectors[wall_idt][dim]

def evaluate_normal_load(F, P, CG, mesh, ds, wall_idt):
    top_dim = mesh.topology().dim()
    
    normal_vector = _get_normal_vector(wall_idt, top_dim)
    return evaluate_load(F, P, CG, mesh, ds, wall_idt, normal_vector, normal_vector)


def evaluate_shear_load(F, P, CG, mesh, ds, wall_idt, direction):    
    top_dim = mesh.topology().dim()

    unit_vectors = {
            "xdir": {
                2 : df.as_vector([1.0, 0.0]),
                3 : df.as_vector([1.0, 0.0, 0.0]),
                },
            "ydir": {
                2 : df.as_vector([0.0, 1.0]),
                3 : df.as_vector([0.0, 1.0, 0.0]),
                },
            "zdir": {
                3 : df.as_vector([0.0, 0.0, 1.0]),
                },
            }
    
    unit_vector = unit_vectors[direction][top_dim]
    normal_vector = _get_normal_vector(wall_idt, top_dim)

    return evaluate_load(F, P, CG, mesh, ds, wall_idt, normal_vector, unit_vector)


def evaluate_load(F, P, CG, mesh, ds, wall_idt, normal_vector, unit_vector):
    load = df.project(df.inner(P * normal_vector, unit_vector), CG)

    total_load = df.assemble(load * ds(wall_idt))

    area = df.assemble(         # length in 2D
        df.det(F)
        * df.inner(df.inv(F).T * unit_vector, unit_vector)
        * ds(wall_idt)
    )

    return total_load/area


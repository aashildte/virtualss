
import dolfin as df
from mpi4py import MPI


def external_pressure_term(external_pressure_fun, F, v, mesh, ds):
    facet_norm = df.FacetNormal(mesh)
    return external_pressure_fun* df.inner(v, df.det(F) * df.inv(F) * facet_norm) * ds


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


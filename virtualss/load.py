
import dolfin as df
from mpi4py import MPI

from virtualss import get_length, get_width, get_height


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


def evaluate_deformation_xdir(u, mesh, boundary_markers, ds):
    """

    Evaluates relative shortening in the x direction, as taken as the average
    displacement at the "xmax" wall minus the displacement at the "xmin" wall,
    divided by the original domain length.

    Multiply result by 100 to get the shortening/widening in %.

    Args:
        u - displacement
        mesh - domain
        boundary_markers - dictionary defining subdomains and wall identities,
            as defined for a cubical domain
        ds - corresponding meshfunction

    Returns:
        float: relative shortening/widening

    """
    
    top_dim = mesh.topology().dim()
    if top_dim == 2:
        dir_vector = df.as_vector([1., 0.])
    elif top_dim == 3:
        dir_vector = df.as_vector([1., 0., 0.])
    else:
        raise NotImplementedError()

    xcomp = df.inner(u, dir_vector)

    xmin_idt = boundary_markers["xmin"]["idt"]
    xmax_idt = boundary_markers["xmax"]["idt"]

    disp_min = df.assemble(xcomp*ds(xmin_idt))
    disp_max = df.assemble(xcomp*ds(xmax_idt))

    length = get_length(mesh)
    relative_deformation_change = (disp_max - disp_min)/length

    return relative_deformation_change


def evaluate_deformation_ydir(u, mesh, boundary_markers, ds):
    """

    Evaluates relative shortening in the y direction, as taken as the average
    displacement at the "ymax" wall minus the displacement at the "ymin" wall,
    divided by the original domain width.

    Multiply result by 100 to get the shortening/widening in %.

    Args:
        u - displacement
        mesh - domain
        boundary_markers - dictionary defining subdomains and wall identities,
            as defined for a cubical domain
        ds - corresponding meshfunction

    Returns:
        float: relative shortening/widening

    """
 
    top_dim = mesh.topology().dim()
    if top_dim == 2:
        dir_vector = df.as_vector([0., 1.])
    elif top_dim == 3:
        dir_vector = df.as_vector([0., 1., 0.])
    else:
        raise NotImplementedError()

    ycomp = df.inner(u, dir_vector)

    ymin_idt = boundary_markers["ymin"]["idt"]
    ymax_idt = boundary_markers["ymax"]["idt"]

    disp_min = df.assemble(ycomp*ds(ymin_idt))
    disp_max = df.assemble(ycomp*ds(ymax_idt))

    width = get_width(mesh)
    relative_deformation_change = (disp_max - disp_min)/width

    return relative_deformation_change

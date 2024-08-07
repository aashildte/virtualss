
import dolfin as df
from mpi4py import MPI

from virtualss import get_length, get_width, get_height


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


def evaluate_deformation_xmax(u, mesh, boundary_markers, ds):
    """

    Evaluates relative shortening in the x direction, as taken as the average
    displacement at the "xmax" wall divided by the original domain length.

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

    disp_max = df.assemble(xcomp*ds(xmax_idt))

    length = get_length(mesh)
    relative_deformation_change = disp_max/length

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


def evaluate_deformation_ymax(u, mesh, boundary_markers, ds):
    """

    Evaluates relative shortening in the y direction, as taken as the average
    displacement at the "ymax" wall divided by the original domain width.

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

    ymax_idt = boundary_markers["ymax"]["idt"]
    disp_max = df.assemble(ycomp*ds(ymax_idt))

    width = get_width(mesh)
    relative_deformation_change = disp_max/width

    return relative_deformation_change


def evaluate_deformation_zdir(u, mesh, boundary_markers, ds):
    """

    Evaluates relative shortening in the y direction, as taken as the average
    displacement at the "zmax" wall minus the displacement at the "zmin" wall,
    divided by the original domain height.

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
    
    assert top_dim == 3, "Error: Stretch in the z direction only makes sense in 3D"

    dir_vector = df.as_vector([0., 0., 1.])

    zcomp = df.inner(u, dir_vector)

    zmin_idt = boundary_markers["zmin"]["idt"]
    zmax_idt = boundary_markers["zmax"]["idt"]

    disp_min = df.assemble(zcomp*ds(zmin_idt))
    disp_max = df.assemble(zcomp*ds(zmax_idt))

    height = get_height(mesh)
    relative_deformation_change = (disp_max - disp_min)/height

    return relative_deformation_change


def evaluate_deformation_zmax(u, mesh, boundary_markers, ds):
    """

    Evaluates relative shortening in the y direction, as taken as the average
    displacement at the "zmax" wall divided by the original domain height.

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
    
    assert top_dim == 3, "Error: Stretch in the z direction only makes sense in 3D"

    dir_vector = df.as_vector([0., 0., 1.])

    zcomp = df.inner(u, dir_vector)

    zmax_idt = boundary_markers["zmax"]["idt"]
    disp_max = df.assemble(zcomp*ds(zmax_idt))

    height = get_height(mesh)
    relative_deformation_change = disp_max/height

    return relative_deformation_change

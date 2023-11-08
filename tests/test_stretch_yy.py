"""

Unit tests for the code in the "stretch_yy" file, testing stretch and load
functions by assessing whether resulting stretch/load matches applied values.

Åshild Telle / University of Washington / 2023

"""

import dolfin as df
import numpy as np
import pytest

from virtualss import (
    CardiacModel,
    get_boundary_markers,
    stretch_yy_fixed_sides,
    stretch_yy_comp,
    stretch_yy_load,
    evaluate_normal_load,
    evaluate_deformation_ydir,
)


@pytest.mark.parametrize("dim", [2, 3])
def test_stretch_yy_fixed_sides(dim):
    N = 3

    if dim == 2:
        mesh = df.UnitSquareMesh(N, N)
    else:
        mesh = df.UnitCubeMesh(N, N, N)

    cm = CardiacModel(mesh)

    boundary_markers, ds = get_boundary_markers(mesh)
    bcs, bcsfun = stretch_yy_fixed_sides(cm.V, boundary_markers)

    assigned_stretch = 0.01
    bcsfun.k = assigned_stretch

    cm.solve(bcs)

    evaluated_stretch = evaluate_deformation_ydir(cm.u, mesh, boundary_markers, ds)

    state = cm.state
    u, _ = state.split()

    assert np.isclose(
        assigned_stretch, evaluated_stretch
    ), "Error: Assigned stretch != resulting stretch"


@pytest.mark.parametrize("dim", [2, 3])
def test_stretch_yy_ycomp(dim):
    N = 3

    if dim == 2:
        mesh = df.UnitSquareMesh(N, N)
    else:
        mesh = df.UnitCubeMesh(N, N, N)

    cm = CardiacModel(mesh)

    boundary_markers, ds = get_boundary_markers(mesh)
    bcs, bcsfun = stretch_yy_comp(cm.V, boundary_markers)

    assigned_stretch = 0.01
    bcsfun.k = assigned_stretch

    cm.solve(bcs)

    evaluated_stretch = evaluate_deformation_ydir(cm.u, mesh, boundary_markers, ds)

    assert np.isclose(
        assigned_stretch, evaluated_stretch
    ), "Error: Assigned stretch != resulting stretch"


@pytest.mark.parametrize("dim", [2, 3])
def test_stretch_yy_load(dim):
    N = 3

    if dim == 2:
        mesh = df.UnitSquareMesh(N, N)
    else:
        mesh = df.UnitCubeMesh(N, N, N)

    cm = CardiacModel(mesh, remove_rm=True)

    boundary_markers, ds = get_boundary_markers(mesh)
    ext_pressure_terms, ext_pressure_fun = stretch_yy_load(
        cm.F, cm.v, mesh, boundary_markers, ds
    )

    cm.weak_form += sum(ext_pressure_terms)

    assigned_load = 0.1
    ext_pressure_fun.k = assigned_load

    cm.solve()

    wall_idt = boundary_markers["ymax"]["idt"]
    evaluated_load = evaluate_normal_load(cm.F, cm.P, mesh, ds, wall_idt)

    assert np.isclose(
        assigned_load, evaluated_load
    ), "Error: Assigned load != resulting load"

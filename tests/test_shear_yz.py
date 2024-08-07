"""

Unit tests for the code in the "shear_yz" file, testing shear functions
by assessing whether resulting shear/load matches applied values.

Åshild Telle / University of Washington / 2024

"""

import dolfin as df
import numpy as np
import pytest

from virtualss import (
    CardiacModel,
    get_boundary_markers,
    simple_shear_yz,
    evaluate_normal_load,
    evaluate_deformation_ymax,
    evaluate_deformation_zmax,
)


def test_simple_shear_yz():
    N = 3
    mesh = df.UnitCubeMesh(N, N, N)

    cm = CardiacModel(mesh)

    boundary_markers, ds = get_boundary_markers(mesh)
    bcs, bcsfun = simple_shear_yz(cm.V, boundary_markers)

    assigned_shear = 0.05
    bcsfun.k = assigned_shear

    cm.solve(bcs)

    shear_ydir = evaluate_deformation_ymax(cm.u, mesh, boundary_markers, ds)
    shear_zdir = evaluate_deformation_zmax(cm.u, mesh, boundary_markers, ds)
 
    assert np.isclose(
        assigned_shear / 2, shear_zdir, rtol=0.05
    ), f"Error: Assigned shear: {assigned_shear / 2} != resulting shear: {shear_ydir}"

    assert np.isclose(
        shear_ydir, 0.0
    ), "Error: Domain not fixed in perpendicular direction"

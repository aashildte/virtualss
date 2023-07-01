"""

Script for stretching the domain by applying a load.

Åshild Telle / University of Washington / 2023–2022

"""

import matplotlib.pyplot as plt
import numpy as np
import dolfin as df
from mpi4py import MPI

from virtualss import CardiacModel, get_boundary_markers, stretch_ff_load_2D, stretch_ff_load_3D, evaluate_stretch

mesh_2D = df.UnitSquareMesh(3, 3)
mesh_3D = df.UnitCubeMesh(3, 3, 3)

for mesh, num_free_degrees, stretch_fun in zip(
    [mesh_2D, mesh_3D], [3, 6], [stretch_ff_load_2D, stretch_ff_load_3D]
):

    cm = CardiacModel(mesh, num_free_degrees)

    state, test_state, F = cm.state, cm.test_state, cm.F
    u, _, _ = state.split()

    # define weak form terms
    boundary_markers, ds = get_boundary_markers(mesh)
    Gext, pressure_fun, rm = stretch_fun(state, test_state, F, mesh, boundary_markers, ds)

    # add to weak form
    cm.weak_form += Gext + rm

    # iterate over these values:
    load_values = np.linspace(0, 5, 10)
    stretch_values = []

    # solve problem
    for l in load_values:
        print(f"Applied load: {l} kPa")
        pressure_fun.k = l

        cm.solve()

        stretch = evaluate_stretch(u, mesh, ds)
        stretch_values.append(100*stretch)
        print(stretch)

    plt.plot(stretch_values, load_values)
plt.show()

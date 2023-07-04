"""

Script for stretching the domain by applying a load.

Åshild Telle / University of Washington / 2023–2022

"""

import matplotlib.pyplot as plt
import numpy as np
import dolfin as df
from mpi4py import MPI

from virtualss import (
    CardiacModel,
    evaluate_stretch,
    get_boundary_markers,
    get_length,
    stretch_ff_xcomp_2D,
    stretch_ff_xcomp_3D,
    evaluate_normal_load,
)

# define mesh and cardiac mechanics
mesh_2D = df.UnitSquareMesh(3, 3)
mesh_3D = df.UnitCubeMesh(3, 3, 3)


for mesh, stretch_fun in zip(
    [mesh_2D, mesh_3D], [stretch_ff_xcomp_2D, stretch_ff_xcomp_3D]
):

    num_free_degrees = 0
    cm = CardiacModel(mesh, num_free_degrees)

    state, test_state, F = cm.state, cm.test_state, cm.F
    u, _ = state.split()
    V = cm.state_space.sub(0)

    # boundary markers
    L = get_length(mesh)
    boundary_markers, ds = get_boundary_markers(mesh)

    # define weak form terms
    bcs, bcsfun = stretch_fun(L, V, boundary_markers, mesh)

    # iterate over these values:
    stretch_values = np.linspace(0, 0.122, 10)
    load_values = []

    cm.solve(bcs=bcs)

    # solve problem
    for i, s in enumerate(stretch_values):
        print(f"Applied stretch: {100*s} %")
        bcsfun.k = s

        cm.solve(bcs=bcs)

        load = evaluate_normal_load(
            cm.F, cm.P, mesh, ds, 1
        ) + evaluate_normal_load(cm.F, cm.P, mesh, ds, 2)
        load_values.append(load)


    plt.plot(100*np.array(stretch_values), load_values)

plt.xlabel("Stretch (%)")
plt.ylabel("Load (kPa)")
plt.show()

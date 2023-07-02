"""

This script generates four different meshes of different shapes, verifying
that the stress-strain curve (tracked load vs. displacement) is shape independent
using "noslip" boundary conditions.

Åshild Telle / University of Washington, Simula Research Laboratory / 2023–2022

"""

import matplotlib.pyplot as plt
import numpy as np
import dolfin as df
from mpi4py import MPI

from virtualss import CardiacModel, get_boundary_markers, get_length, stretch_ff_componentwise_2D, evaluate_normal_load

# define mesh and cardiac mechanics
N = 10
mesh1 = df.UnitSquareMesh(N, N)
mesh2 = df.RectangleMesh(df.Point(0, 0), df.Point(2, 1), 2*N, N)
mesh3 = df.RectangleMesh(df.Point(0, 0), df.Point(1, 2), N, N)

markers = ["D", "o", "X"]

for mesh, marker in zip([mesh1, mesh2, mesh3], markers):
    cm = CardiacModel(mesh)
    V, P, F, state = cm.V, cm.P, cm.F, cm.state

    # deformation of choice
    deformation_mode = "stretch_ff"
    fixed_sides = "componentwise"

    boundary_markers, ds = get_boundary_markers(mesh)
    length = get_length(mesh)
    bcs, bcsfun = stretch_ff_componentwise_2D(length, V, boundary_markers)

    wall_idt = 2     # max_x

    normal_load = []

    # iterate over these values:
    stretch_values = np.linspace(0, 0.2, 10)

    # solve problem
    for s in stretch_values:
        print(f"Domain stretch: {100*s:.0f} %")
        bcsfun.k = s

        cm.solve(bcs)

        load = evaluate_normal_load(F, P, mesh, ds, wall_idt)
        normal_load.append(load)

    plt.plot(100*stretch_values, normal_load, marker=marker)

plt.xlabel("Stretch (%)")
plt.ylabel("Load (kPa)")

plt.legend(["Unit square", "x2 length in xdim", "x2 length in ydim"])
plt.tight_layout()
plt.savefig(f"2D_{fixed_sides}.png", dpi=300)
plt.show()

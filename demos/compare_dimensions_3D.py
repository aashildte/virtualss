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

from virtualss import CardiacModel, define_boundary_conditions, evaluate_normal_load

# define mesh and cardiac mechanics
N = 5
mesh1 = df.UnitCubeMesh(N, N, N)
mesh2 = df.BoxMesh(df.Point(0, 0, 0), df.Point(2, 1, 1), 2*N, N, N)
mesh3 = df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 2, 1), N, 2*N, N)
mesh4 = df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 2), N, N, 2*N)

markers = ["D", "o", "X", 6]

for mesh, marker in zip([mesh1, mesh2, mesh3, mesh4], markers):
    cm = CardiacModel(mesh)
    V, P, F = cm.V, cm.P, cm.F

    # deformation of choice
    deformation_mode = "stretch_ff"
    fixed_sides = "componentwise"
    bcs, bc_fun, ds = define_boundary_conditions(deformation_mode, fixed_sides, mesh, V)
    wall_idt = 2     # max_x

    normal_load = []

    # iterate over these values:
    stretch_values = np.linspace(0, 0.2, 10)

    # solve problem
    for s in stretch_values:
        print(f"Domain stretch: {100*s:.0f} %")
        bc_fun.k = s

        cm.solve(bcs)

        load = evaluate_normal_load(F, P, mesh, ds, wall_idt)
        normal_load.append(load)
    
    plt.plot(100*stretch_values, normal_load, marker=marker)

    print("her??")

plt.xlabel("Stretch (%)")
plt.ylabel("Load (kPa)")

plt.legend(["Unit cube", "x2 length in xdim", "x2 length in ydim", "x2 lengt in zdim"])
plt.tight_layout()
plt.savefig(f"3D_{fixed_sides}.png", dpi=300)
plt.show()

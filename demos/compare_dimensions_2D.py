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

from virtualss import define_weak_form, define_boundary_conditions, evaluate_normal_load

# define mesh and cardiac mechanics
mesh1 = df.UnitSquareMesh(3, 3)
mesh2 = df.RectangleMesh(df.Point(0, 0), df.Point(2, 1), 6, 3)
mesh3 = df.RectangleMesh(df.Point(0, 0), df.Point(1, 2), 3, 6)

for mesh in [mesh1, mesh2, mesh3]:
    weak_form, state, V, P, F = define_weak_form(mesh)

    # deformation of choice
    deformation_mode = "stretch_ff"
    fixed_sides = "noslip"
    bcs, bc_fun, ds = define_boundary_conditions(deformation_mode, fixed_sides, mesh, V)
    wall_idt = 2     # max_x

    normal_load = []

    # iterate over these values:
    stretch_values = np.linspace(0, 0.2, 10)

    # solve problem
    for s in stretch_values:
        print(f"Domain stretch: {100*s:.0f} %")
        bc_fun.k = s

        df.solve(weak_form == 0, state, bcs=bcs)
        u, _ = state.split()

        load = evaluate_normal_load(F, P, mesh, ds, wall_idt)
        normal_load.append(load)

    plt.plot(normal_load)

plt.legend(["Unit cube", "x2 in xdim", "x2 in ydim", "x2 in zdim"])
plt.show()

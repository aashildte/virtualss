"""

This script generates four different meshes of different shapes, verifying
that the stress-strain curve (tracked load vs. displacement) is shape independent
and dimension independent using the "xx_xcomp" stretch function. The same holds
true for "yy_ycomp" and "zz_zcomp" (feel free to try!).

Åshild Telle / University of Washington, Simula Research Laboratory / 2023–2022

"""

import matplotlib.pyplot as plt
import numpy as np
import dolfin as df
from mpi4py import MPI

from virtualss import CardiacModel, get_boundary_markers, get_length, stretch_xx_xcomp, evaluate_normal_load

# define meshes in various dimensions and of varying size along different directions
N = 3

meshes = [
    df.UnitSquareMesh(N, N),
    df.RectangleMesh(df.Point(0, 0), df.Point(2, 1), 2*N, N),
    df.RectangleMesh(df.Point(0, 0), df.Point(1, 2), N, N),
    df.UnitCubeMesh(N, N, N),
    df.BoxMesh(df.Point(0, 0, 0), df.Point(2, 1, 1), 2*N, N, N),
    df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 2, 1), N, 2*N, N),
    df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 2), N, N, 2*N),
]

for mesh in meshes:
    cm = CardiacModel(mesh)
    V, P, F, state = cm.V, cm.P, cm.F, cm.state

    # deformation of choice
    deformation_mode = "stretch_ff"
    fixed_sides = "componentwise"

    boundary_markers, ds = get_boundary_markers(mesh)
    bcs, bcsfun = stretch_xx_xcomp(V, boundary_markers)

    wall_idt = boundary_markers["xmax"]["idt"]

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

    plt.plot(100*stretch_values, normal_load)

plt.xlabel("Stretch (%)")
plt.ylabel("Load (kPa)")

plt.legend(["Unit square (2D)", "x2 length (2D)", "x2 width (2D)", "Unit cube (3D)", "x2 length (3D)", "x2 width (3D)", "x2 height (3D)"])
plt.tight_layout()
plt.show()

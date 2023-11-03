"""

Script for performing all stretch and shear experiments; plots shear and normal components.

Åshild Telle / University of Washington, Simula Research Laboratory / 2023–2022

"""


import numpy as np
import matplotlib.pyplot as plt
import dolfin as df
from mpi4py import MPI

from virtualss import (
    CardiacModel,
    evaluate_normal_load,
    evaluate_shear_load,
    get_boundary_markers,
    stretch_xx_fixed_sides,
    shear_xy_fixed_sides,
    shear_xz_fixed_sides,
    shear_yx_fixed_sides,
    stretch_xx_fixed_sides,
    shear_yz_fixed_sides,
    shear_zx_fixed_sides,
    shear_zy_fixed_sides,
    stretch_xx_fixed_sides,
)

# define mesh and cardiac mechanics
mesh = df.UnitCubeMesh(2, 2, 2)

deformation_modes_3D = [
    "stretch_ff",
    "shear_fs",
    "shear_fn",
    "shear_sf",
    "stretch_ss",
    "shear_sn",
    "shear_nf",
    "shear_ns",
    "stretch_ff",
]

deformation_funs = [
    stretch_xx_fixed_sides,
    shear_xy_fixed_sides,
    shear_xz_fixed_sides,
    shear_yx_fixed_sides,
    stretch_xx_fixed_sides,
    shear_yz_fixed_sides,
    shear_zx_fixed_sides,
    shear_zy_fixed_sides,
    stretch_xx_fixed_sides,
]

walls = ["xmax"]*3 + ["ymax"]*3 + ["zmax"]*3
directions = ["xdir", "ydir", "zdir"]*3 

fig, ax = plt.subplots(3, 3, sharey=True, sharex=True)
axes = ax.flatten()

for deformation_mode, deformation_fun, axis, wall, direction in zip(
    deformation_modes_3D, deformation_funs, axes, walls, directions
):
    cm = CardiacModel(mesh, 0)

    # extract variables needed for boundary conditions + for evaluation/tracking
    V, P, F = cm.V, cm.P, cm.F

    boundary_markers, ds = get_boundary_markers(mesh)
    bcs, bc_fun = deformation_fun(V, boundary_markers)
    wall_idt = boundary_markers[wall]["idt"]

    # iterate over these values:
    if "stretch" in deformation_mode:
        stretch_values = np.linspace(0, 0.1, 20)
    else:
        stretch_values = np.linspace(0, 0.4, 20)

    normal_load = []
    shear_load = []

    # solve problem
    for s in stretch_values:
        print(f"Domain stretch: {100*s:.0f} %")
        bc_fun.k = s

        cm.solve(bcs)

        load_n = evaluate_normal_load(F, P, mesh, ds, wall_idt)
        load_s = evaluate_shear_load(F, P, mesh, ds, wall_idt, direction)
        normal_load.append(load_n)
        shear_load.append(load_s)

    axis.plot(100 * stretch_values, normal_load, label="Normal load")
    if "shear" in deformation_mode:
        axis.plot(100 * stretch_values, shear_load, label="Shear load")

    axis.set_xlabel("Stretch (%)")
    axis.set_ylabel("Load (kPa)")
    axis.legend()

plt.show()

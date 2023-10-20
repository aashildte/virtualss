"""

Example scripts for running a stretch experiments along the x direction in
which both sides are fixed in area (one side is completely fixed at 0, the
other side is assigned an incremental x component while we assign y = z = 0).

Åshild Telle / University of Washington / 2023

"""

import matplotlib.pyplot as plt
import numpy as np
import dolfin as df
from mpi4py import MPI

from virtualss import (
    CardiacModel,
    get_boundary_markers,
    evaluate_normal_load,
    shear_xy_fixed_sides,
)

# define mesh and initiate instance of class from which we get the weak form
N = 4
mesh = df.UnitCubeMesh(N, N, N)

cm = CardiacModel(mesh, 0)

# extract variables needed for boundary conditions + for evaluation/tracking
V, P, F, state = cm.V, cm.P, cm.F, cm.state
u, _ = state.split()

# define boundary conditions for our deformation of choice
boundary_markers, ds = get_boundary_markers(mesh)
bcs, bc_fun = shear_xy_fixed_sides(V, boundary_markers)
wall_idt = boundary_markers["xmax"]["idt"]

# track displacement and save to file + save load values
fout = df.XDMFFile(MPI.COMM_WORLD, "displacement3D_shear_fixed_sides.xdmf")

# iterate over these values:
stretch_values = np.linspace(0, 0.2, 20)

# track load values each stretch value
normal_load = []

# solve problem
for stretch in stretch_values:
    print(f"Domain shear: {100*stretch:.0f} %")
    bc_fun.k = stretch

    cm.solve(bcs)

    load = evaluate_normal_load(F, P, mesh, ds, wall_idt)
    normal_load.append(load)
    fout.write_checkpoint(u, "Displacement (µm)", stretch, append=True)

fout.close()

# finally plot the resulting stretch/stress curve
plt.plot(100 * stretch_values, normal_load)
plt.xlabel("Stretch (%)")
plt.ylabel("Load (kPa)")
plt.show()

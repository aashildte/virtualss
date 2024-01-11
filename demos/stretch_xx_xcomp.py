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
    get_boundary_markers,
    stretch_xx_comp,
    evaluate_normal_load,
)

# define mesh and cardiac mechanics
N = 10
#mesh = df.UnitCubeMesh(3, 3, 3)
mesh = df.UnitSquareMesh(N, N)
cm = CardiacModel(mesh)

# extract variables needed for boundary conditions + for evaluation/tracking
V, P, F, state, PK1 = cm.V, cm.P, cm.F, cm.state, cm.P
u, _ = state.split()
T = df.TensorFunctionSpace(mesh, "CG", 2)

V = cm.state_space.sub(0)

# boundary markers
boundary_markers, ds = get_boundary_markers(mesh)
wall_idt = boundary_markers["xmax"]["idt"]

# define weak form terms
bcs, bcsfun = stretch_xx_comp(V, boundary_markers)

# iterate over these values:
stretch_values = np.linspace(0, 0.122, 10)
load_values = []

cm.solve(bcs=bcs)

# track displacement and save to file + save load values
fout_disp = df.XDMFFile(MPI.COMM_WORLD, "displacement_stretch_2D.xdmf")
fout_PK1 = df.XDMFFile(MPI.COMM_WORLD, "PK1_stretch_2D.xdmf")

# solve problem
for i, s in enumerate(stretch_values):
    print(f"Applied stretch: {100*s:.2f} %")
    bcsfun.k = s

    cm.solve(bcs=bcs)

    load = evaluate_normal_load(cm.F, cm.P, mesh, ds, wall_idt)
    load_values.append(load)

    fout_disp.write_checkpoint(u, "Displacement (µm)", s, append=True)
    fout_PK1.write_checkpoint(df.project(PK1, T), "Piola-Kirchhoff stress (kPa)", s, append=True)
    
fout_disp.close()
fout_PK1.close()


plt.plot(100*np.array(stretch_values), load_values)

plt.xlabel("Stretch (%)")
plt.ylabel("Load (kPa)")
plt.show()

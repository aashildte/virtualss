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
    stretch_xx_xcomp,
    evaluate_normal_load,
)

# define mesh and cardiac mechanics
mesh = df.UnitCubeMesh(3, 3, 3)
cm = CardiacModel(mesh)

state, test_state, F = cm.state, cm.test_state, cm.F
u, _ = state.split()
V = cm.state_space.sub(0)

# boundary markers
boundary_markers, ds = get_boundary_markers(mesh)
wall_idt = boundary_markers["xmax"]["idt"]

# define weak form terms
bcs, bcsfun = stretch_xx_xcomp(V, boundary_markers)

# iterate over these values:
stretch_values = np.linspace(0, 0.122, 10)
load_values = []

cm.solve(bcs=bcs)

# solve problem
for i, s in enumerate(stretch_values):
    print(f"Applied stretch: {100*s:.2f} %")
    bcsfun.k = s

    cm.solve(bcs=bcs)

    load = evaluate_normal_load(cm.F, cm.P, mesh, ds, wall_idt)
    load_values.append(load)


plt.plot(100*np.array(stretch_values), load_values)

plt.xlabel("Stretch (%)")
plt.ylabel("Load (kPa)")
plt.show()

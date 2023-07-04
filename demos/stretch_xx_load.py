"""

Script for stretching the domain by applying a load.

Åshild Telle / University of Washington / 2023–2022

"""

import matplotlib.pyplot as plt
import numpy as np
import dolfin as df
from mpi4py import MPI

from virtualss import CardiacModel, get_boundary_markers, stretch_xx_load, evaluate_deformation_xdir

mesh = df.UnitCubeMesh(3, 3, 3)

cm = CardiacModel(mesh, remove_rm=True)

# define weak form terms
boundary_markers, ds = get_boundary_markers(mesh)
external_pressure_terms, pressure_fun = stretch_xx_load(cm.F, cm.v, mesh, boundary_markers, ds)

# add to weak form
cm.weak_form += sum(external_pressure_terms)

# iterate over these values:
load_values = np.linspace(0, 2.5, 20)
stretch_values = []

# solve problem
for l in load_values:
    print(f"Applied load: {l:.2f} kPa")
    pressure_fun.k = l

    cm.solve()

    stretch = evaluate_deformation_xdir(cm.u, mesh, boundary_markers, ds)
    stretch_values.append(100*stretch)

plt.plot(stretch_values, load_values)
plt.xlabel("Stretch (%)")
plt.ylabel("Load (kPa)")
plt.show()

"""

Example scripts for running a stretch experiments along the x direction in
which both sides are fixed in area (one side is completely fixed at 0, the
other side is assigned an incremental x component while we assign y = z = 0).

Åshild Telle / University of Washington / 2023

Suggested ways to modify this script:

- You might replace
      share_xy_comp          and         "xmax"       and       "ydir"
  with 
      shear_yx_comp          and         "ymax"       and       "xdir"

- You might make this 3D simply by changing the geometry:
      df.UnitSquareMesh(N, N)
  with
      df.UnitCubeMesh(N, N, N)
  and in 3D all combinations with z should work as well, i.e., combinations
      share_xz_comp          and         "xmax"
      share_zx_comp          and         "zmax"
      share_yz_comp          and         "ymax"
      share_zy_comp          and         "zmax"

"""

import matplotlib.pyplot as plt
import numpy as np
import dolfin as df
from mpi4py import MPI

from virtualss import (
    CardiacModel,
    get_boundary_markers,
    evaluate_normal_load,
    evaluate_shear_load,
)
from virtualss import simple_shear_xy as shear_fn
wall = "xmax"
direction = "ydir"

# define mesh and initiate instance of class from which we get the weak form
N = 5
#mesh = df.UnitCubeMesh(N, N, N)
mesh = df.UnitSquareMesh(N, N)

cm = CardiacModel(mesh, 0)

# extract variables needed for boundary conditions + for evaluation/tracking
V, P, F, state, PK1 = cm.V, cm.P, cm.F, cm.state, cm.P
u, _ = state.split()
T = df.TensorFunctionSpace(mesh, "CG", 2)

# define boundary conditions for our deformation of choice
boundary_markers, ds = get_boundary_markers(mesh)
bcs, bc_fun = shear_fn(V, boundary_markers)
wall_idt = boundary_markers[wall]["idt"]

# track displacement and save to file + save load values
fout_disp = df.XDMFFile(MPI.COMM_WORLD, "displacement_shear_2D_alternative.xdmf")
fout_PK1 = df.XDMFFile(MPI.COMM_WORLD, "PK1_shear_2D_alternative.xdmf")

# iterate over these values:
stretch_values = np.linspace(0, 0.2, 20)

# track load values each stretch value
normal_load = []
shear_load = []

# solve problem
for stretch in stretch_values:
    print(f"Domain shear: {100*stretch:.0f}%")
    #bc_fun.a = stretch
    bc_fun.k = stretch

    cm.solve(bcs)

    load_n = evaluate_normal_load(F, P, mesh, ds, wall_idt)
    load_s = evaluate_shear_load(F, P, mesh, ds, wall_idt, direction)
    normal_load.append(load_n)
    shear_load.append(load_s)
    fout_disp.write_checkpoint(u, "Displacement (µm)", stretch, append=True)
    fout_PK1.write_checkpoint(df.project(PK1, T), "Piola-Kirchhoff stress (kPa)", stretch, append=True)
    
fout_disp.close()
fout_PK1.close()

# finally plot the resulting stretch/stress curve
plt.plot(100 * stretch_values, normal_load)
plt.plot(100 * stretch_values, shear_load)
plt.xlabel("Stretch (%)")
plt.ylabel("Load (kPa)")
plt.tight_layout()
plt.show()

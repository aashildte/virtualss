"""

Script for manual tuning of parameters a and a_f in the simplified
Holzapfel model to match changes in stress-strain curves.

Åshild Telle / University of Washington / 2023

"""

import matplotlib.pyplot as plt
import numpy as np
import dolfin as df

from virtualss import (
    CardiacModel,
    get_boundary_markers,
    stretch_xx_comp,
    stretch_yy_comp,
    evaluate_normal_load,
)


def fiber_dir_stretch(mesh, material_parameters, stretch_goal):
    cm = CardiacModel(mesh, material_parameters=material_parameters)
    state, test_state, F = cm.state, cm.test_state, cm.F
    u, _ = state.split()
    V = cm.state_space.sub(0)

    # boundary markers
    boundary_markers, ds = get_boundary_markers(mesh)
    wall_idt = boundary_markers["xmax"]["idt"]

    # define weak form terms
    bcs, bcsfun = stretch_xx_comp(V, boundary_markers)

    # iterate over these values:
    stretch_values = np.linspace(0, stretch_goal, 20)
    load_values = []

    cm.solve(bcs=bcs)

    # solve problem
    for i, s in enumerate(stretch_values):
        print(f"Applied stretch: {100*s:.2f} %")
        bcsfun.k = s

        cm.solve(bcs=bcs)

        load = evaluate_normal_load(cm.F, cm.P, mesh, ds, wall_idt)
        load_values.append(load)

    return stretch_values, load_values


def sheet_dir_stretch(mesh, material_parameters, stretch_goal):
    cm = CardiacModel(mesh, material_parameters=material_parameters)
    state, test_state, F = cm.state, cm.test_state, cm.F
    u, _ = state.split()
    V = cm.state_space.sub(0)

    # boundary markers
    boundary_markers, ds = get_boundary_markers(mesh)
    wall_idt = boundary_markers["ymax"]["idt"]

    # define weak form terms
    bcs, bcsfun = stretch_yy_comp(V, boundary_markers)

    # iterate over these values:
    stretch_values = np.linspace(0, stretch_goal, 20)
    load_values = []

    cm.solve(bcs=bcs)

    # solve problem
    for i, s in enumerate(stretch_values):
        print(f"Applied stretch: {100*s:.2f} %")
        bcsfun.k = s

        cm.solve(bcs=bcs)

        load = evaluate_normal_load(cm.F, cm.P, mesh, ds, wall_idt)
        load_values.append(load)

    return stretch_values, load_values


# set parameters for either fiber or transverse direction stretch here!
fiber_dir = 1  # 0 for sheet/transverse direction stretch

if fiber_dir == 1:
    a_scaling = [1.0, 1.0, 1.0]
    af_scaling = [0.715, 1.0, 1.285]  # values found by manual adjustmnet
    alpha_fiber = 1
    alpha_transv = 0.2
    filename = "fiber_direction_stretch_pertubration.png"
else:
    a_scaling = [0.75, 1.0, 1.25]  # values found by manual adjustment
    af_scaling = [1.0355, 1.0, 0.9645]  # values found by manual adjustment
    alpha_fiber = 0.2
    alpha_transv = 1
    filename = "transverse_direction_stretch_pertubration.png"


# define mesh; perform stretch experiments; track load values and print difference
mesh = df.UnitCubeMesh(2, 2, 2)

stretch_goal = 0.05  # 5%
fiber_loads_final = []
sheet_loads_final = []

colors = ["tab:blue", "darkgray", "tab:red"]

for sv1, sv2, color in zip(a_scaling, af_scaling, colors):
    material_parameters = {"a": sv1 * 2.92, "b": 5.60, "a_f": sv2 * 11.84, "b_f": 17.95}
    print(material_parameters)

    # fiber dir
    stretch_values, load_values = fiber_dir_stretch(
        mesh, material_parameters, stretch_goal
    )
    fiber_loads_final.append(load_values[-1])

    if color == "tab:blue":
        plt.plot(
            100 * np.array(stretch_values),
            load_values,
            "--",
            color=color,
            alpha=alpha_fiber,
        )
    elif color == "tab:red":
        plt.plot(
            100 * np.array(stretch_values),
            load_values,
            marker="+",
            color=color,
            alpha=alpha_fiber,
        )
    else:
        plt.plot(
            100 * np.array(stretch_values), load_values, color=color, alpha=alpha_fiber
        )

    # sheet dir
    stretch_values, load_values = sheet_dir_stretch(
        mesh, material_parameters, stretch_goal
    )
    sheet_loads_final.append(load_values[-1])

    if color == "tab:blue":
        plt.plot(
            100 * np.array(stretch_values),
            load_values,
            "--",
            color=color,
            alpha=alpha_transv,
        )
    elif color == "tab:red":
        plt.plot(
            100 * np.array(stretch_values),
            load_values,
            marker="+",
            color=color,
            alpha=alpha_transv,
        )
    else:
        plt.plot(
            100 * np.array(stretch_values), load_values, color=color, alpha=alpha_transv
        )

plt.legend(
    [
        "25% decrease; fiber dir. stretch",
        "Baseline; fiber dir. stretch",
        "25% increase; fiber dir. stretch",
        "25% decrease; transv. dir. stretch",
        "Baseline; transv. dir. stretch",
        "25% increase; transv. dir. stretch",
    ]
)

l1, l2, l3 = fiber_loads_final
print("Load difference in fiber direction stretch: ")
print(
    f"* Decrease: {round((l2 - l1)/l2*100, 2)}%, increase: {round((l3 - l2)/l2*100,2)}%"
)

l1, l2, l3 = sheet_loads_final
print("Load difference in transverse direction stretch: ")
print(
    f"* Decrease: {round((l2 - l1)/l2*100, 2)}%, increase: {round((l3 - l2)/l2*100,2)}%"
)


plt.xlabel("Stretch (%)")
plt.ylabel("Load (kPa)")

plt.tight_layout()
plt.savefig(filename, dpi=300)

plt.show()

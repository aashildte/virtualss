import dolfin as df
import ufl
import numpy as np
import matplotlib.pyplot as plt

df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"
df.parameters["form_compiler"]["quadrature_degree"] = 4


def psi_holzapfel(
    F,
    a=0.074,
    b=4.878,
    a_f=2.628,
    b_f=5.214,
):
    dim = 3
    J = ufl.det(F)
    J_iso = pow(J, -float(1) / dim)
    C = J_iso ** 2 * F.T * F

    e1 = ufl.as_vector([1.0, 0.0, 0.0])

    IIFx = ufl.tr(C)
    I4e1 = ufl.inner(C * e1, e1)

    cond = lambda a: ufl.conditional(a > 0, a, 0)

    W_hat = a / (2 * b) * (ufl.exp(b * (IIFx - dim)) - 1)
    W_f = a_f / (2 * b_f) * (ufl.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

    return W_hat + W_f


class FreeBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class LeftBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < df.DOLFIN_EPS


class RightBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 1) < df.DOLFIN_EPS


def _evaluate_avg_disp(F, f, mesh, ds, wall_idt):
    normal_vector = df.FacetNormal(mesh)

    area = df.assemble(
        df.det(F) * df.inner(df.inv(F).T * normal_vector, normal_vector) * ds(wall_idt)
    )

    return df.assemble(f * ds(wall_idt)) #/ area


def evaluate_stretch(u, F, ds):
    e1 = df.as_vector([1.0, 0.0, 0.0])
    xcomp = df.inner(u, e1)

    disp_min = _evaluate_avg_disp(F, xcomp, mesh, ds, 1)
    disp_max = _evaluate_avg_disp(F, xcomp, mesh, ds, 2)

    return disp_max - disp_min


def load_approach(mesh, ds):
    P2 = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    R = df.VectorElement("Real", mesh.ufl_cell(), 0, 6)

    state_space = df.FunctionSpace(mesh, df.MixedElement([P2, P1, R]))
    V = state_space.sub(0)

    state = df.Function(state_space)
    test_state = df.TestFunction(state_space)

    u, p, r = df.split(state)
    v, q, _ = df.split(test_state)

    # Kinematics
    d = len(u)
    I = ufl.Identity(d)  # Identity tensor
    F = ufl.variable(I + ufl.grad(u))  # Deformation gradient
    J = ufl.det(F)

    psi = psi_holzapfel(F)
    P = df.diff(psi, F) + p * J * ufl.inv(F.T)

    u, p, r = df.split(state)
    v, q, _ = df.split(test_state)

    # load term
    N = df.FacetNormal(mesh)

    pressure_fun = df.Expression("-k/2", k=0, degree=2)
    Gext_left = pressure_fun * df.inner(v, df.det(F) * df.inv(F) * N) * ds(1)
    Gext_right = pressure_fun * df.inner(v, df.det(F) * df.inv(F) * N) * ds(2)

    Gext = Gext_left + Gext_right

    # rigid motion term
    

    position = df.SpatialCoordinate(mesh)
    
    RM = [
        df.Constant((1, 0, 0)),
        df.Constant((0, 1, 0)),
        df.Constant((0, 0, 1)),
        df.cross(position, df.Constant((1, 0, 0))),
        df.cross(position, df.Constant((0, 1, 0))),
        df.cross(position, df.Constant((0, 0, 1))),
    ]
    
    Pi = sum(df.dot(u, zi) * r[i] * df.dx for i, zi in enumerate(RM))
    rm = df.derivative(Pi, state, test_state)
    
    # elasticity + pressure terms

    elasticity_term = ufl.inner(P, ufl.grad(v)) * df.dx
    pressure_term = q * (J - 1) * df.dx

    weak_form = elasticity_term + pressure_term + rm + Gext

    # iterate until we reach 15 % stretch

    stretch_goal = 0.1
    stretch = 0
    load_increment = 0.4
    load = 0

    df.solve(weak_form == 0, state)

    load_values = []
    stretch_values = []

    while stretch < stretch_goal:
        load += load_increment
        print(f"Applied load: {load} kPa")
        load_values.append(load)
        pressure_fun.k = load

        df.solve(weak_form == 0, state)

        stretch = evaluate_stretch(u, F, ds)
        stretch_values.append(stretch)

    return load_values, stretch_values


def evaluate_load(F, P, mesh, ds, wall_idt):
    normal_vector = df.FacetNormal(mesh)

    load = df.inner(P * normal_vector, normal_vector)
    total_load = df.assemble(load * ds(wall_idt))
    area = df.assemble(
        df.det(F)
        * df.inner(df.inv(F).T * normal_vector, normal_vector)
        * ds(wall_idt)
    )

    return total_load/area



def displacement_approach(mesh, ds, left_b, right_b):
    P2 = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    R = df.VectorElement("Real", mesh.ufl_cell(), 0, 3)

    state_space = df.FunctionSpace(mesh, df.MixedElement([P2, P1, R]))
    V = state_space.sub(0)
    
    state = df.Function(state_space)
    test_state = df.TestFunction(state_space)

    u, p, r = df.split(state)
    v, q, _ = df.split(test_state)

    # Kinematics
    d = len(u)
    I = ufl.Identity(d)  # Identity tensor
    F = ufl.variable(I + ufl.grad(u))  # Deformation gradient
    J = ufl.det(F)

    psi = psi_holzapfel(F)
    P = df.diff(psi, F) + p * J * ufl.inv(F.T)
    sigma = (1 / df.det(F)) * P * F.T

    u, p, r = df.split(state)
    v, q, _ = df.split(test_state)

    # rigid motion term

    position = df.SpatialCoordinate(mesh)

    RM = [
        df.Constant((0, 1, 0)),
        df.Constant((0, 0, 1)),
        df.cross(position, df.Constant((1, 0, 0))),
    ]

    Pi = sum(df.dot(u, zi) * r[i] * df.dx for i, zi in enumerate(RM))

    rm = df.derivative(Pi, state, test_state)

    # elasticity + pressure terms

    elasticity_term = ufl.inner(P, ufl.grad(v)) * df.dx
    pressure_term = q * (J - 1) * df.dx

    weak_form = elasticity_term + pressure_term + rm

    # boundary conditions
    
    bcsfun_left = df.Expression("k", k=0, degree=1)
    bcsfun_right = df.Expression("k", k=0, degree=1)

    bcs = [
        df.DirichletBC(V.sub(0), bcsfun_left, left_b),
        df.DirichletBC(V.sub(0), bcsfun_right, right_b),
    ]


    # iterate until we reach 15 % stretch

    stretch_goal = 0.1
    stretch = 0
    stretch_increment = 0.01
    load = 0

    df.solve(weak_form == 0, state, bcs=bcs)

    load_values = []
    stretch_values = []

    while stretch < stretch_goal:
        stretch += stretch_increment
        print(f"Applied stretch: {stretch} kPa")
        stretch_values.append(stretch)
        bcsfun_right.k = stretch/2
        bcsfun_left.k = -stretch/2

        df.solve(weak_form == 0, state, bcs=bcs)

        assert abs(stretch - evaluate_stretch(u, F, ds)) < 0.0001, \
                f"Error: assigned stretch {stretch} != evaluated stretch {evaluate_stretch(u, F, ds)}"
        
        load_left = evaluate_load(F, P, mesh, ds, 1)
        load_right = evaluate_load(F, P, mesh, ds, 2)
        load = load_left + load_right

        load_values.append(load)
    
    return load_values, stretch_values


mesh = df.UnitCubeMesh(4, 4, 4)

# Initialize subdomain instances
free_b = FreeBoundary()
left_b = LeftBoundary()
right_b = RightBoundary()

# Initialize mesh function for boundary domains
boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
left_b.mark(boundaries, 1)
right_b.mark(boundaries, 2)

# Define new measures associated with the exterior boundaries
ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

load_D, stretch_D = displacement_approach(mesh, ds, left_b, right_b)
load_L, stretch_L = load_approach(mesh, ds)

plt.plot(stretch_L, load_L, "o-")
plt.plot(stretch_D, load_D, "o-")
plt.legend(["Load based", "Displacement based"])
plt.show()

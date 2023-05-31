"""

Functions for defining cardiac mechanics equations including
strain energy functions and weak form terms.

Åshild Telle / University of Washington, Simula Research Laboratory / 2023–2022

TODO maybe make into a class to assess attributes?

"""

import dolfin as df
import ufl


def psi_holzapfel(
    F,
    dim,
    a=0.074,
    b=4.878,
    a_f=2.628,
    b_f=5.214,
):
    """

    Declares the strain energy function for a simplified holzapfel formulation.

    Args:
        F - deformation tensor
        dim - number of dimensions; 2 or 3
        a - isotropic contribution (in kPa)
        b - isotropic exponential
        a_f - fiber direction contribution under stretch (in kPa)
        b_f - fiber direction exponential

    Returns:
        psi(F), scalar function

    """

    J = ufl.det(F)
    J_iso = pow(J, -float(1) / dim)
    C = J_iso ** 2 * F.T * F

    if dim == 2:
        e1 = ufl.as_vector([1.0, 0.0])
    else:
        e1 = ufl.as_vector([1.0, 0.0, 0.0])

    IIFx = ufl.tr(C)
    I4e1 = ufl.inner(C * e1, e1)

    cond = lambda a: ufl.conditional(a > 0, a, 0)

    W_hat = a / (2 * b) * (ufl.exp(b * (IIFx - 3)) - 1)
    W_f = a_f / (2 * b_f) * (ufl.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

    return W_hat + W_f


def define_weak_form(mesh, material_parameters={}):
    """

    Defines function spaces (P1 x P2) and functions to solve for, as well
    as the weak form for the problem itself. This assumes a fully incompressible
    formulation, solving for the displacement and the hydrostatic pressure.

    Args:
        mesh (df.Mesh): domain to solve equations over
        material_parameters - dictionary; use parameters as keys

    Returns:
        weak form (ufl form), state, displacement, boundary conditions
        stretch_fun (ufl form): function that assigns Dirichlet bcs
                on wall to be stretched/extended

    """

    df.parameters["form_compiler"]["cpp_optimize"] = True
    df.parameters["form_compiler"]["representation"] = "uflacs"
    df.parameters["form_compiler"]["quadrature_degree"] = 4

    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

    state_space = df.FunctionSpace(mesh, df.MixedElement([P2, P1]))
    V = state_space.sub(0)

    state = df.Function(state_space)
    test_state = df.TestFunction(state_space)

    u, p = df.split(state)
    v, q = df.split(test_state)

    # Kinematics
    d = len(u)
    I = ufl.Identity(d)  # Identity tensor
    F = ufl.variable(I + ufl.grad(u))  # Deformation gradient
    J = ufl.det(F)

    # Weak form

    metadata = {"quadrature_degree": 4}
    dx = ufl.Measure("dx", domain=mesh, metadata=metadata)

    dim = mesh.topology().dim()

    psi = psi_holzapfel(F, dim=dim, **material_parameters)
    P = ufl.diff(psi, F) + p * J * ufl.inv(F.T)

    elasticity_term = ufl.inner(P, ufl.grad(v)) * dx
    pressure_term = q * (J - 1) * dx

    weak_form = elasticity_term + pressure_term

    return weak_form, state, V, P, F

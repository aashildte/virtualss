"""

Functions for defining cardiac mechanics equations including strain energy
functions and weak form terms. The point of this class is to provide a simple
continuum mechanics framework such that the examples in the demos can be run
on their own. However, this can easily be replaced by your own more advanced
FEniCS code.

Åshild Telle / University of Washington / 2023–2022

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

    W_hat = a / (2 * b) * (ufl.exp(b * (IIFx - dim)) - 1)
    W_f = a_f / (2 * b_f) * (ufl.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

    return W_hat + W_f


class CardiacModel:
    def __init__(self, mesh, remove_rm=False, material_parameters={}):
        """

        Defines function spaces (P1 x P2) and functions to solve for, as well
        as the weak form for the problem itself. This assumes a fully incompressible
        formulation, solving for the displacement and the hydrostatic pressure.

        Args:
            mesh (df.Mesh): domain to solve equations over
            remove_rm (boolean): If set to be true, a weak form term is added
                for removing rigid motion using Lagrangian multipliers
            material_parameters - dictionary; use parameters as keys

        Returns:
            weak form (ufl form), state, displacement, boundary conditions
            stretch_fun (ufl form): function that assigns Dirichlet bcs
                    on wall to be stretched/extended

        """
        
        self.mesh = mesh
        self.material_parameters = material_parameters
        self.remove_rm = remove_rm
        self.dim = 2 if mesh.geometric_dimension() == 2 else 3

        self.set_compiler_options()
        self.define_state_spaces()
        self.define_state_functions()
        self.define_weak_form()

    def set_compiler_options(self):
        df.parameters["form_compiler"]["cpp_optimize"] = True
        df.parameters["form_compiler"]["representation"] = "uflacs"
        df.parameters["form_compiler"]["quadrature_degree"] = 4


    def define_state_spaces(self):
        mesh =self.mesh

        P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        
        if self.remove_rm:
            num_dofs = 3 if self.dim == 2 else 6
            R = ufl.VectorElement("Real", mesh.ufl_cell(), 0, num_dofs)
            state_space = df.FunctionSpace(mesh, df.MixedElement([P2, P1, R]))
        else:
            state_space = df.FunctionSpace(mesh, df.MixedElement([P2, P1]))

        V = state_space.sub(0)

        self.state_space, self.V = state_space, V

    def define_state_functions(self):
        state_space = self.state_space
        dim = self.mesh.topology().dim()

        state = df.Function(state_space)
        test_state = df.TestFunction(state_space)

        if self.remove_rm:
            u, p, r = df.split(state)
            v, q, _ = df.split(test_state)
        else:
            u, p = df.split(state)
            v, q = df.split(test_state)

        # Kinematics
        d = len(u)
        I = ufl.Identity(d)  # Identity tensor
        F = ufl.variable(I + ufl.grad(u))  # Deformation gradient
        J = ufl.det(F)
        
        psi = psi_holzapfel(F, dim=dim, **self.material_parameters)
        P = ufl.diff(psi, F) + p * J * ufl.inv(F.T)

        self.P, self.F, self.J, = P, F, J
        self.u, self.p, self.v, self.q = u, p, v, q
        self.state, self.test_state = state, test_state
        
        if self.remove_rm:
            self.r = r
        

    def _remove_rigid_motion_term(self):
        state, test_state = self.state, self.test_state
        u, r = self.u, self.r
        mesh = self.mesh

        position = df.SpatialCoordinate(mesh)

        if self.dim == 3:
            RM = [
                df.Constant((1, 0, 0)),
                df.Constant((0, 1, 0)),
                df.Constant((0, 0, 1)),
                df.cross(position, df.Constant((1, 0, 0))),
                df.cross(position, df.Constant((0, 1, 0))),
                df.cross(position, df.Constant((0, 0, 1))),
            ]
        else:
            RM = [
                df.Constant((1, 0)),
                df.Constant((0, 1)),
                df.Expression(("-x[1]", "x[0]"), degree=1),
            ]

        Pi = sum(df.dot(u, zi) * r[i] * df.dx for i, zi in enumerate(RM))

        return df.derivative(Pi, state, test_state)


    def define_weak_form(self):
        P, J, u, v, p, q = self.P, self.J, self.u, self.v, self.p, self.q

        elasticity_term = ufl.inner(P, ufl.grad(v)) * df.dx
        pressure_term = q * (J - 1) * df.dx

        weak_form = elasticity_term + pressure_term

        if self.remove_rm:
            weak_form += self._remove_rigid_motion_term()

        self.weak_form = weak_form


    def solve(self, bcs=[]):
        df.solve(self.weak_form == 0, self.state, bcs=bcs)

# Virtual Stress-Strain experiments

Fenics-based code for performing stretch experiments on "tissue slabs", performing stress-strain curves. The code should work in 2D and 3D on any rectangular or boxlike mesh.

We assume that the rectangle/box is aligned with the cartesian coordinate system, such that e.g. "x" direction will correspond to the direction given by the vector (1, 0) or (1, 0, 0). A deformation of the kind "xx" will happen along this axis.

*The code is experimental and might not produce even remotely correct results.*

# Tensile/compressive deformation

The following deformation modes are supported:
* stretch\_xx
* stretch\_yy
* stretvh\_zz
  
which can be applied using either
* fixed surfaces on both ends ("minimum" side and "maximum" side in either dimension); here the area won't change so only the middle part will deform
* fixed components on both ends; here only the x, y, or z component will be fixed such that the area shrinks with deformation and the rectangular or box-like shape is preserved overall
* an applied load; which will be applied symmetrically on both sides as in pulling on something with equal force

The "fixed component" and "applied load" approaches should be identical for a homogeneous domain, as in, for every load there is a corresponding displacement which results in the same kind of deformation for the two modes. These modes are also shape- and dimension-independent, as demonstrated in demos/compare\_dimensions.py.

Even if the deformations/functions are named "stretch" any compressive deformation can also be performed simply by applying negative instead of positive values.


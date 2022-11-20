# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Elasticity
#
# Copyright © 2020-2022 Garth N. Wells and Michal Habera
#
# This demo solves the equations of static linear elasticity using a
# smoothed aggregation algebraic multigrid solver. The demo is
# implemented in {download}`demo_elasticity.py`.

# +
from contextlib import ExitStack

import numpy as np
import math

import ufl
from dolfinx import la, fem
from dolfinx.fem import (Expression, Function, FunctionSpace,
                         VectorFunctionSpace, dirichletbc, form,
                         locate_dofs_topological, Constant)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, GhostMode, create_box,
                          locate_entities_boundary, meshtags)
from ufl import dx, grad, inner, dot, div

from mpi4py import MPI
from petsc4py import PETSc

import sympy as sp

dtype = PETSc.ScalarType
# -

# ## Operator's nullspace
#
# Smooth aggregation algebraic multigrid solvers require the so-called
# 'near-nullspace', which is the nullspace of the operator in the
# absence of boundary conditions. The below function builds a PETSc
# NullSpace object. For this 3D elasticity problem the nullspace is
# spanned by six vectors -- three translation modes and three rotation
# modes.


def build_nullspace(V):
    """Build PETSc nullspace for 3D elasticity"""

    # Create list of vectors for building nullspace
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    ns = [la.create_petsc_vector(index_map, bs) for i in range(6)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace (x, y and z dofs)
        dofs = [V.sub(i).dofmap.list.array for i in range(3)]

        # Build the three translational rigid body modes
        for i in range(3):
            basis[i][dofs[i]] = 1.0

        # Build the three rotational rigid body modes
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        basis[3][dofs[0]] = -x1
        basis[3][dofs[1]] = x0
        basis[4][dofs[0]] = x2
        basis[4][dofs[2]] = -x0
        basis[5][dofs[2]] = x1
        basis[5][dofs[1]] = -x2

    # Orthonormalise the six vectors
    la.orthonormalize(ns)
    assert la.is_orthonormal(ns)

    return PETSc.NullSpace().create(vectors=ns)

# Create a box Mesh
Lx = 1
Ly = 1
Lz = 0.5

msh = create_box(MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                                  np.array([Lx, Ly, Lz])], [40, 40, 20], # 2: 40, 40, 20 # 3: 30, 30, 20
                 CellType.tetrahedron, GhostMode.shared_facet)

# Create a centripetal source term ($f = \rho \omega^2 [x_0, \, x_1]$)

ω, ρ = 300.0, 10.0
x = ufl.SpatialCoordinate(msh)
#f = ufl.as_vector((ρ * ω**2 * x[0], ρ * ω**2 * x[1], 0.0))
f = ufl.as_vector((0.0, 0.0, 0.0))

###
R = 0.15
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
def ffunc(x, a, b, c, d, f):
    R = 0.15
    return np.exp(-(a*x**10 + b*x**8  + c*x**6 + d*x**4 + f*x**2))*R
def cap(x):
    R = 0.15
    y = np.array([])
    for x_i in x:
        if (x_i >= -R) and (x_i <= R):
            f = np.sqrt(R**2-x_i**2)
        else:
            f = 0.
        y = np.append(y,f)
    return y

xdata = np.linspace(-1.1*R, 1.1*R, 300)
ydata = cap(xdata)
plt.plot(xdata, ydata, 'b-', label='data')

popt, pcov = curve_fit(ffunc, xdata, ydata)

plt.plot(xdata, ffunc(xdata, *popt), 'g--')
plt.axis([-1.2*R, 1.2*R, 0, 1.1*R])
plt.show()
###
x0 = 0.5
y0 = 0.5
T = ufl.as_vector((0.0, 0.0, ufl.exp(-((x[0]-0.5)**2 + (x[1]-0.5)**2)/(0.1)**2)))
T_2 = ufl.as_vector((0.0, 0.0, 10**8*R*ufl.exp(-(popt[0]*((x[0]-x0)**2 + (x[1]-y0)**2)**5 + popt[1]*((x[0]-x0)**2 + (x[1]-y0)**2)**4 + popt[2]*((x[0]-x0)**2 + (x[1]-y0)**2)**3 + popt[3]*((x[0]-x0)**2 + (x[1]-y0)**2)**2 + popt[4]*((x[0]-x0)**2 + (x[1]-y0)**2)**1)/1)))
# Set the elasticity parameters and create a function that computes and
# expression for the stress given a displacement field.

# +
E = 26.90e10 #1.0e9
ν = 0.47
μ = E / (2.0 * (1.0 + ν))
λ = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν))

def c_new(C11, C12, C44, alpha):
    c11, c12, c44 = sp.symbols('c11 c12 c44')

    C_sym = sp.Array([[[[c11, 0, 0],
                        [0, c12, 0],
                        [0, 0, c12]],
                       
                        [[0, c44, 0],
                         [c44, 0, 0],
                         [0, 0, 0]],
                        
                        [[0, 0, c44],
                         [0, 0, 0],
                         [c44, 0, 0]]],
                      
                      
                      [[[0, c44, 0],
                        [c44, 0, 0],
                        [0, 0, 0]],
                       
                       [[c12, 0, 0],
                        [0, c11, 0],
                        [0, 0, c12]],
                       
                       [[0, 0, 0],
                        [0, 0, c44],
                        [0, c44, 0]]],
                      
                      
                      [[[0, 0, c44],
                        [0, 0, 0],
                        [c44, 0, 0]],
                       
                       [[0, 0, 0],
                        [0, 0, c44],
                        [0, c44, 0]],
                       
                       [[c12, 0, 0],
                        [0, c12, 0],
                        [0, 0, c11]]]])
    
    # CT1 = sp.tensorcontraction(sp.tensorproduct(C_sym, alpha), (3,5))
    # CT2 = sp.tensorcontraction(sp.tensorproduct(CT1, alpha), (2,5))
    # CT3 = sp.tensorcontraction(sp.tensorproduct(CT2, alpha), (1,5))
    # CT4 = sp.tensorcontraction(sp.tensorproduct(CT3, alpha), (0,5))
    
    CT4 = sp.tensorcontraction(sp.tensorproduct(alpha, alpha, alpha, alpha, C_sym), (1,8), (3,9), (5,10), (7,11))
    
    # CT1 = sp.tensorcontraction(sp.tensorproduct(alpha, C_sym), (1,5))
    # CT2 = sp.tensorcontraction(sp.tensorproduct(alpha, CT1), (1,4))
    # CT3 = sp.tensorcontraction(sp.tensorproduct(alpha, CT2), (1,3))
    # CT4 = sp.tensorcontraction(sp.tensorproduct(alpha, CT3), (1,2))
    
    CT_f = CT4.subs([(c11, C11), (c12, C12), (c44, C44)])
    
    CT_list = CT_f.tolist()

    my_a = np.array(CT_list, dtype = float)
    
    C = ufl.as_matrix(((my_a[0,0,0,0], my_a[0,0,1,1], my_a[0,0,2,2], my_a[0,0,1,2], my_a[0,0,2,0], my_a[0,0,0,1]),
                   (my_a[1,1,0,0], my_a[1,1,1,1], my_a[1,1,2,2], my_a[1,1,1,2], my_a[1,1,2,0], my_a[1,1,0,1]),
                   (my_a[2,2,0,0], my_a[2,2,1,1], my_a[2,2,2,2], my_a[2,2,1,2], my_a[2,2,2,0], my_a[2,2,0,1]),
                   (my_a[1,2,0,0], my_a[1,2,1,1], my_a[1,2,2,2], my_a[1,2,1,2], my_a[1,2,2,0], my_a[1,2,0,1]),
                   (my_a[2,0,0,0], my_a[2,0,1,1], my_a[2,0,2,2], my_a[2,0,1,2], my_a[2,0,2,0], my_a[2,0,0,1]),
                   (my_a[0,1,0,0], my_a[0,1,1,1], my_a[0,1,2,2], my_a[0,1,1,2], my_a[0,1,2,0], my_a[0,1,0,1])))
    return C

alpha_210 = sp.Array([[0,            0,             -1],
                  [-1/sp.sqrt(5), 2/sp.sqrt(5),  0],
                  [ 2/sp.sqrt(5), 1//sp.sqrt(5), 0]])

alpha_id = sp.Array([[1,0,0],
                     [0,1,0],
                     [0,0,1]])

alpha_111 = sp.Array([[-1/sp.sqrt(6), -1/sp.sqrt(6), 2/sp.sqrt(6)],
                      [ 1/sp.sqrt(2), -1/sp.sqrt(2), 0],
                      [ 1/sp.sqrt(3),  1/sp.sqrt(3), 1/sp.sqrt(3)]])

c11 = 26.90*10**10
c12 = 10.77*10**10
c44 = 7.64*10**10

AA = 9.5*10**(-8)
kk = 1054
delta = np.sqrt(AA/kk) # cm
h = 1*delta #0.001/40 # cm

C = c_new(c11, c12, 0.5*(c11-c12), alpha_id)

def σ(v):
    """Return an expression for the stress σ given a displacement field"""
    return 2.0 * μ * ufl.sym(grad(v)) + λ * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(len(v))
def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)
def epsilon_n(u, d, h):
    u0, u1, u2 = ufl.split(u)
    eps11 = u0.dx(0)
    eps12 = 0.5*(u0.dx(1) + u1.dx(0))
    eps13 = 0.5*(h/d*u0.dx(2) + d/h*u2.dx(0))
    eps22 = u1.dx(1)
    eps23 = 0.5*(h/d*u1.dx(2) + d/h*u2.dx(1))
    eps33 = u2.dx(2)
    eps = ufl.as_tensor(((eps11, eps12, eps13),
                         (eps12, eps22, eps23),
                         (eps13, eps23, eps33)))
    return eps
# -

# A function space space is created and the elasticity variational
# problem defined:


V = VectorFunctionSpace(msh, ("Lagrange", 2))
u = ufl.TrialFunction(V)
#eps = epsilon_n(u, delta, h)
eps = epsilon(u)
v = ufl.TestFunction(V)

eps_vec = ufl.as_vector((eps[0,0], eps[1,1], eps[2,2], eps[1,2], eps[0,2], eps[0,1]))
sigma_vec = dot(C, eps_vec) #C*eps_vec

sigma_mat = ufl.as_matrix(((sigma_vec[0], sigma_vec[5], sigma_vec[4]),
                       (sigma_vec[5], sigma_vec[1], sigma_vec[3]),
                       (sigma_vec[4], sigma_vec[3], sigma_vec[2])))

# beta_v = ufl.as_matrix(((1/delta, 0, 0),
#                         (0, 1/delta, 0),
#                         (0, 0, 1/h)))
beta_v = ufl.as_matrix(((1, 0, 0),
                        (0, 1, 0),
                        (0, 0, 1)))

a = form(inner(sigma_mat, epsilon(v))*dx)
#a = form(inner(σ(u), grad(v)) * dx)
# L = form(-dot(T_2,v)*ds) #form(inner(f, v) * dx)

T_expr = Expression(T_2, V.element.interpolation_points())
T_f = Function(V)
T_f.interpolate(T_expr)

def disp_exp(x):
    R = 0.15
    return np.stack(np.zeros(x.shape[1]), np.zeros(x.shape[1]), -10**(-10)*R*np.exp(-(x[0]**2 + x[1]**2)))
# u_b = Function(V)
# u_b.interpolate(disp_exp)

with XDMFFile(msh.comm, "out_elasticity/T.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(T_f)

# A homogeneous (zero) boundary condition is created on $x_0 = 0$ and
# $x_1 = 1$ by finding all boundary facets on $x_0 = 0$ and $x_1 = 1$,
# and then creating a Dirichlet boundary condition object.

facets = locate_entities_boundary(msh, dim=2,
                                  marker=lambda x: np.isclose(x[2], 0.0, atol=0.0))
#np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[1], 1.0))
bc = dirichletbc(np.zeros(3, dtype=dtype),
                 locate_dofs_topological(V, entity_dim=2, entities=facets), V=V)

top_facets = locate_entities_boundary(msh, dim=2,
                                  marker=lambda x: np.isclose(x[2], Lz, atol=0.0))
top_bc = dirichletbc(T_f, locate_dofs_topological(V, entity_dim=2, entities=top_facets))
bc_all = [bc]
mt = meshtags(msh, 2, top_facets, 1)

ds = ufl.Measure("ds", subdomain_data=mt)
L = form(-dot(T_2,v)*ds(1)) #form(dot(Constant(msh, (0.,0.,0.)),v)*dx) #form(-dot(T_2,v)*ds(1))

# ## Assemble and solve
#
# The bilinear form `a` is assembled into a matrix `A`, with
# modifications for the Dirichlet boundary conditions. The line
# `A.assemble()` completes any parallel communication required to
# computed the matrix.

# +
A = assemble_matrix(a, bcs=bc_all)
A.assemble()
# -

# The linear form `L` is assembled into a vector `b`, and then modified
# by `apply_lifting` to account for the Dirichlet boundary conditions.
# After calling `apply_lifting`, the method `ghostUpdate` accumulates
# entries on the owning rank, and this is followed by setting the
# boundary values in `b`.

# +
b = assemble_vector(L)
apply_lifting(b, [a], bcs=[bc_all])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bc_all)
# -

# Create the near-nullspace and attach it to the PETSc matrix:

null_space = build_nullspace(V)
A.setNearNullSpace(null_space)

# Set PETSc solver options, create a PETSc Krylov solver, and attach the
# matrix `A` to the solver:

# +
# Set solver options
opts = PETSc.Options()
opts["ksp_type"] = "cg" #"cg" "gmres"
opts["ksp_rtol"] = 1.0e-10
opts["pc_type"] = "gamg"

# Use Chebyshev smoothing for multigrid
opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_pc_type"] = "jacobi"

# Improve estimate of eigenvalues for Chebyshev smoothing
opts["mg_levels_esteig_ksp_type"] = "cg"
opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

# Create PETSc Krylov solver and turn convergence monitoring on
solver = PETSc.KSP().create(msh.comm)
solver.setFromOptions()

# Set matrix operator
solver.setOperators(A)
# -

# Create a solution {py:class}`Function<dolfinx.fem.Function>`, `uh`, and
# solve:

# +
uh = Function(V)

# Set a monitor, solve linear system, and display the solver
# configuration
solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
solver.solve(b, uh.vector)
solver.view()

# Scatter forward the solution vector to update ghost values
uh.x.scatter_forward()
# -

# ## Post-processing
#
# The computed solution is now post-processed.
#
# Expressions for the deviatoric and Von Mises stress are defined:

# +
sigma_dev = σ(uh) - (1 / 3) * ufl.tr(σ(uh)) * ufl.Identity(len(uh))
sigma_vm = ufl.sqrt((3 / 2) * inner(sigma_dev, sigma_dev))
# -
# 22-component of deformation tensor
#eps_22 = grad(uh)[2,2]
u0, u1, u2 = ufl.split(uh)
eps_22 = u2.dx(2)
# Next, the Von Mises stress is interpolated in a piecewise-constant
# space by creating an {py:class}`Expression<dolfinx.fem.Expression>`
# that is interpolated into the
# {py:class}`Function<dolfinx.fem.Function>` `sigma_vm_h`.

# +
W = FunctionSpace(msh, ("Discontinuous Lagrange", 0))
sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
sigma_vm_h = Function(W)
sigma_vm_h.interpolate(sigma_vm_expr)

# projection of eps_22
V_1 = FunctionSpace(msh, ("Lagrange", 3))
#V_n = VectorFunctionSpace(msh, ("Lagrange", 3))
eps_22_expr = Expression(eps_22, V_1.element.interpolation_points())
eps_22_h = Function(V_1)
eps_22_h.interpolate(eps_22_expr)

# projection of dz of eps_22
# w =  ufl.TestFunction(V_n)
# u = ufl.TrialFunction(V_n)

# u2.interpolate(V_n)
# a1 = (dot(u,w)*dx)
# n = ufl.FacetNormal(msh)
# L1 = (u2*dot(w,n)*ds - u2*div(w)*dx)

# problem = fem.petsc.LinearProblem(a1, L1, bcs = [])
# ge = problem.solve()
# ge1, ge2, ge3 = ufl.split(ge)

eps_22_dz = grad(eps_22_h)[2]
V_1 = FunctionSpace(msh, ("Lagrange", 3))
eps_22_dz_expr = Expression(eps_22_dz, V_1.element.interpolation_points())
eps_22_dz_h = Function(V_1)
eps_22_dz_h.interpolate(eps_22_dz_expr)
# -

# Save displacement field `uh` and the Von Mises stress `sigma_vm_h` in
# XDMF format files.

# +
with XDMFFile(msh.comm, "out_elasticity/displacements.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)

# Save solution to XDMF format
with XDMFFile(msh.comm, "out_elasticity/von_mises_stress.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(sigma_vm_h)

# Save eps_22
with XDMFFile(msh.comm, "out_elasticity/eps_22.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(eps_22_h) #ge

# Save eps_22_dz
with XDMFFile(msh.comm, "out_elasticity/eps_22_dz.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(eps_22_dz_h)
# -

# Finally, we compute the $L^2$ norm of the displacement solution
# vector. This is a collective operation (i.e., the method `norm` must
# be called from all MPI ranks), but we print the norm only on rank 0.

# +
unorm = uh.x.norm()
if msh.comm.rank == 0:
    print("Solution vector norm:", unorm)
# -

# The solution vector norm can be a useful check that the solver is
# computing the same result when running in serial and in parallel.

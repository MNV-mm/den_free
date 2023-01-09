#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 13:34:24 2023

@author: mnv
"""

from fenics import *
import math

mesh = Mesh("/home/mnv/Documents/python_doc/llg_nl/MESH.xml")

El = FiniteElement('CG', triangle, 1)
nFS = FunctionSpace(mesh, El)
El = VectorElement('CG', triangle, 1, dim=3)
nFS3 = FunctionSpace(mesh, El)

u = TrialFunction(nFS)
v = TestFunction(nFS)

kk = 1000 #1054# erg/cm**3 - unaxial anisotropy constant
M_s = 3.46
t0 = 1*45/180*math.pi

ub = Expression(("0", "-sin(2*atan(exp(x[1]/d)))*cos(a) + cos(2*atan(exp(x[1]/d)))*sin(a)", "sin(a)*sin(2*atan(exp(x[1]/d))) + cos(2*atan(exp(x[1]/d)))*cos(a)"), degree = 4, d=1, a = t0)
m = project(ub,nFS3)

m1, m2, m3 = split(m)
m_2d = as_vector((m1,m2))
m_b = m_2d

# vtkfile = File('/media/mnv/A2E41E9EE41E74AF/graphs/phi_prev.pvd')
# vtkfile << phi_prev

bv_bl = Expression('0',degree = 3)
bv_nl = Expression("-4*p*(2*b*atan(exp(x[1]/b))*cos(t0) + x[1]*sin(t0) - b*asinh(1/2*(2*exp(2*x[1]/b) + exp(4*x[1]/b))/(1+exp(2*x[1]/b)))*sin(t0))", degree=4, p = math.pi, b = 1, t0 = t0)
phi_prev = project(bv_bl, nFS)
#bv_nl = Expression("4*p*(2*b*atan(exp(x[1]/b))*sin(t0))", degree=4, p = math.pi, b = b, t0 = t0)
def boundary(x, on_boundary):
    return on_boundary

wall_type = 'neel'
if wall_type == 'bloch':
    BC = DirichletBC(nFS, bv_bl, boundary)
    
if wall_type == 'neel':
    BC = DirichletBC(nFS, bv_nl, boundary)

f_pi = math.pi*4
F_Pi = Constant(f_pi)
n = FacetNormal(mesh)

#F = dot(grad(u),grad(v))*dx + F_Pi*(m1.dx(0) + m2.dx(1))*v*dx - F_Pi*v*dot(n,m_b)*ds
F = -F_Pi*v*dot(m_b,n)*ds - dot(grad(v),grad(u))*dx + F_Pi*dot(m_2d,grad(v))*dx # + F_Pi*v*dot(grad(u),n)*ds
a = lhs(F)
L = rhs(F)

A = assemble(a)
b = assemble(L)
#BC.apply(A,b)

solver = KrylovSolver('gmres', 'ilu')
solver.parameters["nonzero_initial_guess"] = True
u = Function(nFS)
u.vector()[:] = phi_prev.vector()
U = u.vector()

solver.solve(A, U, b)
u.vector()[:] = U
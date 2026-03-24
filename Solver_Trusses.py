#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:34:19 2021

@author: kendrick
"""

import numpy as np

# compute unknown displacements 
def ComputeDisplacements(K, F, n_unknowns):
    # extract submatrix of unknowns
    K11 = K[0:n_unknowns,0:n_unknowns]
    F1 = F[0:n_unknowns]
    
    d = np.linalg.solve(K11,F1)
    
    return d

# postprocess the forces at known displacement nodes
def PostprocessReactions(K, d, F, n_unknowns, nodes):
    # These are computed net forces and do not
    # take into account external loads applied
    # at these nodes
    F = np.matmul(K[n_unknowns:,0:n_unknowns], d)
    
    # Postprocess the reactions
    for node in nodes:
        if node.xidx >= n_unknowns:
            node.AddReactionXForce(F[node.xidx-n_unknowns][0] - node.xforce_external)
        if node.yidx >= n_unknowns:
            node.AddReactionYForce(F[node.yidx-n_unknowns][0] - node.yforce_external)
        
    return F

# determine internal member loads
def ComputeMemberForces(bars):
    # Compute member forces for all bars using equation 14-23 
    
    for bar in bars:
        # get properties
        E = bar.E
        A = bar.A
        L = bar.Length()
        
        # direction cosines
        lambdax, lambday = bar.LambdaTerms()
        
        # nodal displacements
        Dx_i = bar.init_node.xdisp
        Dy_i = bar.init_node.ydisp
        Dx_j = bar.end_node.xdisp
        Dy_j = bar.end_node.ydisp
        
        # equation 14-23
        q = (A * E / L) * (
            -lambdax * Dx_i
            -lambday * Dy_i
            + lambdax * Dx_j
            + lambday * Dy_j
        )
        
        # store result
        bar.axial_load = q
    
# compute the normal stresses
def ComputeNormalStresses(bars):
    # Compute normal stress for all bars
    
    for bar in bars:
        # stress = force / area
        bar.normal_stress = bar.axial_load / bar.A

def ComputeBucklingLoad(bars):
    for bar in bars:
        E = bar.E
        I = bar.It
        L = bar.Length()
        K = 1.0
        
        Pcr = (np.pi**2 * E * I) / ((K * L)**2)
        
        bar.buckling_load = Pcr

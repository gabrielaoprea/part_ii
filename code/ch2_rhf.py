#!/usr/bin/env python

from pyscf import gto, scf
import pyscf
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from numpy.linalg import inv
from matplotlib import pyplot as plt
import rhf_perturbation as pt
import approximants as apx
from scipy.interpolate import pade

bond_lengths = np.arange(0.2, 4, 0.1)
e_list_es_1 = []
e_list_es_2 = []
for i in bond_lengths:
    x = 0.779*i
    y = 0.626*i
    at_string ='C 0 0 0; H -{x_val}, {y_val} 0; H {x_val}, {y_val}, 0'.format(x_val = x, y_val = y)
    #bond_lengths = np.arange(0.2,4.6,0.05)
    mol = gto.M(
        atom = at_string,  # in Angstrom
        basis = 'STO-3G',
        #spin = 1,
    )

    myhf = scf.HF(mol)
    myhf.max_cycle = 100000
    myhf.kernel()

    fock_at = myhf.get_fock()
    density = myhf.make_rdm1()
    hcore = myhf.get_hcore()
    mo_en = myhf.mo_energy
    perm = pt.create_permutations(mol, mo_en)
    perm.reverse()
    spin = pt.get_spin(perm)
    orb = myhf.mo_coeff
    orb_og = myhf.mo_coeff
    fock_mo = orb.transpose().dot(fock_at).dot(orb)
    hcore_mo = orb.transpose().dot(hcore).dot(orb)
    #fock = pt.full_fock(fock_mo,perm) 
    ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
    mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
    nuc = mol.energy_nuc()
    fock, h = pt.get_full_matrices(nuc, hcore_mo, fock_mo, mo_integrals, perm)
    psi, e = pt.mppt(h, fock, 10,0)
    e_list_es_1.append(list(e))


    fock_at = myhf.get_fock()
    density = myhf.make_rdm1()
    hcore = myhf.get_hcore()
    mo_en = myhf.mo_energy
    perm = pt.create_permutations(mol, mo_en)
    perm.reverse()
    spin = pt.get_spin(perm)
    orb = myhf.mo_coeff
    orb_og = myhf.mo_coeff
    orb[:,[3, 4]] = orb[:,[4,3]]
    fock_mo = orb.transpose().dot(fock_at).dot(orb)
    hcore_mo = orb.transpose().dot(hcore).dot(orb)
    #fock = pt.full_fock(fock_mo,perm) 
    ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
    mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
    nuc = mol.energy_nuc()
    fock, h = pt.get_full_matrices(nuc, hcore_mo, fock_mo, mo_integrals, perm)
    psi, e = pt.mppt(h, fock, 10,0)
    e_list_es_2.append(list(e))

print("Bond_lengths")
print(bond_lengths)
print("First es")
print(e_list_es_1)
print("Second es")

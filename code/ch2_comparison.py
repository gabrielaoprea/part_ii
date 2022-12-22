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

x = 0.779*1.11
y = 0.626*1.11
at_string ='C 0 0 0; H -{x_val}, {y_val} 0; H {x_val}, {y_val}, 0'.format(x_val = x, y_val = y)
#bond_lengths = np.arange(0.2,4.6,0.05)
mol = gto.M(
    atom = at_string,  # in Angstrom
    basis = 'STO-3G',
    #spin = 1,
    )

print("Delta")

myhf = scf.HF(mol)
myhf.kernel()

fock_at = myhf.get_fock()
density = myhf.make_rdm1()
hcore = myhf.get_hcore()
mo_en = myhf.mo_energy
perm = pt.create_permutations(mol, mo_en)
perm.reverse()
spin = pt.get_spin(perm)
orb = myhf.mo_coeff
orb[:,[3, 4]] = orb[:,[4,3]]
fock_mo = orb.transpose().dot(fock_at).dot(orb)
hcore_mo = orb.transpose().dot(hcore).dot(orb)
#fock = pt.full_fock(fock_mo,perm) 
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
nuc = mol.energy_nuc()
fock, h = pt.get_full_matrices(nuc, hcore_mo, fock_mo, mo_integrals, perm)
psi, e_delta = pt.mppt(h, fock, 100,0)
print("Series")
print(e_delta)

print("Approximants")
r, p, q = apx.quadratic_approximant(e_delta, 2, 2, 2)
discriminant = p**2 - 4*q*r
root = discriminant.roots()
print(root)
root = np.abs(root)
root = list(root)
print(root)

print("MOM")
myhf = scf.HF(mol)
myhf.max_cycle = 100000
myhf.scf()
fock=myhf.get_fock()
density=myhf.make_rdm1()
hcore=myhf.get_hcore()
mo_en = myhf.mo_energy
orb = myhf.mo_coeff
occ = myhf.mo_occ

occ[3] = 0
occ[4] = 2
mo_en = myhf.mo_energy
b = scf.HF(mol)
b.max_cycle = 100000
dm_u = b.make_rdm1(orb, occ)
b = scf.addons.mom_occ(b, orb, occ)
b.scf(dm_u)
            
fock=b.get_fock()
density=b.make_rdm1()
hcore=b.get_hcore()
mo_en = b.mo_energy
orb = b.mo_coeff
perm = pt.create_permutations(mol, mo_en)
perm.reverse()
orb = b.mo_coeff
occ = b.mo_occ
print(occ)
fock_mo = orb.transpose().dot(fock).dot(orb)
hcore_mo = orb.transpose().dot(hcore).dot(orb)
#fock = pt.full_fock(fock_mo,perm) 
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
nuc = mol.energy_nuc()
fock, h = pt.get_full_matrices(nuc, hcore_mo, fock_mo, mo_integrals, perm)
psi, e_mom = pt.mppt(h, fock, 100,0)
print("Series")
print(e_mom)

print("Approximants")
r, p, q = apx.quadratic_approximant(e_mom, 2, 2, 2)
discriminant = p**2 - 4*q*r
root = discriminant.roots()
print(root)
root = np.abs(root)
root = list(root)
print(root)


#!/usr/bin/env python

from pyscf import gto, scf, lo
from pyscf.tools import cubegen
import pyscf
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from numpy.linalg import inv
from matplotlib import pyplot as plt
import rhf_perturbation as pt
import approximants as apx
from scipy.interpolate import pade

mol = gto.M(
    atom = "H 0 0 0; F 0 0 0.94",
    basis = {'H':'631G', 'F':'STO3G'},
    )

myhf = scf.HF(mol)
myhf.max_cycle = 10000
myhf.scf()
d_gs = myhf.make_rdm1()

orb = myhf.mo_coeff
occ = myhf.mo_occ
occ[0] = 0
occ[5] = 2
dm_u = myhf.make_rdm1(orb, occ)
b_1 = scf.addons.mom_occ(myhf, orb, occ)
b_1.scf(dm_u)
d_1 = b_1.make_rdm1()

fock_at = b_1.get_fock()
d_1 = b_1.make_rdm1()
hcore = b_1.get_hcore()
mo_en = b_1.mo_energy
orb = b_1.mo_coeff
perm = pt.create_permutations(mol, mo_en)
perm.reverse()
orb = b_1.mo_coeff
occ = b_1.mo_occ
k = 0
for i in range(len(occ)):
    if occ[i] == 2:
       orb[:,[k, i]] = orb[:,[i, k]]
       k +=1

fock_mo = orb.transpose().dot(fock_at).dot(orb)
hcore_mo = orb.transpose().dot(hcore).dot(orb)
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
nuc = mol.energy_nuc()
fock, h = pt.get_full_matrices(nuc, hcore_mo, fock_mo, mo_integrals, perm)
psi, e = pt.mppt(h, fock, 20, 0) 

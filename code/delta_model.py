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

orb = myhf.mo_coeff
fock_at = myhf.get_fock()
density = myhf.make_rdm1()
hcore = myhf.get_hcore()
mo_en = myhf.mo_energy
perm = pt.create_permutations(mol, mo_en)
perm.reverse()
spin = pt.get_spin(perm)
orb = myhf.mo_coeff
orb_og = myhf.mo_coeff
orb[:,[0, 5]] = orb[:,[5, 0]]
fock_mo = orb.transpose().dot(fock_at).dot(orb)
hcore_mo = orb.transpose().dot(hcore).dot(orb)
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
nuc = mol.energy_nuc()
fock, h = pt.get_full_matrices(nuc, hcore_mo, fock_mo, mo_integrals, perm)
psi, e = pt.mppt(h, fock, 20, 0)

#!/usr/bin/env python

from pyscf import gto, scf
import pyscf
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from numpy.linalg import inv
from matplotlib import pyplot as plt

def energy(energy_list, permutation):
    '''
    Returns the energy of a specific electronic configuration. 
    Arguments:
        energy_list - list of energies of orbitals
        permutation - list of 0 and 1 representing occupancy of the orbitals with energies given by energy_list; 
                      NB: permutation is only for one spin - alpha or beta. 
    '''
    l = len(energy_list)
    s = 0
    for i in range(l):
        if permutation[i] == 1:
            s+=energy_list[i]
    return s

def create_permutations(molecule, mo_en):
    '''
    Creates a list of all the possible permutations of alpha and beta electrons in the given molecule.
    The list is an N x 2 2D array - N = # of possible states; 2 = alpha electrons and beta electrons
    configurations are given separately.
    '''
    perm =[]
    n = len(mo_en) #number of orbitals 
    occ_alpha = (molecule.nelectron+molecule.spin)//2 #no of beta electrons
    occ_beta = molecule.nelectron-occ_alpha #no of alpha electrons
    alpha = [1]*occ_alpha+[0]*(n-occ_alpha)
    beta = [1]*occ_beta+[0]*(n-occ_beta)
    for i in multiset_permutations(alpha):
        for j in multiset_permutations(beta):
            perm.append([i,j])
    return perm

def get_es(energy_list, molecule, list_of_permutations):
    '''
    Returns the list of energies of all the possible electronic configurations of a molecule.
    Arguments:
        energy_list - list of energies either of orbitals or hcore;
        molecule - Mole object;
        list_of_permutations - all possible states expressed as permutations.
    '''
    f_energies = []
    for i in list_of_permutations:
        p_energy = energy(energy_list, i[0]) + energy(energy_list, i[1])
        f_energies.append(p_energy)
    return f_energies

def get_fock_e0(energy_list):
    '''
    Returns the list of energies with respect to the ground state.
    '''
    e0 = np.min(energy_list)
    f = []
    for i in range(len(energy_list)):
        f.append(energy_list[i]-e0)
    return f

def strip_of_0(list_of_values):
    return [value for value in list_of_values if value]

def make_diagonal_matrix(list_of_values):
    '''
    Returns a diagonal matrix - the values on the diagonal are given by the argument list_of_values.
    '''
    n = len(list_of_values)
    m = np.zeros((n,n))
    for i in range(n):
        m[i][i] = list_of_values[i]
    return m

def electron_change(state):
    '''
    Argument: 
        array which represents the difference between 2 states - i.e. array of 0, 1 and -1 (0 if no change, 1 if electrom moved ups, -1 if electron moved down)
    Returns:
        2 lists with the initial and final positions of electrons that change place.
    '''
    pos_i = []
    pos_f = []
    for i in range(len(state)):
        if state[i] == 1:
            pos_i.append(i)
        if state[i] == -1:
            pos_f.append(i)
    return pos_i, pos_f

def electron_pos(state):
    '''
    Argument:
        list of 1s and 0s representing the electrons
    Returns:
        list of position of electrons in the orbitals (i.e. no. of each orbital which has electrons).
    '''
    pos = []
    for i in range(len(state)):
        if state[i] == 1:
            pos.append(i)
    return pos

def difference(es_1, es_2):
    '''
    Arguments:
        2 states of the form defined in the creation of the permutations (list of alpha electrons, list of beta electrons)
    Returns:
        n_alpha, n_beta - no of electrons with spin alpha and beta that have changed position during the transition
        change_alpha, change_beta - positions in which thesel electrons where in the initial and final states (first half of the list = initial, second halfof the list = final)
    '''
    es_1_alpha = np.array(es_1[0])
    es_1_beta = np.array(es_1[1])
    es_2_alpha = np.array(es_2[0])
    es_2_beta = np.array(es_2[1])
    d_alpha = es_1_alpha - es_2_alpha
    d_beta = es_1_beta - es_2_beta
    change_alpha_i, change_alpha_f = electron_change(d_alpha)
    change_beta_i,change_beta_f = electron_change(d_beta)
    n_alpha = len(change_alpha_i) 
    n_beta = len(change_beta_i)
    change_alpha = change_alpha_i + change_alpha_f 
    change_beta = change_beta_i + change_beta_f
    return n_alpha, n_beta, change_alpha, change_beta

def get_pairs(list_of_permutations):
    '''
    Returns a list of all pairs of permutations that differ by 1 or 2 electrons. Each element in the list has the form:
        [i, j, d_alpha, d_beta, change_alpha, change_beta]
        i, j - #of the permutations in the pair
        d_alpha, d_beta, change_alpha, change_beta - as defined in the function difference
    '''
    n = len(list_of_permutations)
    pairs = []
    for i in range(0, n):
        for j in range(0, n):
            d_alpha, d_beta, change_alpha, change_beta = difference(list_of_permutations[i],list_of_permutations[j])
            if d_alpha + d_beta <3:
                pairs.append([i, j, d_alpha, d_beta, change_alpha, change_beta])
    return pairs

def full_fock(mo_fock, perm):
    '''
    Arguments:
        mo_fock - Fock matrix in the molecularorbital basis
        perm - list of all possiblepermutations in the Hilbert space
    Returns:
        Fock matrix in the full Hilbert space
    '''
    # Extract diagonal of Fock matrix
    fock_diag = np.diag(mo_fock)

    # Initialise many-body Fock matrix
    n = len(perm)
    fock = np.zeros((n,n))
    for i in range(n): 
        # Get diagonal terms
        fock[i,i] = (np.sum(fock_diag[np.argwhere(perm[i][0])]) 
                   + np.sum(fock_diag[np.argwhere(perm[i][1])]))

        # Save this permutation for reference
        permi0 = perm[i][0]
        permi1 = perm[i][1]

        # Loop over unique off-diagonal terms
        for j in range(i):
            # Get the "excitation string"
            exa = np.logical_xor(permi0,perm[j][0])
            exb = np.logical_xor(permi1,perm[j][1])

            # Get the number of excitations
            nexa = int(np.sum(exa)/2)
            nexb = int(np.sum(exb)/2)

            # Decide if we have a non-zero excitation
            if(nexa + nexb == 1): 
                # Find the particle-hole excitation
                if(nexa == 1):
                    # We have an alpha excitation!
                    part = np.logical_and(perm[i][0],exa)
                    hole = np.logical_and(perm[j][0],exa)
                else:
                    # We have an beta excitation!
                    part = np.logical_and(perm[i][1],exb)
                    hole = np.logical_and(perm[j][1],exb)

                # Save the corresponding Fock matrix element
                ibra = np.argwhere(part)[0,0]
                iket = np.argwhere(hole)[0,0]
                fock[i,j] = mo_fock[ibra,iket] 
                fock[j,i] = fock[i,j] 

    return fock

def get_phase(pbra,pket,pr=False):
    '''
    Compute the relative phase of an excitation between two occupation vectors
    Arguments:
        pbra - Occupation vector of the bra state
        pket - Occupation vector of the ket state
    Returns:
        phase - Integer, -1 or 1, with the relative phase
    '''

    # Check we've got the right dimensions
    assert(len(pbra)==len(pket))

    # Determine which occ vector starts furthest to the left, 
    # this will be our reference
    if(np.argwhere(pbra)[0] < np.argwhere(pket)[0]):
        ref = np.array(np.copy(pbra))
        prt = np.array(np.copy(pket))
    else:
        ref = np.array(np.copy(pket))
        prt = np.array(np.copy(pbra))
  
    # Determine the excitation indices by comparing occupation vectors
    # using bitstring tricks.
    exci = np.logical_xor(prt,ref,dtype=int)
    ihole = np.argwhere(np.logical_and(exci,ref)).flatten()
    ipart = np.argwhere(np.logical_and(exci,prt)).flatten()
 
    # Double check we have the same number of particles and holes, 
    # and save the number of excitations
    assert(len(ihole) == len(ipart))
    nex = len(ihole)

    # Now we count the number of permutations that are required to 
    # realise the excitation. If odd, we get phase = -1, otherwise phase = 1
    phase = 1
    for k in range(nex):
        i, a = ihole[k], ipart[k]
        phase *= pow(-1,sum(ref[i+1:a]))
        ref[i] = 0
        ref[a] = 1

    return phase

def get_spin(perm):
    n = len(perm)
    spin = np.zeros((n,n))
    for ibra, perm_bra in enumerate(perm):
        #Get this permutation
        pbra_a = np.asarray(perm_bra[0]) 
        pbra_b = np.asarray(perm_bra[1])

        #Find number of unpaired electrons
        n_unp =  np.sum(np.abs(pbra_a - pbra_b))

        #Get diagonal spin terms
        spin[ibra, ibra] = n_unp/2

        #Loop over unique off-diagonal terms
        for iket in range(ibra+1, n):
            #Get this permutation
            pket_a = np.asarray(perm[iket][0]) 
            pket_b = np.asarray(perm[iket][1])

            # Get the "excitation string"
            exa = np.logical_xor(pbra_a,pket_a)
            exb = np.logical_xor(pbra_b,pket_b)

            # Get common occupied orbitals
            comma = np.argwhere(np.logical_and(pbra_a,pket_a)).flatten()
            commb = np.argwhere(np.logical_and(pbra_b,pket_b)).flatten() 

            # Get the number of excitations
            nexa = int(np.sum(exa)/2)
            nexb = int(np.sum(exb)/2)

            if nexa == 1 and nexb == 1:
                holea = np.logical_and(pbra_a,exa)
                holeb = np.logical_and(pbra_b,exb)
                parta = np.logical_and(pket_a,exa)
                partb = np.logical_and(pket_b,exb)
                # Get indices of the excitation
                i = np.argwhere(holea).flatten()[0]
                j = np.argwhere(holeb).flatten()[0]
                a = np.argwhere(parta).flatten()[0]
                b = np.argwhere(partb).flatten()[0]
                if i == b and j == a:
                    spin[ibra,iket] = -1
            # Get relative phase
            phase  = get_phase(pbra_a,pket_a) * get_phase(pbra_b,pket_b)
            # And apply to matrix elements
            spin[ibra,iket]        *= phase
            
            # Hermitize...
            spin[iket,ibra] = spin[ibra,iket] 
    return spin

def get_full_matrices(en_nuc, mo_hcore, mo_fock, mo_integrals, perm):
    '''
    Compute the Hamiltonian and Fock matrices in the full Hilbert space.
    Combining these tasks into one function minimises the amount of logic
    we need to do when comparing excitations.
    Returns: 
        fock, hamiltonian
    ''' 

    # Extract diagonal of Fock matrix
    fock_diag  = np.diag(mo_fock)
    hcore_diag = np.diag(mo_hcore)
    nmo = len(fock_diag)

    # Initialise many-body Fock matrix
    n = len(perm)
    fock        = np.zeros((n,n))
    hamiltonian = np.zeros((n,n))

    # Loop over permutations
    for ibra, perm_bra in enumerate(perm):
        # Get this permutation
        pbra_a = np.asarray(perm_bra[0]) 
        pbra_b = np.asarray(perm_bra[1])

        # Find occupied orbitals 
        occa = np.argwhere(pbra_a).flatten()
        occb = np.argwhere(pbra_b).flatten()

        # Get diagonal Fock terms
        fock[ibra,ibra] = np.sum(fock_diag[occa]) + np.sum(fock_diag[occb])

        # Build a effective fock matrix for this reference
        J  = sum(mo_integrals[k,k,:,:] for k in occa) + sum(mo_integrals[k,k,:,:] for k in occb)
        Ka = sum(mo_integrals[:,k,k,:] for k in occa) 
        Kb = sum(mo_integrals[:,k,k,:] for k in occb) 
        tmpFa = mo_hcore + J - Ka
        tmpFb = mo_hcore + J - Kb
    
        # Save the diagonal Hamiltonian term
        hamiltonian[ibra,ibra] = (en_nuc + 0.5 * sum(mo_hcore[k,k] + tmpFa[k,k] for k in occa) 
                                         + 0.5 * sum(mo_hcore[k,k] + tmpFb[k,k] for k in occb)) 

        # Loop over unique off-diagonal terms
        for iket in range(ibra+1,n):
            # Get this permutation
            pket_a = np.asarray(perm[iket][0]) 
            pket_b = np.asarray(perm[iket][1])

            # Get the "excitation string"
            exa = np.logical_xor(pbra_a,pket_a)
            exb = np.logical_xor(pbra_b,pket_b)

            # Get common occupied orbitals
            comma = np.argwhere(np.logical_and(pbra_a,pket_a)).flatten()
            commb = np.argwhere(np.logical_and(pbra_b,pket_b)).flatten() 

            # Get the number of excitations
            nexa = int(np.sum(exa)/2)
            nexb = int(np.sum(exb)/2)

            # Decide if we have a non-zero excitation
            if(nexa + nexb == 1): 
                # Find the particle-hole excitation
                if(nexa == 1):
                    # We have an alpha excitation!
                    hole = np.logical_and(pbra_a,exa)
                    part = np.logical_and(pket_a,exa)
                    alpha = True
                else:
                    # We have an beta excitation!
                    hole = np.logical_and(pbra_b,exb)
                    part = np.logical_and(pket_b,exb)
                    alpha = False

                i = np.argwhere(hole)[0,0]
                a = np.argwhere(part)[0,0]

                if(alpha):
                    hamiltonian[ibra,iket] = tmpFa[i,a]
                else:
                    hamiltonian[ibra,iket] = tmpFb[i,a]

            elif(nexa + nexb == 2):
                # Two excitations, either both alpha, both beta, or one alpha and one beta 
                if(nexa == 2):
                    # We have two alpha excitations!
                    hole = np.logical_and(pbra_a,exa)
                    part = np.logical_and(pket_a,exa)
                    # Get indices of the excitation
                    i, j = np.argwhere(hole).flatten()
                    a, b = np.argwhere(part).flatten()
                    # Save the matrix element
                    hamiltonian[ibra,iket] = mo_integrals[i,a,j,b] - mo_integrals[i,b,j,a]
                elif(nexb == 2):
                    # We have two alpha excitations!
                    hole = np.logical_and(pbra_b,exb)
                    part = np.logical_and(pket_b,exb)
                    # Get indices of the excitation
                    i, j = np.argwhere(hole).flatten()
                    a, b = np.argwhere(part).flatten()
                    # Save the matrix element
                    hamiltonian[ibra,iket] = mo_integrals[i,a,j,b] - mo_integrals[i,b,j,a]
                else:
                    # We have one alpha and one beta excitations!
                    holea = np.logical_and(pbra_a,exa)
                    holeb = np.logical_and(pbra_b,exb)
                    parta = np.logical_and(pket_a,exa)
                    partb = np.logical_and(pket_b,exb)
                    # Get indices of the excitation
                    i = np.argwhere(holea).flatten()[0]
                    j = np.argwhere(holeb).flatten()[0]
                    a = np.argwhere(parta).flatten()[0]
                    b = np.argwhere(partb).flatten()[0]
                    # Save the matrix element
                    hamiltonian[ibra,iket] = mo_integrals[i,a,j,b]

            # Get relative phase
            phase  = get_phase(pbra_a,pket_a) * get_phase(pbra_b,pket_b)
            # And apply to matrix elements
            fock[ibra,iket]        *= phase
            hamiltonian[ibra,iket] *= phase
            
            # Hermitize...
            fock[iket,ibra] = fock[ibra,iket] 
            hamiltonian[iket,ibra] = hamiltonian[ibra,iket]

    return fock, hamiltonian

def get_mo_integrals(ao_integrals, orb):
    '''
    Arguments:
        ao_integrals - 2-electron integrals, as returned by ao2mo in the atomic basis
        orb - orbital coefficients
    Returns:
        mo_integrals - 2-electron integrals in the molecularorbital basis
    '''
    dim = len(orb)
    temp = np.zeros((dim,dim,dim,dim))  
    temp2 = np.zeros((dim,dim,dim,dim))  
    temp3= np.zeros((dim,dim,dim,dim))  
    mo_integrals = np.zeros((dim,dim,dim,dim))
    for i in range(0,dim):  
        for m in range(0,dim):  
            temp[i,:,:,:] += orb[m,i]*ao_integrals[m,:,:,:]  
        for j in range(0,dim):  
            for n in range(0,dim):  
                temp2[i,j,:,:] += orb[n,j]*temp[i,n,:,:]  
            for k in range(0,dim):  
                for o in range(0,dim):  
                    temp3[i,j,k,:] += orb[o,k]*temp2[i,j,o,:]  
                for l in range(0,dim):  
                    for p in range(0,dim):  
                        mo_integrals[i,j,k,l] += orb[p,l]*temp3[i,j,k,p]
    return mo_integrals

def get_full_h(hcore, fock, mo_integrals, perm, en_nuc):
    '''
    Arguments:
        hcore, fock - respective matrices in the molecular orbital basis
        mo_integrals - 2-electron integrals in the molecular orbital basis
        perm - list of all state sin the Hilbert space
        en_nuc - nuclear repulsion
    Returns:
        hamiltonian - full Hamiltonian in the Hilbert space
    '''
    n = len(perm)
    #print(mo_integrals)
    hamiltonian = np.zeros((n,n))
    hcore_en  = np.diagonal(hcore)
    for k in range(n):
        hamiltonian[k,k] = energy(hcore_en, perm[k][0]) + energy(hcore_en, perm[k][1])+ en_nuc
        alpha_pos = electron_pos(perm[k][0])
        beta_pos = electron_pos(perm[k][1])
        s = 0 
        for i in alpha_pos:
            for j in alpha_pos:
                s+= mo_integrals[i,i,j,j] - mo_integrals[i,j,i,j]
            for j in beta_pos:
                s+= mo_integrals[i,i,j,j]
        for i in beta_pos:
            for j in beta_pos:
                s+= mo_integrals[i,i,j,j] - mo_integrals[i,j,i,j]
            for j in alpha_pos:
                s+= mo_integrals[i,i,j,j]
        hamiltonian[k,k]+= s/2
    pairs = get_pairs(perm)
    for i in pairs:
        if i[2] + i[3] == 1:
            hamiltonian[i[0], i[1]] = fock[i[0],i[1]]
        if i[2] == 1 and i[3] == 1:
            hamiltonian[i[0],i[1]] = mo_integrals[i[4][1],i[4][0],i[5][1],i[5][0]]
    return hamiltonian

def project(matrix):
    '''
    Projects matrix out of the reference state.
    '''
    n = len(matrix)
    projector = np.identity(n)
    projector = np.delete(projector, 0, 1)
    m = projector.transpose().dot(matrix).dot(projector)
    return m

def first_order(h_0, h_1):
    '''
    Calculates the first order corrections to the wavefunction and energy, as these cannot be calculated using the iterative formula.
    '''
    n = len(h_0)
    eig = np.identity(n)
    psi_1 = np.zeros((n, 1))
    e_0 = h_0[0,0]
    for i in range(1, n):
        g = np.reshape(eig[:,i], (len(eig),1))
        #print(g)
        psi_1 += (h_1[0,i]/(e_0 - h_0[i,i]))*g
        #print(psi_1)
    e_1 = h_1[0,0]
    return psi_1, e_1

def mppt(h_tot, h_0, order, epsilon):
    '''
    Performs the perturbation theory.
    Arguments:
        h_tot - full Hamiltonian in the Hilbert space
        h_0 - unperturbed Hamiltonian, usually the Fock matrix
        order - the numberof corrections that will be calculated
        epsilon - level shift; use 0 if no level shift needed
    Returns:
        psi - list of all the eigenstate corrections, starting with the unperturbed eigenstate
        e - list of all energy corrections, starting with 0th order.
    '''
    # Get the dimensions of the FCI Hilbert space
    nfci = h_tot.shape[0]

    # Get perturbation Hamiltonian
    h_1 = h_tot - h_0

    #######################################################################
    # HGAB: Here is a modified recursive approach which uses the recursive
    #       formula for everything including |psi_1>, and also makes more 
    #       extensive use of numpy arrays.
    #######################################################################
    # Let's store the perturbed wave functions in columns of psi_pt
    psi_pt = np.zeros((nfci,order), dtype = complex)
    # Let's store the perturbed energies in another array
    e_pt   = np.zeros(order)

    # Define zeroth-order energy and wave function
    psi_pt[0,0] = 1
    e_pt[0] = np.real(np.einsum('j,jk,k',psi_pt[:,0],h_0,psi_pt[:,0]))

    # Define projector onto non-model space
    # Q = I - |0><0|
    Proj = np.identity(nfci) - psi_pt[:,[0]].dot(psi_pt[:,[0]].T)
    s,u  = np.linalg.eigh(Proj) # Here are our eigenvalues and eigenvectors of Proj
    proj_ind = np.argwhere(s==1).flatten() # We want the non-null eigenvectors
    Q = u[:,proj_ind] # We then extract out the corresponding eigenvalues

    # Let's now define the the inverse of the reference Hamiltonian once 
    # R = Q (Q^T ( H_0 - E_0) Q )^{-1} * Q.T 
    R = Q.dot(np.linalg.inv(Q.T.dot(h_0 - np.identity(nfci) * (e_pt[0]- epsilon*1j)).dot(Q))).dot(Q.T)

    # Now get first-order wave function from recursive formula
    for n in range(1,order):
        # Get the wave function correction
        psi_pt[:,n] = - (np.einsum('ij,jk,k->i',R,h_1,psi_pt[:,n-1]) - np.einsum('ij,jk,k->i',R,psi_pt[:,n-1:0:-1],e_pt[1:n]))
        # Get the energy correction
        e_pt[n]     = np.real(psi_pt[:,0].T.dot(h_1).dot(psi_pt[:,n-1]))
        #print('{:10d} {:20.10f} {:20.10f}'.format(n,e_pt[n],sum(e_pt[:n+1])))

    #######################################################################
    # Our final wave functions are stored in the columns of 
    #     psi_pt[:,:]
    # and our MPn corrections are stored in the elements of 
    #     e_pt[:]
    #######################################################################
    return psi_pt, e_pt

def srg_2(h_tot, h_0, s):
    h_1 = h_tot - h_0 
    nfci = h_tot.shape[0]
    e2 = 0
    for i in range(1,nfci):
        delta = h_0[0,0] - h_0[i,i]
        e2+= h_1[0,i]**2/delta *(1-np.exp(-2*s*(delta**2)))
    e2 = e2/2
    e = h_tot[0,0] + e2
    return e

def different_lambda(corr_list, l):
    '''
    Calculates series for different values of lambda i.e. multiplies every element with lambda^order.
    '''
    e =[]
    for i in range(len(corr_list)):
        e.append(corr_list[i]*(l**i))
    return e

def shanks_transformation(s_list):
    '''
    Performes a Shanks transformation on a list.
    '''
    n = len(s_list)
    s = 0 
    shank_list = []
    for i in range(1, n-1):
        t = (s_list[i+1]*s_list[i-1]-s_list[i]**2)/(s_list[i+1]-2*s_list[i]+s_list[i-1])
        shank_list.append(t)
    return shank_list

def convergence_radius(e_list):
    '''
    Given a list of energy corrections, plots Ek+1/Ek and the Shanks transformed series.
    '''
    ratio = []
    n = len(e_list)
    for i in range(n-1):
        ratio.append(abs(e_list[i+1]/e_list[i]))
    shanks = shanks_transformation(ratio)
    plt.plot(ratio,label = 'Initial series')
    plt.plot(shanks, label ='Shanks transformed series')
    plt.xlabel("k")
    plt.ylabel("E(k+1)/E(k)")
    plt.title("Determination of radius of convergence for H2 at bond length = 0.74 Angstrom")
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def plot_e(e):
    '''
    Plots the energy corrections - without the seroth order which is generally too large and wouldn't allow other features to be observed.
    '''
    e_0 = e[0]
    x = []
    e.pop(0)
    for i in range(len(e)):
        x.append(i+1)
    plt.plot(x,e)
    plt.xlabel("Correction order")
    plt.ylabel("Energy correction")
    plt.title("Energy corrections for H2 bond length = 0.74 Angstrom in 6-31G")
    plt.show()

def get_degeneracies(h_0, h_1):
    '''
    Gets all groups of degeneracies i.e. returns list of lists with all states that have 
    the same energy.
    '''
    nfci = h_0.shape[0]
    list_deg = []
    list_j = []
    for i in range(nfci):
        list_i = []
        if i in list_j:
            continue
        for j in range(i,nfci):
            if (abs(h_0[i,i] - h_0[j,j])<0.0001):
                list_i.append(j)
                list_j.append(j)
        if len(list_i)>1:
            list_deg.append(list_i)
    return list_deg

def get_deg_zeroth(h_0, h_1, list_deg):
    '''
    Gets the linear combinations of the degenerate eigenstates that diagonalise H1.
    '''
    n = len(list_deg)
    h_0_deg = np.zeros((n,n))
    h_1_deg = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            h_0_deg[i,j] = h_0[list_deg[i], list_deg[j]]
            h_1_deg[i,j] = h_1[list_deg[i], list_deg[j]]
    sol = np.linalg.eigh(h_1_deg)
    eigval = sol[0]
    eigvec = sol[1]
    return eigval, eigvec

def degenerate_pt(h_tot, h_0, order):
    '''
    Identifies all states that are degenerate in H0 and performs degenerate perturbation theory for
    all groups of degeneracy. Returns a a list with energy corrections for all states + list with all
    eigenstate corrections.
    '''
    h_1 = h_tot-h_0
    nfci = h_0.shape[0]
    e_pt = []
    psi_pt = []
    eigenvalues = np.identity(nfci)
    degeneracy_groups = get_degeneracies(h_0, h_1)
    print(degeneracy_groups)
    for i in degeneracy_groups:
        eigval, eigvec = get_deg_zeroth(h_0, h_1, i)
        n = len(i)
        for j in range(n):
            for k in range(n):
                eigenvalues[i[j],i[k]] = eigvec[j,k]
        h_tot_new = eigenvalues.T.dot(h_tot).dot(eigenvalues)
        h_0_new = eigenvalues.T.dot(h_0).dot(eigenvalues)
        #print('New h')
        #print(h_tot_new)
        #print(h_0_new)
        for ref_state in i:
            h0 = np.copy(h_0_new)
            htot = np.copy(h_tot_new)
            h0[:,[0, ref_state]] = h0[:,[ref_state, 0]]
            h0[[0, ref_state],:] = h0[[ref_state, 0],:]
            htot[:,[0, ref_state]] = htot[:,[ref_state, 0]]
            htot[[0, ref_state],:] = htot[[ref_state, 0],:]
            for degenerate_state in reversed(i):
                if degenerate_state!=ref_state:
                    h0 = np.delete(h0, degenerate_state, 0)
                    h0 = np.delete(h0, degenerate_state, 1)
                    htot = np.delete(htot, degenerate_state, 0)
                    htot = np.delete(htot, degenerate_state, 1)  
            print(ref_state)
            psi, e = mppt(htot, h0, order)
            print(e)
            psi_pt.append(psi)
            e_pt.append(list(e))
    return psi_pt, e_pt

def enpt(h_tot, order):
    '''
    Performs the perturbation theory with the Epstein-Nesbet partitioning.
    Arguments:
        h_tot - full Hamiltonian in the Hilbert space
        order - the numberof corrections that will be calculated
        epsilon - level shift
    Returns:
        psi - list of all the eigenstate corrections, starting with the unperturbed eigenstate
        e - list of all energy corrections, starting with 0th order.
    '''
    # Get the dimensions of the FCI Hilbert space
    nfci = h_tot.shape[0]

    # Get perturbation Hamiltonian
    d =  np.diag(h_tot)
    h_0 = np.diagflat(d)
    h_1 = h_tot - h_0

    #######################################################################
    # HGAB: Here is a modified recursive approach which uses the recursive
    #       formula for everything including |psi_1>, and also makes more 
    #       extensive use of numpy arrays.
    #######################################################################
    # Let's store the perturbed wave functions in columns of psi_pt
    psi_pt = np.zeros((nfci,order), dtype = complex)
    # Let's store the perturbed energies in another array
    e_pt   = np.zeros(order)

    # Define zeroth-order energy and wave function
    psi_pt[0,0] = 1
    e_pt[0] = np.real(np.einsum('j,jk,k',psi_pt[:,0],h_0,psi_pt[:,0]))

    # Define projector onto non-model space
    # Q = I - |0><0|
    Proj = np.identity(nfci) - psi_pt[:,[0]].dot(psi_pt[:,[0]].T)
    s,u  = np.linalg.eigh(Proj) # Here are our eigenvalues and eigenvectors of Proj
    proj_ind = np.argwhere(s==1).flatten() # We want the non-null eigenvectors
    Q = u[:,proj_ind] # We then extract out the corresponding eigenvalues

    # Let's now define the the inverse of the reference Hamiltonian once 
    # R = Q (Q^T ( H_0 - E_0) Q )^{-1} * Q.T 
    R = Q.dot(np.linalg.inv(Q.T.dot(h_0 - np.identity(nfci) * e_pt[0]).dot(Q))).dot(Q.T)

    # Now get first-order wave function from recursive formula
    for n in range(1,order):
        # Get the wave function correction
        psi_pt[:,n] = - (np.einsum('ij,jk,k->i',R,h_1,psi_pt[:,n-1]) - np.einsum('ij,jk,k->i',R,psi_pt[:,n-1:0:-1],e_pt[1:n]))
        # Get the energy correction
        e_pt[n]     = np.real(psi_pt[:,0].T.dot(h_1).dot(psi_pt[:,n-1]))
        #print('{:10d} {:20.10f} {:20.10f}'.format(n,e_pt[n],sum(e_pt[:n+1])))

    #######################################################################
    # Our final wave functions are stored in the columns of 
    #     psi_pt[:,:]
    # and our MPn corrections are stored in the elements of 
    #     e_pt[:]
    #######################################################################
    return psi_pt, e_pt

def get_dyson(mo_fock, mo_integrals, perm):
    fock_diag  = np.diag(mo_fock)
    nmo = len(fock_diag)
    dyson_diag = np.copy(fock_diag)

    # Initialise many-body Fock matrix
    n = len(perm)
    dyson        = np.zeros((n,n))
    ref = perm[0]
    ref_a = np.asarray(ref[0]) 
    ref_b = np.asarray(ref[1])

    # Find occupied orbitals 
    occa = list(np.argwhere(ref_a).flatten())
    occb = list(np.argwhere(ref_b).flatten())
    occ = occa + occb
    orb = range(nmo)
    neocc = []
    for i in orb:
        if i not in occ:
            neocc.append(i)
    for i in orb:
        for a in occ:
            for p in neocc:
                for b in occ:
                    dyson_diag[i] +=((mo_integrals[i,a,p,b]-mo_integrals[i,p,a,b])**2)/(2*(fock_diag[i]+fock_diag[p]-fock_diag[a]-fock_diag[b]))
    for i in orb:
        for p in neocc:
            for a in occ:
                for q in neocc:
                    dyson_diag[i] +=((mo_integrals[i,p,a,q]-mo_integrals[i,a,p,q])**2)/(2*(fock_diag[i]+fock_diag[a]-fock_diag[p]-fock_diag[q]))

    for ibra, perm_bra in enumerate(perm):
        # Get this permutation
        pbra_a = np.asarray(perm_bra[0]) 
        pbra_b = np.asarray(perm_bra[1])

        # Find occupied orbitals 
        occa = np.argwhere(pbra_a).flatten()
        occb = np.argwhere(pbra_b).flatten()

        # Get diagonal Fock terms
        dyson[ibra,ibra] = np.sum(dyson_diag[occa]) + np.sum(dyson_diag[occb])
    dyson_mo = np.diagflat(dyson_diag)
    return dyson, dyson_mo

def coupled_pt(h_tot, h_0, order, state):
    '''
    Identifies all states that are degenerate in H0 and performs degenerate perturbation theory for
    all groups of degeneracy. Returns a a list with energy corrections for all states + list with all
    eigenstate corrections.
    '''
    h_1 = h_tot-h_0
    nfci = h_0.shape[0]
    eigenvalues = np.identity(nfci)
    group = [0, state]
    eigval, eigvec = get_deg_zeroth(h_0, h_1, group)
    print(eigval)
    print(eigvec)
    n = len(group)
    for j in range(n):
        for k in range(n):
            eigenvalues[group[j],group[k]] = eigvec[j,k]
    h_tot_new = eigenvalues.T.dot(h_tot).dot(eigenvalues)
    h_0_new = eigenvalues.T.dot(h_0).dot(eigenvalues)
    h0 = np.copy(h_0_new)
    htot = np.copy(h_tot_new)
    h0 = np.delete(h0, 0, 0)
    h0 = np.delete(h0, 0, 1)
    htot = np.delete(htot, 0, 0)
    htot = np.delete(htot, 0, 1)  
    h0[:,[0, state-1]] = h0[:,[state-1, 0]]
    h0[[0, state-1],:] = h0[[state-1, 0],:]
    htot[:,[0, state-1]] = htot[:,[state-1, 0]]
    htot[[0, state-1],:] = htot[[state-1, 0],:]
    psi, e = mppt(htot, h0, order, 0)
    return psi, e
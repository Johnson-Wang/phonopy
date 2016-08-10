# Copyright (C) 2011 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import sys
from phonopy.structure.cells import get_reduced_bases
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors
from phonopy.structure.atoms import Atoms

def get_force_constants(set_of_forces,
                        symmetry,
                        supercell,
                        atom_list=None,
                        decimals=None):

    if atom_list==None:
        force_constants = run_force_constants(
            supercell,
            symmetry,
            set_of_forces,
            range(supercell.get_number_of_atoms()))
    else:
        force_constants = run_force_constants(supercell,
                                              symmetry,
                                              set_of_forces,
                                              atom_list)

    if decimals:
        return force_constants.round(decimals=decimals)
    else:
        return force_constants

def cutoff_force_constants(force_constants,
                           supercell,
                           cutoff_radius,
                           symprec=1e-5):
    num_atom = supercell.get_number_of_atoms()
    reduced_bases = get_reduced_bases(supercell.get_cell(), symprec)
    positions = np.dot(supercell.get_positions(),
                       np.linalg.inv(reduced_bases))
    for i in range(num_atom):
        pos_i = positions[i]
        for j in range(num_atom):
            pos_j = positions[j]
            min_distance = get_shortest_distance_in_PBC(pos_i,
                                                        pos_j,
                                                        reduced_bases)
            if min_distance > cutoff_radius:
                force_constants[i, j] = 0.0

def get_shortest_distance_in_PBC(pos_i, pos_j, reduced_bases):
    distances = []
    for k in (-1, 0, 1):
        for l in (-1, 0, 1):
            for m in (-1, 0, 1):
                diff = pos_j + np.array([k, l, m]) - pos_i
                distances.append(np.linalg.norm(np.dot(diff, reduced_bases)))
    return np.min(distances)
                        

def symmetrize_force_constants(force_constants, iteration=3):
    for i in range(iteration):
        set_permutation_symmetry(force_constants)
        set_translational_invariance(force_constants)

def run_force_constants(supercell,
                        symmetry,
                        set_of_forces,
                        atom_list):
    """
    Bare force_constants is returned.

    Force constants, Phi, are calculated from sets for forces, F, and
    atomic displacement, d:
      Phi = -F / d
    This is solved by matrix pseudo-inversion.
    Crsytal symmetry is included when creating F and d matrices.

    force_constants[ i, j, a, b ]
      i: Atom index of finitely displaced atom.
      j: Atom index at which force on the atom is measured.
      a, b: Cartesian direction indices = (0, 1, 2) for i and j, respectively
    """
    
    force_constants = np.zeros((supercell.get_number_of_atoms(),
                                supercell.get_number_of_atoms(),
                                3, 3), dtype='double')

    # Fill force_constants[ displaced_atoms, all_atoms_in_supercell ]
    atom_list_done = _get_force_constants_disps(force_constants,
                                                supercell,
                                                set_of_forces,
                                                symmetry)

    # Distribute non-equivalent force constants to those equivalent
    symprec = symmetry.get_symmetry_tolerance()
    rotations = symmetry.get_symmetry_operations()['rotations']
    trans = symmetry.get_symmetry_operations()['translations']
    positions = supercell.get_scaled_positions()
    lattice = supercell.get_cell().T
    distribute_force_constants(force_constants,
                               atom_list,
                               atom_list_done,
                               lattice,
                               positions,
                               rotations,
                               trans,
                               symprec)

    return force_constants

def distribute_force_constants(force_constants,
                               atom_list,
                               atom_list_done,
                               lattice,
                               positions,
                               rotations,
                               trans,
                               symprec,
                               is_symmetrize=False):
    for atom_disp in atom_list:
        if atom_disp in atom_list_done:
            continue
            
        if is_symmetrize:
            map_atom_disps, map_syms = get_all_atom_mappings_by_symmetry(
                atom_list_done,
                atom_disp,
                rotations, 
                trans,
                positions,
                symprec)
        
            for i, pos_i in enumerate(positions):
                for map_atom_disp, map_sym in zip(map_atom_disps, map_syms):
                    # L R L^-1
                    rot_cartesian = np.double(similarity_transformation(
                        lattice, rotations[map_sym]))
                    _distribute_fc2_part(force_constants,
                                        positions,
                                        atom_disp,
                                        map_atom_disp,
                                        i,
                                        rot_cartesian,
                                        rotations[map_sym],
                                        trans[map_sym],
                                        symprec)
        
                force_constants[atom_disp, i] /= len(map_atom_disps)
        else:
            map_atom_disp, map_sym = get_atom_mapping_by_symmetry(
                atom_list_done,
                atom_disp,
                rotations, 
                trans,
                positions,
                symprec)
    
            # L R L^-1
            rot_cartesian = np.double(similarity_transformation(
                    lattice, rotations[map_sym]))
    
            _distribute_fc2_part(force_constants,
                                 positions,
                                 atom_disp,
                                 map_atom_disp,
                                 rot_cartesian,
                                 rotations[map_sym],
                                 trans[map_sym],
                                 symprec)

def _distribute_fc2_part(force_constants,
                         positions,
                         atom_disp,
                         map_atom_disp,
                         rot_cartesian,
                         r,
                         t,
                         symprec):

    try:
        raise ImportError
        import phonopy._phonopy as phonoc
        phonoc.distribute_fc2(force_constants,
                              positions,
                              atom_disp,
                              map_atom_disp,
                              rot_cartesian,
                              r,
                              t,
                              symprec)
    except ImportError:
        for i, pos_i in enumerate(positions):
            rot_pos = np.dot(pos_i, r.T) + t
            rot_atom = -1
            for j, pos_j in enumerate(positions):
                diff = pos_j - rot_pos
                if (abs(diff - np.rint(diff)) < symprec).all():
                    rot_atom = j
                    break
        
            if rot_atom < 0:
                print 'Input forces are not enough to calculate force constants,'
                print 'or something wrong (e.g. crystal structure does not match).'
                raise ValueError
            
            # R^-1 P R (inverse transformation)
            # f1 = np.dot(np.kron(rot_cartesian.T, rot_cartesian.T), force_constants[map_atom_disp, rot_atom].flatten()).reshape(3,3)
            # f2=  similarity_transformation(rot_cartesian.T,force_constants[map_atom_disp,rot_atom])
            # diff=np.abs(f1-f2)
            # assert diff.max()<1e-7

            force_constants[atom_disp, i] += similarity_transformation(
                rot_cartesian.T,
                force_constants[map_atom_disp,
                                rot_atom])
    
    
def get_atom_mapping_by_symmetry(atom_list_done,
                                 atom_number,
                                 rotations,
                                 translations,
                                 positions,
                                 symprec=1e-5):
    """
    Find a mapping from an atom to an atom in the atom list done.
    """

    for i, (r, t) in enumerate(zip(rotations, translations)):
        rot_pos = np.dot(positions[atom_number], r.T) + t
        for j in atom_list_done:
            diff = positions[j] - rot_pos
            if (abs(diff - np.rint(diff)) < symprec).all():
                return j, i

    print 'Input forces are not enough to calculate force constants,'
    print 'or something wrong (e.g. crystal structure does not match).'
    raise ValueError

def get_all_atom_mappings_by_symmetry(atom_list_done,
                                      atom_number,
                                      rotations,
                                      translations,
                                      positions,
                                      symprec=1e-5):
    """
    Find mappings from an atom to atoms in the atom list done.
    """

    map_atoms = []
    map_syms = []
    for i, (r, t) in enumerate(zip(rotations, translations)):
        rot_pos = np.dot(positions[atom_number], r.T) + t
        for j in atom_list_done:
            diff = positions[j] - rot_pos
            if (abs(diff - np.rint(diff)) < symprec).all():
                map_atoms.append(j)
                map_syms.append(i)
                break

    if len(map_atoms) == 0:
        print 'Input forces are not enough to calculate force constants,'
        print 'or something wrong (e.g. crystal structure does not match).'
        raise ValueError

    return map_atoms, map_syms

def _get_force_constants_disps(force_constants,
                               supercell,
                               set_of_forces,
                               symmetry):
    """
    Phi = -F / d
    """
    
    symprec = symmetry.get_symmetry_tolerance()
    disp_atom_list = np.unique(
        [forces.get_atom_number() for forces in set_of_forces])

    for disp_atom_number in disp_atom_list:
        disps = []
        sets_of_forces = []

        for forces in set_of_forces:
            if forces.get_atom_number() != disp_atom_number:
                continue

            disps.append(forces.get_displacement())
            sets_of_forces.append(forces.get_forces())

        site_symmetry = symmetry.get_site_symmetry(disp_atom_number)

        solve_force_constants(force_constants,
                              disp_atom_number,
                              disps,
                              sets_of_forces,
                              supercell,
                              site_symmetry,
                              symprec)

    return disp_atom_list


def solve_force_constants(force_constants,
                          disp_atom_number,
                          displacements,
                          sets_of_forces,
                          supercell,
                          site_symmetry,
                          symprec):
    lat = supercell.get_cell().T
    positions = supercell.get_scaled_positions()
    pos_center = positions[disp_atom_number].copy()
    positions -= pos_center
    rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                 site_symmetry,
                                                 symprec)
    site_sym_cart = [similarity_transformation(lat, sym)
                     for sym in site_symmetry]
    rot_disps = get_rotated_displacement(displacements, site_sym_cart)
    inv_displacements = np.linalg.pinv(rot_disps)

    for i in range(supercell.get_number_of_atoms()):
        combined_forces = []
        for forces in sets_of_forces:
            combined_forces.append(
                get_rotated_forces(forces[rot_map_syms[:, i]],
                                   site_sym_cart))

        combined_forces = np.reshape(combined_forces, (-1, 3))
        force_constants[disp_atom_number, i] = -np.dot(
            inv_displacements, combined_forces)

def get_positions_sent_by_rot_inv(positions,
                                  site_symmetry,
                                  symprec):
    rot_map_syms = []
    for sym in site_symmetry:
        rot_map = np.zeros(len(positions), dtype='intc')
        rot_pos = np.dot(positions, sym.T)
        is_found = False
        for i, rot_pos_i in enumerate(rot_pos):
            diff = positions - rot_pos_i
            j = np.nonzero((
                    np.abs(diff - np.rint(diff)) < symprec).all(axis=1))[0]
            rot_map[j] = i

        rot_map_syms.append(rot_map)

    return np.intc(rot_map_syms)

def get_rotated_displacement(displacements, site_sym_cart):
    rot_disps = []
    for u in displacements:
        rot_disps.append([np.dot(sym, u) for sym in site_sym_cart])
    return np.reshape(rot_disps, (-1, 3))

def get_rotated_forces(forces_syms, site_sym_cart):
    rot_forces = []
    for forces, sym_cart in zip(forces_syms, site_sym_cart):
        rot_forces.append(np.dot(forces, sym_cart.T))

    return rot_forces

def set_tensor_symmetry(force_constants,
                        lattice,
                        positions,
                        rotations,
                        translations,
                        symprec):
    """
    Full force constants are symmetrized using crystal symmetry.
    This method extracts symmetrically equivalent sets of atomic pairs and
    take sum of their force constants and average the sum.
    
    Since get_force_constants_disps may include crystal symmetry, this method
    is usually meaningless.
    """

    fc_bak = force_constants.copy()

    # Create mapping table between an atom and the symmetry operated atom
    # map[ i, j ]
    # i: atom index
    # j: operation index
    mapping = []
    for pos_i in positions:
        map_local = []
        for rot, trans in zip(rotations, translations):
            rot_pos = np.dot(pos_i, rot.T) + trans
            for j, pos_j in enumerate(positions):
                diff = pos_j - rot_pos
                if (abs(diff -diff.round()) < symprec).all():
                    map_local.append(j)
                    break
        mapping.append(map_local)
    mapping = np.array(mapping)

    # Look for the symmetrically equivalent force constant tensors
    for i, pos_i in enumerate(positions):
        for j, pos_j in enumerate(positions):
            tmp_fc = np.zeros((3, 3), dtype='double')
            for k, rot in enumerate(rotations):
                cart_rot = similarity_transformation(lattice, rot)

                # Reverse rotation of force constant is summed
                tmp_fc += similarity_transformation(cart_rot.T,
                                                    fc_bak[mapping[i, k],
                                                           mapping[j, k]])
            # Take average and set to new force cosntants
            force_constants[i, j] = tmp_fc / len(rotations)

def set_translational_rotational_invariance(force_constants, supercell):
    """
    Translational invariance and rotational invariance are imposed. The algorithm
    is adopted from J. Carrete, W. Li, et al., Materials Research Letters 0, 1 (2016).
    The force constants are transformed from Cartesian coordinate (3N x 3N) to internal coordinates ((3N-6) x (3N-6))
    using lst method and then transformed back.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    import scipy
    import matplotlib.pyplot as plt
    pos = supercell.get_positions()
    natom = supercell.get_number_of_atoms()
    Jacobian = np.zeros((natom, 3, natom * 3 - 6), dtype='double')
    unique_atoms = np.unique(supercell.s2u_map)
    i0 = unique_atoms[0]
    p0 = pos[i0]
    rest = range(1,natom)
    if len(unique_atoms) >= 2:
        i1 = unique_atoms[1]
        p1 = pos[i1]
        rest.remove(i1)
        if len(unique_atoms) >= 3:
            i2 = unique_atoms[2]
            p2 = pos[i2]
            rest.remove(i2)
    else:
        i1 = 1
        p1=pos[i1]
        rest.remove(i1)
    if len(rest) == natom - 2:
        for i in rest:
            p2 = pos[i]
            i2= i
            if not (np.abs(np.cross(p0-p1, p0-p2)) < 1e-7).all(): # colinear
                rest.remove(i)
                break

    assert len(get_equivalent_smallest_vectors_cart(i1, i0, supercell)) == 1
    assert len(get_equivalent_smallest_vectors_cart(i2, i0, supercell)) == 1
    assert len(get_equivalent_smallest_vectors_cart(i2, i1, supercell)) == 1

    d10 = get_equivalent_smallest_vectors_cart(i1, i0, supercell)[0]
    d20 = get_equivalent_smallest_vectors_cart(i2, i0, supercell)[0]
    d21 = get_equivalent_smallest_vectors_cart(i2, i1, supercell)[0]
    dist = lambda x: np.sqrt(np.sum(x ** 2))
    Jacobian[i0, :, 0] = d10 / dist(d10)
    Jacobian[i1, :, 0] = -Jacobian[i0, :, 0]
    Jacobian[i0, :, 1] = d20 / dist(d20)
    Jacobian[i2, :, 1] = - Jacobian[i0, :, 1]
    Jacobian[i1, :, 2] = d21 / dist(d21)
    Jacobian[i2, :, 2] = - Jacobian[i1, :, 2]
    I = np.eye(3)
    for index, j in enumerate(rest):
        i = index + 1
        dj0 = get_equivalent_smallest_vectors_cart(j, i0, supercell)[0]
        dj1 = get_equivalent_smallest_vectors_cart(j, i1, supercell)[0]
        dj2 = get_equivalent_smallest_vectors_cart(j, i2, supercell)[0]


        if np.abs(np.dot(dj0, np.cross(dj1, dj2))) < 1e-7: # coplanar
            scale20 = np.cross(np.cross(d20, d10), d20)
            s20 = np.dot(dj0, scale20) * scale20 / np.dot(scale20, scale20)
            scale10 = np.cross(np.cross(d10, d21), d10)
            s10 = np.dot(dj0, scale10) * scale10 / np.dot(scale10, scale10)
            scale21 = np.cross(np.cross(d21, d20), d21)
            s21 = np.dot(dj0, scale21) * scale21 / np.dot(scale21, scale21)

            ds10d0 = -2 * np.outer(d21, d10) + I * np.dot(d10, d21) + np.outer(d10, d21)
            ds10d1 = - np.eye(3) * np.dot(d10, d10) + 2 * np.outer(d21, d10) - I * np.dot(d10, d21) - np.outer(d10, d21 - d10)
            ds10d2 = I * np.dot(d10, d10) - np.outer(d10, d10)
            ds20d0 = -I * np.dot(d20, d20) - 2 * np.outer(d10, d20) + I * np.dot(d20, d10) + np.outer(d20, d20 + d10)
            ds20d1 = I * np.dot(d20, d20) - np.outer(d20, d20)
            ds20d2 = 2 * np.outer(d10, d20) - I * np.dot(d20, d10) - np.outer(d20, d10)
            ds21d0 = -I * np.dot(d21, d21) + np.outer(d21, d21)
            ds21d1 = -2 * np.outer(d20, d21) + I * np.dot(d21, d20) + np.outer(d21, d20)
            ds21d2 = I * np.dot(d21, d21) + 2 * np.outer(d20, d21) - I * np.dot(d21, d20) - np.outer(d21, d21 + d20)


            Jacobian[j,:, i*3] = dj0 / dist(dj0)
            Jacobian[i0,:, i*3] = - Jacobian[j,:, i*3]

            d10dj = scale10 / dist(scale10)
            d10d0 = (-scale10 + np.dot(dj0-s10, ds10d0)) / dist(scale10)
            d10d1 = np.dot(dj0-s10, ds10d1) / dist(scale10)
            d10d2 = np.dot(dj0-s10, ds10d2) / dist(scale10)
            d20dj = scale20/ dist(scale20)
            d20d0 = (-scale20 + np.dot(dj0-s20, ds20d0)) / dist(scale20)
            d20d1 = np.dot(dj0-s20, ds20d1) / dist(scale20)
            d20d2 = np.dot(dj0-s20, ds20d2) / dist(scale20)
            d21dj = scale21 / dist(scale21)
            d21d0 = (-scale21 + np.dot(dj0-s21, ds21d0)) / dist(scale21)
            d21d1 =  np.dot(dj0-s21, ds21d1) / dist(scale21)
            d21d2 =  np.dot(dj0-s21, ds21d2) / dist(scale21)

            if not (np.abs(np.dot(dj0, scale10)) < 1e-8 or np.abs(np.dot(dj0, scale20)) < 1e-8):
                Jacobian[[i0,i1,i2,j],:,i*3+1] = np.array([d10d0, d10d1, d10d2, d10dj])
                Jacobian[[i0,i1,i2,j],:,i*3+2] = np.array([d20d0, d20d1, d20d2, d20dj])

            elif not (np.abs(np.dot(dj0, scale10)) < 1e-8 or np.abs(np.dot(dj0, scale21)) < 1e-8):
                Jacobian[[i0,i1,i2,j],:,i*3+1] = np.array([d10d0, d10d1, d10d2, d10dj])
                Jacobian[[i0,i1,i2,j],:,i*3+2] = np.array([d21d0, d21d1, d21d2, d21dj])
            elif not (np.abs(np.dot(dj0, scale21)) < 1e-8 or np.abs(np.dot(dj0, scale20)) < 1e-8):
                Jacobian[[i0,i1,i2,j],:,i*3+1] = np.array([d21d0, d21d1, d21d2, d21dj])
                Jacobian[[i0,i1,i2,j],:,i*3+2] = np.array([d20d0, d20d1, d20d2, d20dj])
            else:
                print "Error"
        else:
            Jacobian[j, :, i*3] = dj0 / dist(dj0)
            Jacobian[i0,:,i*3] = - Jacobian[j, :, i*3]
            Jacobian[j,:,i*3+1] = dj1 / dist(dj1)
            Jacobian[i1,:,i*3+1] = - Jacobian[j,:,i*3+1]
            Jacobian[j,:,i*3+2] = dj2 / dist(dj2)
            Jacobian[i2,:,i*3+2] = - Jacobian[j,:,i*3+2]

    # unit_atoms = np.unique(supercell.get_supercell_to_unitcell_map())
    # unit_coords = np.concatenate([i * 3 + np.arange(3) for i in unit_atoms])

    # J0 = Jacobian[unit_atoms].reshape(-1, natom * 3 - 6)
    J = Jacobian.reshape(-1, natom * 3 - 6)

    non_zero_pos = np.where(np.abs(J)>1e-8)
    plt.matshow(J)
    plt.axis('off')
    plt.savefig('Jacobian.pdf')
    # J0_sparse = coo_matrix((np.extract(J0!=0, Jacobian), np.where(J0!=0)), shape = J0.shape)
    J_sparse = coo_matrix((J[non_zero_pos], non_zero_pos), shape = J.shape)
    # eta0_sparse = scipy.sparse.kron(J0_sparse, J_sparse)
    eta_sparse = scipy.sparse.kron(J_sparse, J_sparse)
    fc = force_constants.swapaxes(1,2).flatten()
    # fc0 = force_constants[unit_atoms].swapaxes(1,2).flatten()
    force_constants_internal = lsqr(eta_sparse, fc)
    force_constants_new = eta_sparse.dot(force_constants_internal[0])
    force_constants_new = force_constants_new.reshape((natom, 3, natom, 3)).swapaxes(1,2)
    plt.figure()
    plt.scatter(force_constants.flatten(), force_constants_new.flatten(), color = 'red', s=4)
    plt.plot(np.array([force_constants.min(), force_constants.max()]), np.array([force_constants.min(), force_constants.max()]), color='blue')
    plt.savefig('fc2_compare.pdf')
    plt.figure()
    return force_constants_new

def set_translational_invariance(force_constants):
    """
    Translational invariance is imposed.  This is quite simple
    implementation, which is just taking sum of the force constants in
    an axis and an atom index. The sum has to be zero due to the
    translational invariance. If the sum is not zero, this error is
    uniformly subtracted from force constants.
    """
    set_translational_invariance_per_index(force_constants, index=0)
    set_translational_invariance_per_index(force_constants, index=1)

def set_translational_invariance_per_index(force_constants, index=0):
    if index == 0:
        for i in range(force_constants.shape[1]):
            for j in range(force_constants.shape[2]):
                for k in range(force_constants.shape[3]):
                    force_constants[:, i, j, k] -= np.sum(
                        force_constants[:, i, j, k]) / force_constants.shape[0]
    elif index == 1:
        for i in range(force_constants.shape[0]):
            for j in range(force_constants.shape[2]):
                for k in range(force_constants.shape[3]):
                    force_constants[i, :, j, k] -= np.sum(
                        force_constants[i, :, j, k]) / force_constants.shape[1]


def set_permutation_symmetry(force_constants):
    """
    Phi_ij_ab = Phi_ji_ba
    
    i, j: atom index
    a, b: Cartesian axis index

    This is not necessary for harmonic phonon calculation because this
    condition is imposed when making dynamical matrix Hermite in
    dynamical_matrix.py.
    """
    fc_copy = force_constants.copy()
    for i in range(force_constants.shape[0]):
        for j in range(force_constants.shape[1]):
            force_constants[i, j] = (force_constants[i, j] +
                                     fc_copy[j, i].T) / 2

# Helper methods
def get_equivalent_smallest_vectors_cart(atom_number1_new,
                                         atom_number_orig,
                                         supercell,
                                         symprec=1e-5):
    distances = []
    differences = []
    reduced_bases = get_reduced_bases(supercell.get_cell(), symprec)
    positions = np.dot(supercell.get_positions(),
                       np.linalg.inv(reduced_bases))

    # Atomic positions are confined into the lattice made of reduced bases.
    for pos in positions:
        pos -= np.rint(pos)

    p_pos = positions[atom_number_orig]
    s_pos = positions[atom_number1_new]
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                # The vector arrow is from the atom in primitive to
                # the atom in supercell cell plus a supercell lattice
                # point. This is related to determine the phase
                # convension when building dynamical matrix.
                diff = s_pos + np.array([i, j, k]) - p_pos
                differences.append(diff)
                vec = np.dot(diff, reduced_bases)
                distances.append(np.linalg.norm(vec))

    minimum = min(distances)
    smallest_vectors = []
    for i in range(27):
        if abs(minimum - distances[i]) < symprec:
            smallest_vectors.append(np.dot(differences[i], reduced_bases))
    return smallest_vectors

def rotational_invariance(force_constants,
                          supercell,
                          primitive,
                          symprec=1e-5):
    """
    *** Under development ***
    Just show how force constant is close to the condition of rotational invariance,
    """
    print "Check rotational invariance ..."

    fc = force_constants 
    p2s = primitive.get_primitive_to_supercell_map()

    abc = "xyz"

    # Huang = np.zeros((3,3,3,3), dtype='double')
    for pi, p in enumerate(p2s):
        for i in range(3):
            mat = np.zeros((3, 3), dtype='double')
            for s in range(supercell.get_number_of_atoms()):
                vecs = np.array(get_equivalent_smallest_vectors(
                        s, p, supercell, primitive.get_cell(), symprec))
                m = len(vecs)
                v = np.dot(vecs[:,:].sum(axis=0) / m, primitive.get_cell())
                for j in range(3):
                    for k in range(3):
                        mat[j, k] += (fc[p, s, i, j] * v[k] -
                                      fc[p, s, i, k] * v[j])

            print "Atom %d %s" % (p+1, abc[i])
            for vec in mat:
                print "%10.5f %10.5f %10.5f" % tuple(vec)

        # for s in range(supercell.get_number_of_atoms()):
        #     vecs = np.array(get_equivalent_smallest_vectors(
        #         s, p, supercell, primitive.get_cell(), symprec))
        #     m = len(vecs)
        #     # v = np.dot(vecs[:,:].sum(axis=0) / m, primitive.get_cell())
        #     v = np.dot(vecs[:,:], primitive.get_cell())
        #     for a,b,c,d in np.ndindex((3,3,3,3)):
        #         torque1 = 0; torque2 = 0; torque3 = 0; torque4 = 0
        #         for i, j in np.ndindex((m,m)):
        #             torque1 += fc[p, s, a, c] * v[i, b] * v[j, d]
        #             torque2 += - fc[p, s, b, c] * v[i,a] * v[j, d]
        #             torque3 += fc[p, s, b, d] * v[i, a] * v[j, c]
        #             torque4 += - fc[p, s, a, d] * v[i, b]* v[j, c]
        #         Huang[a,b,c,d] += (torque1 + torque2 + torque3 + torque4) / m**2
            # px1 = np.einsum('ac, b->abc', fc[p, s], v)
            # px2 = np.einsum('bc, a->abc', fc[p, s], v)
            # torque1 = np.einsum('abc, d -> abcd', px1-px2, v)
            # px3 = np.einsum('bd, a->abd', fc[p,s], v)
            # px4 = np.einsum('ad, b->abd', fc[p,s], v)
            # torque2 = np.einsum('abd, c -> abcd', px3 - px4, v)
            # Huang += torque1 + torque2
    # print 'Huang invariance condition'
    # Huang = Huang.reshape(9,9)
    # for i in range(9):
    #     print "%10.5f " * 9 %tuple(Huang[i])


def force_constants_log(force_constants):
    fs = force_constants
    for i, fs_i in enumerate(fs):
        for j, fs_j in enumerate(fs_i):
            for v in fs_j:
                print "force constant (%d - %d): %10.5f %10.5f %10.5f" % (
                    i + 1, j + 1, v[0], v[1], v[2])


def similarity_transformation(rot, mat):
    """ R x M x R^-1 """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))

def show_drift_force_constants(force_constants, name="force constants"):
    num_atom = force_constants.shape[0]
    maxval1 = 0
    maxval2 = 0
    for i, j, k in list(np.ndindex((num_atom, 3, 3))):
        val1 = force_constants[:, i, j, k].sum()
        val2 = force_constants[i, :, j, k].sum()
        if abs(val1) > abs(maxval1):
            maxval1 = val1
        if abs(val2) > abs(maxval2):
            maxval2 = val2
    print ("max drift of %s:" % name), maxval1, maxval2
        

import sys
import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation, set_permutation_symmetry, distribute_force_constants, solve_force_constants, get_rotated_displacement, get_positions_sent_by_rot_inv, set_translational_invariance
from anharmonic.phonon3.displacement_fc3 import get_reduced_site_symmetry, get_bond_symmetry
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors
from anharmonic.file_IO import write_fc2_dat

def get_fc3(supercell,
            disp_dataset,
            fc2,
            symmetry,
            is_translational_symmetry=False,
            is_permutation_symmetry=False,
            verbose=False):
    num_atom = supercell.get_number_of_atoms()
    fc3 = np.zeros((num_atom, num_atom, num_atom, 3, 3, 3), dtype='double')


    _get_fc3_least_atoms(fc3,
                         supercell,
                         disp_dataset,
                         fc2,
                         symmetry,
                         is_translational_symmetry,
                         is_permutation_symmetry,
                         verbose)

    if verbose:
        print "(Copying fc3...)"



    first_disp_atoms = np.unique(
        [x['number'] for x in disp_dataset['first_atoms']])
    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']
    symprec = symmetry.get_symmetry_tolerance()
    lattice = supercell.get_cell().T
    positions = supercell.get_scaled_positions()

    distribute_fc3(fc3,
                   first_disp_atoms,
                   lattice,
                   positions,
                   rotations,
                   translations,
                   symprec,
                   verbose)

    if is_translational_symmetry:
        set_translational_invariance_fc3_per_index(fc3)

    return fc3

def distribute_fc3(fc3,
                   first_disp_atoms,
                   lattice,
                   positions,
                   rotations,
                   translations,
                   symprec,
                   verbose):
    num_atom = len(positions)

    for i in range(num_atom):
        if i in first_disp_atoms:
            continue

        for atom_index_done in first_disp_atoms:
            rot_num = get_atom_mapping_by_symmetry(positions,
                                                   i,
                                                   atom_index_done,
                                                   rotations,
                                                   translations,
                                                   symprec)
            if rot_num > -1:
                i_rot = atom_index_done
                rot = rotations[rot_num]
                trans = translations[rot_num]
                break

        if rot_num < 0:
            print "Position or symmetry may be wrong."
            raise ValueError

        if verbose > 2:
            print "  [ %d, x, x ] to [ %d, x, x ]" % (i_rot + 1, i + 1)
            sys.stdout.flush()

        atom_mapping = np.zeros(num_atom, dtype='intc')
        for j in range(num_atom):
            atom_mapping[j] = get_atom_by_symmetry(positions,
                                                   rot,
                                                   trans,
                                                   j,
                                                   symprec)
        rot_cart_inv = np.double(
            similarity_transformation(lattice, rot).T.copy())

        try:
            import anharmonic._phono3py as phono3c
            phono3c.distribute_fc3(fc3,
                                   i,
                                   atom_mapping,
                                   rot_cart_inv)
        
        except ImportError:
            for j in range(num_atom):
                j_rot = atom_mapping[j]
                for k in range(num_atom):
                    k_rot = atom_mapping[k]
                    fc3[i, j, k] = third_rank_tensor_rotation(
                        rot_cart_inv, fc3[i_rot, j_rot, k_rot])

def set_permutation_symmetry_fc3(fc3):
    num_atom = fc3.shape[0]
    for i in range(num_atom):
        for j in range(i, num_atom):
            for k in range(j, num_atom):
                fc3_elem = set_permutation_symmetry_fc3_elem(fc3, i, j, k)
                copy_permutation_symmetry_fc3_elem(fc3, fc3_elem, i, j, k)

def set_permutation_symmetry_fc3_deprecated(fc3):
    fc3_sym = np.zeros(fc3.shape, dtype='double')
    for (i, j, k) in list(np.ndindex(fc3.shape[:3])):
        fc3_sym[i, j, k] = set_permutation_symmetry_fc3_elem(fc3, i, j, k)

    for (i, j, k) in list(np.ndindex(fc3.shape[:3])):
        fc3[i, j, k] = fc3_sym[i, j, k]

def copy_permutation_symmetry_fc3_elem(fc3, fc3_elem, a, b, c):
    for (i, j, k) in list(np.ndindex(3, 3, 3)):
        fc3[a, b, c, i, j, k] = fc3_elem[i, j, k]
        fc3[c, a, b, k, i, j] = fc3_elem[i, j, k]
        fc3[b, c, a, j, k, i] = fc3_elem[i, j, k]
        fc3[a, c, b, i, k, j] = fc3_elem[i, j, k]
        fc3[b, a, c, j, i, k] = fc3_elem[i, j, k]
        fc3[c, b, a, k, j, i] = fc3_elem[i, j, k]

def set_permutation_symmetry_fc3_elem(fc3, a, b, c):
    tensor3 = np.zeros((3, 3, 3), dtype='double')
    for (i, j, k) in list(np.ndindex(3, 3, 3)):
        tensor3[i, j, k] = (fc3[a, b, c, i, j, k] +
                            fc3[c, a, b, k, i, j] +
                            fc3[b, c, a, j, k, i] +
                            fc3[a, c, b, i, k, j] +
                            fc3[b, a, c, j, i, k] +
                            fc3[c, b, a, k, j, i]) / 6
    return tensor3

def set_translational_invariance_fc3(fc3):
    for i in range(3):
        set_translational_invariance_fc3_per_index(fc3, index=i)

def set_translational_invariance_fc3_per_index(fc3, index=0):
    for i in range(fc3.shape[(1 + index) % 3]):
        for j in range(fc3.shape[(2 + index) % 3]):
            for k in range(fc3.shape[3]):
                for l in range(fc3.shape[4]):
                    for m in range(fc3.shape[5]):
                        if index == 0:
                            fc3[:, i, j, k, l, m] -= np.sum(
                                fc3[:, i, j, k, l, m]) / fc3.shape[0]
                        elif index == 1:
                            fc3[j, :, i, k, l, m] -= np.sum(
                                fc3[j, :, i, k, l, m]) / fc3.shape[1]
                        elif index == 2:
                            fc3[i, j, :, k, l, m] -= np.sum(
                                fc3[i, j, :, k, l, m]) / fc3.shape[2]
    
def third_rank_tensor_rotation(rot_cart, tensor):
    rot_tensor = np.zeros((3,3,3), dtype='double')
    for i in (0,1,2):
        for j in (0,1,2):
            for k in (0,1,2):
                rot_tensor[i, j, k] = _third_rank_tensor_rotation_elem(
                    rot_cart, tensor, i, j, k)
    return rot_tensor

def _get_fc3_least_atoms(fc3,
                         supercell,
                         disp_dataset,
                         fc2,
                         symmetry,
                         is_translational_symmetry,
                         is_permutation_symmetry,
                         verbose):
    symprec = symmetry.get_symmetry_tolerance()
    unique_first_atom_nums = np.unique(
        [x['number'] for x in disp_dataset['first_atoms']])
    for first_atom_num in unique_first_atom_nums:
        _get_fc3_one_atom(fc3,
                          supercell,
                          disp_dataset,
                          fc2,
                          first_atom_num,
                          symmetry.get_site_symmetry(first_atom_num),
                          is_translational_symmetry,
                          is_permutation_symmetry,
                          symprec,
                          verbose)

def get_atom_by_symmetry(positions,
                         rotation,
                         trans,
                         atom_number,
                         symprec=1e-5):

    rot_pos = np.dot(positions[atom_number], rotation.T) + trans
    for i, pos in enumerate(positions):
        diff = pos - rot_pos
        if (abs(diff -diff.round()) < symprec).all():
            return i

    print 'Position or symmetry is wrong.'
    raise ValueError

def get_atom_mapping_by_symmetry(positions,
                                 atom_search, 
                                 atom_target,
                                 rotations,
                                 translations,
                                 symprec):
    map_sym = -1

    for i, (r, t) in enumerate(zip(rotations, translations)):
        rot_pos = np.dot(positions[atom_search], r.T) + t
        diff = rot_pos - positions[atom_target]
        if (abs(diff -diff.round()) < symprec).all():
            map_sym = i
            break

    return map_sym

def get_delta_fc2(dataset_second_atoms,
                  atom1,
                  fc2,
                  supercell,
                  reduced_site_sym,
                  is_translational_symmetry,
                  is_permutation_symmetry,
                  symprec):
    disp_fc2 = get_constrained_fc2(supercell,
                                   dataset_second_atoms,
                                   atom1,
                                   reduced_site_sym,
                                   is_translational_symmetry,
                                   is_permutation_symmetry,
                                   symprec)
    return disp_fc2 - fc2

def get_constrained_fc2(supercell,
                        dataset_second_atoms,
                        atom1,
                        reduced_site_sym,
                        is_translational_symmetry,
                        is_permutation_symmetry,
                        symprec):
    """
    dataset_second_atoms: [{'number': 7,
                            'displacement': [],
                            'delta_forces': []}, ...]
    """
    num_atom = supercell.get_number_of_atoms()
    fc2 = np.zeros((num_atom, num_atom, 3, 3), dtype='double')
    atom_list = np.unique([x['number'] for x in dataset_second_atoms])
    atom_list_done = []
    for atom2 in atom_list:
        disps2 = []
        sets_of_forces = []
        for disps_second in dataset_second_atoms:
            if atom2 != disps_second['number']:
                continue
            atom_list_done.append(atom2)
            bond_sym = get_bond_symmetry(
                reduced_site_sym,
                supercell.get_scaled_positions(),
                atom1,
                atom2,
                symprec)
    
            disps2.append(disps_second['displacement'])
            sets_of_forces.append(disps_second['delta_forces'])
    
        solve_force_constants(fc2,
                              atom2,
                              disps2,
                              sets_of_forces,
                              supercell,
                              bond_sym,
                              symprec)

    # Shift positions according to set atom1 is at origin
    lattice = supercell.get_cell().T
    positions = supercell.get_scaled_positions()
    pos_center = positions[atom1].copy()
    positions -= pos_center
    atom_list = range(num_atom)
    distribute_force_constants(fc2,
                               atom_list,
                               atom_list_done,
                               lattice,
                               positions,
                               np.intc(reduced_site_sym).copy(),
                               np.zeros((len(reduced_site_sym), 3),
                                        dtype='double'),
                               symprec)

    if is_translational_symmetry:
        set_translational_invariance(fc2)

    if is_permutation_symmetry:
        set_permutation_symmetry(fc2)

    return fc2
        

def solve_fc3(fc3,
              first_atom_num,
              supercell,
              site_symmetry,
              displacements_first,
              delta_fc2s,
              symprec):
    lattice = supercell.get_cell().T
    site_sym_cart = [similarity_transformation(lattice, sym)
                     for sym in site_symmetry]
    num_atom = supercell.get_number_of_atoms()
    positions = supercell.get_scaled_positions()
    pos_center = positions[first_atom_num].copy()
    positions -= pos_center
    rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                 site_symmetry,
                                                 symprec)
    
    rot_disps = get_rotated_displacement(displacements_first, site_sym_cart)
    inv_U = np.linalg.pinv(rot_disps)
    for (i, j) in list(np.ndindex(num_atom, num_atom)):
        fc3[first_atom_num, i, j] = np.dot(inv_U, _get_rotated_fc2s(i, j, delta_fc2s, rot_map_syms, site_sym_cart)).reshape(3, 3, 3)

def show_drift_fc3(fc3, name="fc3"):
    # num_atom = fc3.shape[0]
    # maxval1 = 0
    # maxval2 = 0
    # maxval3 = 0
    fc3_sum1 = fc3.sum(axis=0)
    maxval1 = fc3_sum1.flatten()[np.abs(fc3_sum1).argmax()]
    fc3_sum2 = fc3.sum(axis=1)
    maxval2 = fc3_sum2.flatten()[np.abs(fc3_sum2).argmax()]
    fc3_sum3 = fc3.sum(axis=2)
    maxval3 = fc3_sum3.flatten()[np.abs(fc3_sum3).argmax()]

    # for i, j, k, l, m in list(np.ndindex((num_atom, num_atom, 3, 3, 3))):
    #     val1 = fc3[:, i, j, k, l, m].sum()
    #     val2 = fc3[i, :, j, k, l, m].sum()
    #     val3 = fc3[i, j, :, k, l, m].sum()
    #     if abs(val1) > abs(maxval1):
    #         maxval1 = val1
    #     if abs(val2) > abs(maxval2):
    #         maxval2 = val2
    #     if abs(val3) > abs(maxval3):
    #         maxval3 = val3
    print ("max drift of %s:" % name), maxval1, maxval2, maxval3

def cutoff_fc3(fc3,
               supercell,
               disp_dataset,
               symmetry,
               verbose=False):
    if verbose:
        print "Building atom mapping table..."
    fc3_done = _get_fc3_done(supercell, disp_dataset, symmetry)

    if verbose:
        print "Creating contracted fc3..."
    num_atom = supercell.get_number_of_atoms()
    for i in range(num_atom):
        for j in range(i, num_atom):
            for k in range(j, num_atom):
                ave_fc3 = _set_permutation_symmetry_fc3_elem_with_cutoff(
                    fc3, fc3_done, i, j, k)
                copy_permutation_symmetry_fc3_elem(fc3, ave_fc3, i, j, k)

def cutoff_fc3_by_zero(fc3, include_triplet):
    for i, j, k in np.ndindex(fc3.shape[:3]):
        if not include_triplet[i, j, k]:
            fc3[i,j,k] = 0.
    return include_triplet

def _get_fc3_one_atom(fc3,
                      supercell,
                      disp_dataset,
                      fc2,
                      first_atom_num,
                      site_symmetry,
                      is_translational_symmetry,
                      is_permutation_symmetry,
                      symprec,
                      verbose):
    displacements_first = []
    delta_fc2s = []
    for dataset_first_atom in disp_dataset['first_atoms']:
        if first_atom_num != dataset_first_atom['number']:
            continue
        
        displacements_first.append(dataset_first_atom['displacement'])
        if 'delta_fc2' in dataset_first_atom:
            delta_fc2s.append(dataset_first_atom['delta_fc2'])
        else:
            direction = np.dot(dataset_first_atom['displacement'],
                               np.linalg.inv(supercell.get_cell()))
            reduced_site_sym = get_reduced_site_symmetry(
                site_symmetry, direction, symprec)
            delta_fc2s.append(get_delta_fc2(
                    dataset_first_atom['second_atoms'],
                    dataset_first_atom['number'],
                    fc2,
                    supercell,
                    reduced_site_sym,
                    is_translational_symmetry,
                    is_permutation_symmetry,
                    symprec))

    solve_fc3(fc3,
              first_atom_num,
              supercell,
              site_symmetry,
              displacements_first,
              delta_fc2s,
              symprec)

    if verbose:
        print "Displacements for fc3[ %d, x, x ]" % (first_atom_num + 1)
        for i, v in enumerate(displacements_first):
            print "  [%7.4f %7.4f %7.4f]" % tuple(v)
            sys.stdout.flush()
        if verbose > 2:
            print "Site symmetry:"
            for i, v in enumerate(site_symmetry):
                print "  [%2d %2d %2d] #%2d" % tuple(list(v[0])+[i+1])
                print "  [%2d %2d %2d]" % tuple(v[1])
                print "  [%2d %2d %2d]\n" % tuple(v[2])
                sys.stdout.flush()

def _get_rotated_fc2s(i, j, fc2s, rot_map_syms, site_sym_cart):
    num_sym = len(site_sym_cart)
    rotated_fc2s = []
    for fc2 in fc2s:
        for sym, map_sym in zip(site_sym_cart, rot_map_syms):
            fc2_rot = fc2[map_sym[i], map_sym[j]]
            rotated_fc2s.append(similarity_transformation(sym, fc2_rot))
    return np.reshape(rotated_fc2s, (-1, 9))

def _third_rank_tensor_rotation_elem(rot, tensor, l, m, n):
    sum_elems = 0.
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            for k in (0, 1, 2):
                sum_elems += rot[l, i] * rot[m, j] * rot[n, k] * tensor[i, j, k]
    return sum_elems

def _get_fc3_done(supercell, disp_dataset, symmetry):
    num_atom = supercell.get_number_of_atoms()
    fc3_done = np.zeros((num_atom, num_atom, num_atom), dtype='byte')
    symprec = symmetry.get_symmetry_tolerance()
    positions = supercell.get_scaled_positions()
    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']

    atom_mapping = []
    for rot, trans in zip(rotations, translations):
        atom_indices = []
        for rot_pos in (np.dot(positions, rot.T) + trans):
            diff = positions - rot_pos
            diff -= np.rint(diff)
            atom_indices.append(
                np.where((np.abs(diff) < symprec).all(axis=1))[0][0])
        atom_mapping.append(atom_indices)

    for dataset_first_atom in disp_dataset['first_atoms']:
        first_atom_num = dataset_first_atom['number']
        site_symmetry = symmetry.get_site_symmetry(first_atom_num)
        direction = np.dot(dataset_first_atom['displacement'],
                           np.linalg.inv(supercell.get_cell()))
        reduced_site_sym = get_reduced_site_symmetry(
            site_symmetry, direction, symprec)
        least_second_atom_nums = []
        for second_atoms in dataset_first_atom['second_atoms']:
            if second_atoms['included']:
                least_second_atom_nums.append(second_atoms['number'])
        positions_shifted = positions - positions[first_atom_num]
        least_second_atom_nums = np.unique(least_second_atom_nums)

        second_atom_nums = []
        for red_rot in reduced_site_sym:
            for i in least_second_atom_nums:
                second_atom_nums.append(
                    get_atom_by_symmetry(positions_shifted,
                                         red_rot,
                                         np.zeros(3, dtype='double'),
                                         i,
                                         symprec))
        second_atom_nums = np.unique(second_atom_nums)

        for i in range(len(rotations)):
            rotated_atom1 = atom_mapping[i][first_atom_num]
            for j in second_atom_nums:
                fc3_done[rotated_atom1, atom_mapping[i][j]] = 1

    return fc3_done

def _set_permutation_symmetry_fc3_elem_with_cutoff(fc3, fc3_done, a, b, c):
    sum_done = (fc3_done[a, b, c] +
                fc3_done[c, a, b] +
                fc3_done[b, c, a] +
                fc3_done[b, a, c] +
                fc3_done[c, b, a] +
                fc3_done[a, c, b])
    tensor3 = np.zeros((3, 3, 3), dtype='double')
    if sum_done > 0:
        for (i, j, k) in list(np.ndindex(3, 3, 3)):
            tensor3[i, j, k] = (fc3[a, b, c, i, j, k] * fc3_done[a, b, c] +
                                fc3[c, a, b, k, i, j] * fc3_done[c, a, b] +
                                fc3[b, c, a, j, k, i] * fc3_done[b, c, a] +
                                fc3[a, c, b, i, k, j] * fc3_done[a, c, b] +
                                fc3[b, a, c, j, i, k] * fc3_done[b, a, c] +
                                fc3[c, b, a, k, j, i] * fc3_done[c, b, a])
            tensor3[i, j, k] /= sum_done
    return tensor3
#
# def _set_permutation_symmetry_fc3_elem_with_cutoff(fc3, fc3_done, a, b, c):
#     done = np.array([fc3_done[a, b, c],
#                 fc3_done[c, a, b],
#                 fc3_done[b, c, a],
#                 fc3_done[b, a, c],
#                 fc3_done[c, b, a],
#                 fc3_done[a, c, b]])
#     tensor3 = np.zeros((3, 3, 3), dtype='double')
#     if done.sum() > 0:
#         fc = np.array([fc3[a, b, c],
#                        np.einsum('kij',fc3[c, a, b]),
#                        np.einsum('jki',fc3[b, c, a]),
#                        np.einsum('ikj',fc3[a, c, b]),
#                        np.einsum('jik',fc3[b, a, c]),
#                        np.einsum('kji',fc3[c, b, a])])
#         fc_done = fc[np.where(done)]
#         tensor3 = fc_done.sum(axis=0) / len(fc_done)
#         fcmin = np.abs(fc_done).min(axis=0); fcmax = np.abs(fc_done).max(axis=0)
#         discrepancy = np.where(fcmax>1e-2, fcmin / fcmax, 1)
#         if (discrepancy < 1e-2).any():
#             print a, b, c
#         # for (i, j, k) in list(np.ndindex(3, 3, 3)):
#         #     fc = np.array([fc3[a, b, c, i, j, k],
#         #                    fc3[c, a, b, k, i, j],
#         #                    fc3[b, c, a, j, k, i],
#         #                    fc3[a, c, b, i, k, j],
#         #                    fc3[b, a, c, j, i, k],
#         #                    fc3[c, b, a, k, j, i]])
#         #     fc_done = np.extract(done, fc)
#         #     if len(fc_done) > 2:
#         #         tensor3[i, j, k] = (fc_done.sum() - fc_done[np.argmin(np.abs(fc_done))] - fc_done[np.argmax(np.abs(fc_done))]) / (len(fc_done) - 2)
#         #     else:
#         #         tensor3[i, j, k] = fc_done.sum() / len(fc_done)
#     return tensor3
"""
Spglib interface for ASE
"""

# import phonopy._spglib as spg
import phonopy._spglib as spg
import numpy as np

def get_symmetry(bulk, symprec=1e-5, angle_tolerance=-1.0):
    """
    Return symmetry operations as hash.
    Hash key 'rotations' gives the numpy integer array
    of the rotation matrices for scaled positions
    Hash key 'translations' gives the numpy double array
    of the translation vectors in scaled positions
    """

    # Atomic positions have to be specified by scaled positions for spglib.
    positions = bulk.get_scaled_positions().copy()
    lattice = bulk.get_cell().T.copy()
    numbers = np.intc(bulk.get_atomic_numbers()).copy()
  
    # Get number of symmetry operations and allocate symmetry operations
    # multi = spg.multiplicity(cell, positions, numbers, symprec)
    multi = 48 * bulk.get_number_of_atoms()
    rotation = np.zeros((multi, 3, 3), dtype='intc')
    translation = np.zeros((multi, 3))
  
    # Get symmetry operations
    magmoms = bulk.get_magnetic_moments()
    if magmoms == None:
        num_sym = spg.symmetry(rotation,
                               translation,
                               lattice,
                               positions,
                               numbers,
                               symprec,
                               angle_tolerance)
    else:
        num_sym = spg.symmetry_with_collinear_spin(rotation,
                                                   translation,
                                                   lattice,
                                                   positions,
                                                   numbers,
                                                   magmoms,
                                                   symprec,
                                                   angle_tolerance)
  
    return {'rotations': rotation[:num_sym].copy(),
            'translations': translation[:num_sym].copy()}

def get_symmetry_dataset(bulk, symprec=1e-5, angle_tolerance=-1.0):
    """
    number: International space group number
    international: International symbol
    hall: Hall symbol
    transformation_matrix:
      Transformation matrix from lattice of input cell to Bravais lattice
      L^bravais = L^original * Tmat
    origin shift: Origin shift in the setting of 'Bravais lattice'
    rotations, translations:
      Rotation matrices and translation vectors
      Space group operations are obtained by
        [(r,t) for r, t in zip(rotations, translations)]
    wyckoffs:
      Wyckoff letters
    """
    positions = bulk.get_scaled_positions().copy()
    lattice = bulk.get_cell().T.copy()
    numbers = np.intc(bulk.get_atomic_numbers()).copy()
    keys = ('number',
            'international',
            'hall',
            'transformation_matrix',
            'origin_shift',
            'rotations',
            'translations',
            'wyckoffs',
            'equivalent_atoms')
    dataset = {}
    for key, data in zip(keys, spg.dataset(lattice,
                                           positions,
                                           numbers,
                                           symprec,
                                           angle_tolerance)):
        dataset[key] = data

    dataset['international'] = dataset['international'].strip()
    dataset['hall'] = dataset['hall'].strip()
    dataset['transformation_matrix'] = np.double(
        dataset['transformation_matrix'])
    dataset['origin_shift'] = np.double(dataset['origin_shift'])
    dataset['rotations'] = np.intc(dataset['rotations'])
    dataset['translations'] = np.double(dataset['translations'])
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    dataset['wyckoffs'] = [letters[x] for x in dataset['wyckoffs']]
    dataset['equivalent_atoms'] = np.intc(dataset['equivalent_atoms'])

    return dataset

def get_spacegroup(bulk, symprec=1e-5, angle_tolerance=-1.0):
    """
    Return space group in international table symbol and number
    as a string.
    """
    # Atomic positions have to be specified by scaled positions for spglib.
    return spg.spacegroup(bulk.get_cell().T.copy(),
                          bulk.get_scaled_positions().copy(),
                          np.intc(bulk.get_atomic_numbers()).copy(),
                          symprec,
                          angle_tolerance)

def get_pointgroup(rotations):
    """
    Return point group in international table symbol and number.
    The symbols are mapped to the numbers as follows:
    1   "1    "
    2   "-1   "
    3   "2    "
    4   "m    "
    5   "2/m  "
    6   "222  "
    7   "mm2  "
    8   "mmm  "
    9   "4    "
    10  "-4   "
    11  "4/m  "
    12  "422  "
    13  "4mm  "
    14  "-42m "
    15  "4/mmm"
    16  "3    "
    17  "-3   "
    18  "32   "
    19  "3m   "
    20  "-3m  "
    21  "6    "
    22  "-6   "
    23  "6/m  "
    24  "622  "
    25  "6mm  "
    26  "-62m "
    27  "6/mmm"
    28  "23   "
    29  "m-3  "
    30  "432  "
    31  "-43m "
    32  "m-3m "
    """

    # (symbol, pointgroup_number, transformation_matrix)
    return spg.pointgroup(np.intc(rotations).copy())

def refine_cell(bulk, symprec=1e-5, angle_tolerance=-1.0):
    """
    Return refined cell
    """
    # Atomic positions have to be specified by scaled positions for spglib.
    num_atom = bulk.get_number_of_atoms()
    lattice = bulk.get_cell().T.copy()
    pos = np.zeros((num_atom * 4, 3), dtype='double')
    pos[:num_atom] = bulk.get_scaled_positions()

    numbers = np.zeros(num_atom * 4, dtype='intc')
    numbers[:num_atom] = np.intc(bulk.get_atomic_numbers())
    num_atom_bravais = spg.refine_cell(lattice,
                                       pos,
                                       numbers,
                                       num_atom,
                                       symprec,
                                       angle_tolerance)

    return (lattice.T.copy(),
            pos[:num_atom_bravais].copy(),
            numbers[:num_atom_bravais].copy())

def find_primitive(bulk, symprec=1e-5, angle_tolerance=-1.0):
    """
    A primitive cell in the input cell is searched and returned
    as an object of Atoms class.
    If no primitive cell is found, (None, None, None) is returned.
    """

    # Atomic positions have to be specified by scaled positions for spglib.
    positions = bulk.get_scaled_positions().copy()
    lattice = bulk.get_cell().T.copy()
    numbers = np.intc(bulk.get_atomic_numbers()).copy()

    # lattice is transposed with respect to the definition of Atoms class
    num_atom_prim = spg.primitive(lattice,
                                  positions,
                                  numbers,
                                  symprec,
                                  angle_tolerance)
    if num_atom_prim > 0:
        return (lattice.T.copy(),
                positions[:num_atom_prim].copy(),
                numbers[:num_atom_prim].copy())
    else:
        return None, None, None
  
def get_ir_kpoints(kpoint,
                   bulk,
                   is_time_reversal=True,
                   symprec=1e-5):
    """
    Retrun irreducible kpoints
    """
    mapping = np.zeros(kpoint.shape[0], dtype='intc')
    spg.ir_kpoints(mapping,
                   kpoint,
                   bulk.get_cell().T.copy(),
                   bulk.get_scaled_positions().copy(),
                   np.intc(bulk.get_atomic_numbers()).copy(),
                   is_time_reversal * 1,
                   symprec)
    return mapping

def get_kpoint_group(mesh,  point_group_operations, qpoints=[[0.,0.,0.]], is_time_reversal=True):
    reverse_kpt_group = np.zeros((len(point_group_operations) * 2, 3, 3), dtype="intc")
    num_kpt_rots = spg.kpointgroup(reverse_kpt_group,
                                    np.intc(point_group_operations).copy(),
                                    np.intc(mesh).copy(),
                                    np.double(qpoints).copy(),
                                    is_time_reversal)
    return reverse_kpt_group[:num_kpt_rots]

def get_ir_reciprocal_mesh(mesh,
                           bulk,
                           is_shift=np.zeros(3, dtype='intc'),
                           is_time_reversal=True,
                           symprec=1e-5):
    """
    Return k-points mesh and k-point map to the irreducible k-points
    The symmetry is serched from the input cell.
    is_shift=[0, 0, 0] gives Gamma center mesh.
    """

    mapping = np.zeros(np.prod(mesh), dtype='intc')
    mesh_points = np.zeros((np.prod(mesh), 3), dtype='intc')
    spg.ir_reciprocal_mesh(mesh_points,
                           mapping,
                           np.intc(mesh).copy(),
                           np.intc(is_shift).copy(),
                           is_time_reversal * 1,
                           bulk.get_cell().T.copy(),
                           bulk.get_scaled_positions().copy(),
                           np.intc(bulk.get_atomic_numbers()).copy(),
                           symprec)
  
    return mapping, mesh_points
  
def get_stabilized_reciprocal_mesh(mesh,
                                   rotations,
                                   is_shift=np.zeros(3, dtype='intc'),
                                   is_time_reversal=True,
                                   qpoints=np.double([])):
    """
    Return k-point map to the irreducible k-points and k-point grid points .

    The symmetry is searched from the input rotation matrices in real space.
    
    is_shift=[0, 0, 0] gives Gamma center mesh and the values 1 give
    half mesh distance shifts.
    """
    
    mapping = np.zeros(np.prod(mesh), dtype='intc')
    rot_mapping = np.zeros(np.prod(mesh), dtype="intc")
    mesh_points = np.zeros((np.prod(mesh), 3), dtype='intc')

    qpoints = np.double(qpoints).copy()
    if qpoints.shape == (3,):
        qpoints = np.array([qpoints], dtype='double', order='C')
    if qpoints.shape == (0,):
        qpoints = np.array([[0, 0, 0]], dtype='double', order='C')
    spg.stabilized_reciprocal_mesh(mesh_points,
                                   mapping,
                                   rot_mapping,
                                   np.intc(mesh).copy(),
                                   np.intc(is_shift),
                                   is_time_reversal * 1,
                                   np.intc(rotations).copy(),
                                   np.double(qpoints))
    
    return mapping, mesh_points


def get_BZ_grid_points_by_rotations(address_orig,
                                    reciprocal_rotations,
                                    mesh,
                                    bz_map,
                                    is_shift=np.zeros(3, dtype='intc')):
    """
    Rotation operations in reciprocal space ``reciprocal_rotations`` are applied
    to a grid point ``grid_point`` and resulting grid points are returned.
    """

    rot_grid_points = np.zeros(len(reciprocal_rotations), dtype='intc')
    spg.BZ_grid_points_by_rotations(
        rot_grid_points,
        np.array(address_orig, dtype='intc'),
        np.array(reciprocal_rotations, dtype='intc', order='C'),
        np.array(mesh, dtype='intc'),
        np.array(is_shift, dtype='intc'),
        bz_map)

    return rot_grid_points

def relocate_BZ_grid_address(grid_address,
                             mesh,
                             reciprocal_lattice, # column vectors
                             is_shift=np.zeros(3, dtype='intc'),
                             is_bz_map_to_orig=False):
    """
    Grid addresses are relocated inside Brillouin zone.
    Number of ir-grid-points inside Brillouin zone is returned.
    It is assumed that the following arrays have the shapes of
      bz_grid_address[prod(mesh + 1)][3]
      bz_map[prod(mesh * 2)]
    where grid_address[prod(mesh)][3].
    Each element of grid_address is mapped to each element of
    bz_grid_address with keeping element order. bz_grid_address has
    larger memory space to represent BZ surface even if some points
    on a surface are translationally equivalent to the other points
    on the other surface. Those equivalent points are added successively
    as grid point numbers to bz_grid_address. Those added grid points
    are stored after the address of end point of grid_address, i.e.

    |-----------------array size of bz_grid_address---------------------|
    |--grid addresses similar to grid_address--|--newly added ones--|xxx|

    where xxx means the memory space that may not be used. Number of grid
    points stored in bz_grid_address is returned.
    bz_map is used to recover grid point index expanded to include BZ
    surface from grid address. The grid point indices are mapped to
    (mesh[0] * 2) x (mesh[1] * 2) x (mesh[2] * 2) space (bz_map).
    """

    bz_grid_address = np.zeros(
        ((mesh[0] + 1) * (mesh[1] + 1) * (mesh[2] + 1), 3), dtype='intc')
    bz_map = np.zeros(
        (2 * mesh[0]) * (2 * mesh[1]) * (2 * mesh[2]), dtype='intc')
    bz_map_orig = np.zeros((mesh[0] + 1) * (mesh[1] + 1) * (mesh[2] + 1), dtype='intc')
    num_bz_ir = spg.BZ_grid_address(
        bz_grid_address,
        bz_map,
        bz_map_orig,
        grid_address,
        np.array(mesh, dtype='intc'),
        np.array(reciprocal_lattice, dtype='double', order='C'),
        np.array(is_shift, dtype='intc'))
    if is_bz_map_to_orig:
        return bz_grid_address[:num_bz_ir], bz_map, bz_map_orig[:num_bz_ir]
    return bz_grid_address[:num_bz_ir], bz_map

def get_mappings(mesh,
                 rotations,
                 is_shift=np.zeros(3, dtype='intc'),
                 is_time_reversal=True,
                 qpoints=np.double([])):
    """
    Return k-point map to the irreducible k-points and k-point grid points .

    The symmetry is searched from the input rotation matrices in real space.

    is_shift=[0, 0, 0] gives Gamma center mesh and the values 1 give
    half mesh distance shifts.
    """

    mapping = np.zeros(np.prod(mesh), dtype='intc')
    rot_mapping = np.zeros(np.prod(mesh), dtype="intc")
    mesh_points = np.zeros((np.prod(mesh), 3), dtype='intc')
    qpoints = np.double(qpoints).copy()
    if qpoints.shape == (3,):
        qpoints = np.double([qpoints])
    spg.stabilized_reciprocal_mesh(mesh_points,
                                   mapping,
                                   rot_mapping,
                                   np.intc(mesh).copy(),
                                   np.intc(is_shift),
                                   is_time_reversal * 1,
                                   np.intc(rotations).copy(),
                                   np.double(qpoints))

    return mapping,  rot_mapping

def get_reduced_triplets_permute_sym(triplets,
                                     mesh,
                                     first_mapping,
                                     first_rotation,
                                     second_mapping):
    mesh = np.array(mesh)
    triplet_numbers = np.array([len(tri) for tri in triplets], dtype="intc")
    grid_points = np.array([triplet[0][0] for triplet in triplets], dtype="intc")
    triplets_all = np.vstack(triplets)
    triplets_mapping = np.arange(len(triplets_all)).astype("intc")
    sequences = np.array([[0,1,2]] * len(triplets_all), dtype="byte")

    num_irred_triplets =  spg.reduce_triplets_permute_sym(triplets_mapping,
                                           sequences,
                                           triplets_all.astype("intc"),
                                           np.intc(grid_points).copy(),
                                           np.intc(triplet_numbers).copy(),
                                           mesh.astype("intc"),
                                           np.intc(first_mapping).copy(),
                                           np.intc(first_rotation).copy(),
                                           np.intc(second_mapping).copy())
    assert len(np.unique(triplets_mapping)) == num_irred_triplets
    unique_triplets, indices = np.unique(triplets_mapping, return_inverse=True)
    triplets_map = []
    triplet_sequence = []
    num_triplets = 0
    for i, triplets_at_q in enumerate(triplets):
        triplets_map.append(indices[num_triplets:num_triplets+len(triplets_at_q)])
        triplet_sequence.append(sequences[num_triplets:num_triplets+len(triplets_at_q)])
        num_triplets += len(triplets_at_q)
    return unique_triplets,triplets_map, triplet_sequence

def get_reduced_pairs_permute_sym(pairs,
                                 mesh,
                                 first_mapping,
                                 first_rotation,
                                 second_mapping):
    mesh = np.array(mesh)
    pair_numbers = np.array([len(pair) for pair in pairs], dtype="intc")
    grid_points = np.array([pair[0][0] for pair in pairs], dtype="intc")
    pairs_all = np.vstack(pairs)
    pairs_mapping = np.arange(len(pairs_all)).astype("intc")
    sequences = np.array([[0,1]] * len(pairs_all), dtype="byte")

    num_irred_pairs =  spg.reduce_pairs_permute_sym(pairs_mapping,
                                           sequences,
                                           pairs_all.astype("intc"),
                                           np.intc(grid_points).copy(),
                                           np.intc(pair_numbers).copy(),
                                           mesh.astype("intc"),
                                           np.intc(first_mapping).copy(),
                                           np.intc(first_rotation).copy(),
                                           np.intc(second_mapping).copy())
    assert len(np.unique(pairs_mapping)) == num_irred_pairs
    unique_pairs, indices = np.unique(pairs_mapping, return_inverse=True)
    pairs_map = []
    pair_sequence = []
    num_pairs = 0
    for i, pairs_at_q in enumerate(pairs):
        pairs_map.append(indices[num_pairs:num_pairs+len(pairs_at_q)])
        pair_sequence.append(sequences[num_pairs:num_pairs+len(pairs_at_q)])
        num_pairs += len(pairs_at_q)
    return unique_pairs,pairs_map, pair_sequence

def get_triplets_reciprocal_mesh_at_q(fixed_grid_number,
                                      mesh,
                                      rotations,
                                      is_time_reversal=True,
                                      is_return_map=False,
                                      is_return_rot_map=False):

    weights = np.zeros(np.prod(mesh), dtype='intc')
    third_q = np.zeros(np.prod(mesh), dtype='intc')
    mesh_points = np.zeros((np.prod(mesh), 3), dtype='intc')
    mapping = np.zeros(np.prod(mesh), dtype='intc')
    rot_mapping = np.zeros(np.prod(mesh), dtype='intc')

    spg.triplets_reciprocal_mesh_at_q(weights,
                                      mesh_points,
                                      third_q,
                                      mapping,
                                      rot_mapping,
                                      fixed_grid_number,
                                      np.intc(mesh).copy(),
                                      is_time_reversal * 1,
                                      np.intc(rotations).copy())
    assert len(mapping[np.unique(mapping)]) == len(weights[np.nonzero(weights)]), \
        "At grid %d, number of irreducible mapping: %d is not equal to the number of irreducible triplets%d"%\
            (fixed_grid_number, len(mapping[np.unique(mapping)]), len(weights[np.nonzero(weights)]))
    if not is_return_map and not is_return_rot_map:
        return weights, third_q, mesh_points
    elif not is_return_rot_map:
        return weights, third_q,mesh_points, mapping
    else:
        return weights, third_q,mesh_points, mapping, rot_mapping

def get_BZ_triplets_at_q(grid_point,
                         bz_grid_address,
                         bz_map,
                         map_triplets,
                         mesh):
    """grid_address is overwritten."""
    weights = np.zeros_like(map_triplets)
    for g in map_triplets:
        weights[g] += 1
    ir_weights = np.extract(weights > 0, weights)
    triplets = np.zeros((len(ir_weights), 3), dtype='intc')
    num_ir_ret = spg.BZ_triplets_at_q(triplets,
                                      grid_point,
                                      bz_grid_address,
                                      bz_map,
                                      map_triplets,
                                      np.array(mesh, dtype='intc'))

    return triplets, ir_weights

def get_triplets_mapping_at_q(fixed_grid_number,
                              mesh,
                              rotations,
                              is_time_reversal=True):

    weights = np.zeros(np.prod(mesh), dtype='intc')
    third_q = np.zeros(np.prod(mesh), dtype='intc')
    mesh_points = np.zeros((np.prod(mesh), 3), dtype='intc')
    mapping = np.zeros(np.prod(mesh), dtype='intc')
    rot_mapping = np.zeros(np.prod(mesh), dtype='intc')

    spg.triplets_reciprocal_mesh_at_q(weights,
                                      mesh_points,
                                      third_q,
                                      mapping,
                                      rot_mapping,
                                      fixed_grid_number,
                                      np.intc(mesh).copy(),
                                      is_time_reversal * 1,
                                      np.intc(rotations).copy())
    assert len(mapping[np.unique(mapping)]) == len(weights[np.nonzero(weights)]), \
        "At grid %d, number of irreducible mapping: %d is not equal to the number of irreducible triplets%d"%\
            (fixed_grid_number, len(mapping[np.unique(mapping)]), len(weights[np.nonzero(weights)]))
    return mapping, rot_mapping


def get_triplets_inverse_map_sum_at_q(fixed_grid_number,
                                      mesh,
                                      rotations,
                                      is_time_reversal=True):

    weights = np.zeros(np.prod(mesh), dtype='intc')
    third_q = np.zeros(np.prod(mesh), dtype='intc')
    mesh_points = np.zeros((np.prod(mesh), 3), dtype='intc')
    mapping = np.zeros(np.prod(mesh), dtype='intc')
    rot_mapping = np.zeros(np.prod(mesh), dtype='intc')

    spg.triplets_reciprocal_mesh_at_q(weights,
                                      mesh_points,
                                      third_q,
                                      mapping,
                                      rot_mapping,
                                      fixed_grid_number,
                                      np.intc(mesh).copy(),
                                      is_time_reversal * 1,
                                      np.intc(rotations).copy())
    assert len(mapping[np.unique(mapping)]) == len(weights[np.nonzero(weights)]), \
        "At grid %d, number of irreducible mapping: %d is not equal to the number of irreducible triplets%d"%\
            (fixed_grid_number, len(mapping[np.unique(mapping)]), len(weights[np.nonzero(weights)]))
    return mapping, rot_mapping

def get_grid_triplets_at_q(q_grid_point,
                           grid_points,
                           third_q,
                           weights,
                           mesh):
    num_ir_tripltes = (weights > 0).sum()
    triplets = np.zeros((num_ir_tripltes, 3), dtype='intc')
    spg.grid_triplets_at_q(triplets,
                           q_grid_point,
                           grid_points,
                           third_q,
                           weights,
                           np.intc(mesh).copy())
    return triplets


def get_neighboring_grid_points(grid_point,
                                relative_grid_address,
                                mesh,
                                bz_grid_address,
                                bz_map):
    relative_grid_points = np.zeros(len(relative_grid_address), dtype='intc')
    spg.neighboring_grid_points(relative_grid_points,
                                grid_point,
                                relative_grid_address,
                                mesh,
                                bz_grid_address,
                                bz_map)
    return relative_grid_points



######################
# Tetrahedron method #
######################
def get_triplets_tetrahedra_vertices(relative_grid_address,
                                     mesh,
                                     triplets,
                                     bz_grid_address,
                                     bz_map):
    num_tripltes = len(triplets)
    vertices = np.zeros((num_tripltes, 2, 24, 4), dtype='intc')
    for i, tp in enumerate(triplets):
        vertices_at_tp = np.zeros((2, 24, 4), dtype='intc')
        spg.triplet_tetrahedra_vertices(
            vertices_at_tp,
            relative_grid_address,
            np.array(mesh, dtype='intc'),
            tp,
            bz_grid_address,
            bz_map)
        vertices[i] = vertices_at_tp

    return vertices

def get_tetrahedra_relative_grid_address(microzone_lattice):
    """
    reciprocal_lattice:
      column vectors of parallel piped microzone lattice
      which can be obtained by:
      microzone_lattice = np.linalg.inv(bulk.get_cell()) / mesh
    """

    relative_grid_address = np.zeros((24, 4, 3), dtype='intc')
    spg.tetrahedra_relative_grid_address(
        relative_grid_address,
        np.array(microzone_lattice, dtype='double', order='C'))

    return relative_grid_address

def get_all_tetrahedra_relative_grid_address():
    relative_grid_address = np.zeros((4, 24, 4, 3), dtype='intc')
    spg.all_tetrahedra_relative_grid_address(relative_grid_address)

    return relative_grid_address

def get_tetrahedra_integration_weight(omegas,
                                      tetrahedra_omegas,
                                      function='I'):
    if isinstance(omegas, float):
        return spg.tetrahedra_integration_weight(
            omegas,
            np.array(tetrahedra_omegas, dtype='double', order='C'),
            function)
    else:
        integration_weights = np.zeros(len(omegas), dtype='double')
        spg.tetrahedra_integration_weight_at_omegas(
            integration_weights,
            np.array(omegas, dtype='double'),
            np.array(tetrahedra_omegas, dtype='double', order='C'),
            function)
        return integration_weights
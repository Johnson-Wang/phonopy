import numpy as np
import phonopy.structure.spglib as spg
from phonopy.units import THzToEv, Kb
from phonopy.structure.symmetry import Symmetry
from phonopy.structure.tetrahedron_method import TetrahedronMethod

def gaussian(x, sigma):
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-x**2 / 2 / sigma**2)

def occupation(x, t):
    return 1.0 / (np.exp(THzToEv * x / (Kb * t)) - 1)

def get_triplets_at_q(grid_point,
                      mesh,
                      point_group, # real space point group of space group
                      primitive_lattice, # column vectors
                      is_time_reversal=True,
                      stores_triplets_map=False):
    reciprocal_lattice = np.linalg.inv(primitive_lattice)
    (triples_at_q_crude,
     weights_at_q,
     grid_address,
     grid_map)=\
        get_triplets_at_q_crude(grid_point, mesh, point_group, is_time_reversal)

    (triplets_at_q,
     weights,
     bz_grid_address,
     bz_map)=\
        get_BZ_triplets_at_q(grid_point, mesh, reciprocal_lattice, grid_address, grid_map)

    # map_triplets, map_q, grid_address = spg.get_triplets_reciprocal_mesh_at_q(
    #     grid_point,
    #     mesh,
    #     point_group,
    #     is_time_reversal=is_time_reversal)
    # bz_grid_address, bz_map = spg.relocate_BZ_grid_address(grid_address,
    #                                                        mesh,
    #                                                        primitive_lattice)
    # triplets_at_q, weights = spg.get_BZ_triplets_at_q(
    #     grid_point,
    #     bz_grid_address,
    #     bz_map,
    #     map_triplets,
    #     mesh)

    assert np.prod(mesh) == weights.sum(), \
        "Num grid points %d, sum of weight %d" % (
                    np.prod(mesh), weights.sum())

    # These maps are required for collision matrix calculation.
    if not stores_triplets_map:
        grid_map = None

    return triplets_at_q, weights, bz_grid_address, bz_map, grid_map

def get_triplets_at_q_crude(grid_point,
                           mesh,
                           point_group, # real space point group of space group
                           is_time_reversal=True):
    #Looking for triplets that satisfy q+q'+q''=0(G)
    weights, third_q, grid_address, grid_map = spg.get_triplets_reciprocal_mesh_at_q(
        grid_point,
        mesh,
        point_group,
        is_time_reversal,
        is_return_map=True)
    second_q, index = np.unique(grid_map, return_index=True)

    triples_at_q = [[grid_point, q2, q3] for (q2, q3) in zip(second_q, third_q[index])]
    weights_at_q = np.extract(weights > 0, weights)

    assert np.prod(mesh) == weights_at_q.sum(), \
        "Num grid points %d, sum of weight %d" % (
                    np.prod(mesh), weights_at_q.sum())

    return triples_at_q, weights_at_q, grid_address, grid_map

def get_BZ_triplets_at_q(grid_point,
                         mesh,
                         reciprocal_lattice,
                         grid_address,
                         map_q):
    bz_grid_address, bz_map = spg.relocate_BZ_grid_address(grid_address,
                                                           mesh,
                                                           reciprocal_lattice)
    triplets_at_q, weights = spg.get_BZ_triplets_at_q(
        grid_point,
        bz_grid_address,
        bz_map,
        map_q,
        mesh)

    assert np.prod(mesh) == weights.sum(), \
        "Num grid points %d, sum of weight %d" % (
                    np.prod(mesh), weights.sum())

    return triplets_at_q, weights, bz_grid_address, bz_map

def get_nosym_triplets_at_q(grid_point,
                            mesh,
                            primitive_lattice,
                            stores_triplets_map=False):
    grid_address = get_grid_address(mesh)
    map_q = np.arange(len(grid_address), dtype='intc')
    bz_grid_address, bz_map = spg.relocate_BZ_grid_address(grid_address,
                                                           mesh,
                                                           primitive_lattice)
    triplets_at_q, weights = spg.get_BZ_triplets_at_q(
        grid_point,
        bz_grid_address,
        bz_map,
        map_q,
        mesh)

    if not stores_triplets_map:
        map_q = None

    return triplets_at_q, weights, bz_grid_address, bz_map, map_q

# def get_nosym_triplets_at_q(grid_point, mesh):
#     grid_address = get_grid_address(mesh)
#
#     weights = np.ones(len(grid_address), dtype='intc')
#     third_q = np.zeros_like(weights)
#
#     # triplets = np.zeros((len(grid_address), 3), dtype='intc')
#     # for i, g1 in enumerate(grid_address):
#     #     g2 = - (grid_address[grid_point] + g1)
#     #     q = get_grid_point_from_address(g2, mesh)
#     #     triplets[i] = [grid_point, i, q]
#
#     for i, g1 in enumerate(grid_address):
#         g2 = - (grid_address[grid_point] + g1)
#         third_q[i] = get_grid_point_from_address(g2, mesh)
#     triplets = spg.get_grid_triplets_at_q(
#         grid_point,
#         grid_address,
#         third_q,
#         weights,
#         mesh)
#     grid_address = get_grid_address([x + (x % 2 == 0) for x in mesh])
#
#     return triplets, weights, grid_address

def get_grid_address(mesh, is_time_reversal=False, is_shift=np.zeros(3, dtype='intc'), is_return_map=False):
    grid_mapping_table, grid_address = spg.get_stabilized_reciprocal_mesh(
        mesh,
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        is_time_reversal=is_time_reversal,
        is_shift=is_shift)
    if is_return_map:
        return grid_address, grid_mapping_table
    else:
        return grid_address

def get_bz_grid_address(mesh, primitive_lattice, with_boundary=False, is_bz_map_to_pp=False):
    "is_bz_map_to_pp: Brillouine zone mapping to the original parallelpipe"
    grid_address = get_grid_address(mesh)
    bz_grid_address, bz_map, bz_to_pp_map = spg.relocate_BZ_grid_address(grid_address,
                                                           mesh,
                                                           primitive_lattice,
                                                           is_bz_map_to_orig=is_bz_map_to_pp)
    if with_boundary:
        if is_bz_map_to_pp:
            return bz_grid_address, bz_map, bz_to_pp_map
        else:
            return bz_grid_address, bz_map
    else:
        return bz_grid_address[:np.prod(mesh)]

def get_triplets_integration_weights(interaction,
                                     frequency_points,
                                     sigma_object,
                                     neighboring_phonons=True,
                                     triplets=None,
                                     lang='C'):
    "g0: -0 + 1 + 2; g1: -0 + 1 - 2; g2: -0 -1 + 2"
    if triplets == None:
        triplets = interaction.get_triplets_at_q()[0]
    frequencies = interaction.get_phonons()[0]
    num_band = frequencies.shape[1]
    g = np.zeros(
        (3, len(triplets), len(frequency_points), num_band, num_band),
        dtype='double')

    if sigma_object is not None:
        if lang == 'C':
            import anharmonic._phono3py as phono3c
            if isinstance(sigma_object, float):
                phono3c.triplets_integration_weights_with_sigma(
                    g,
                    frequency_points,
                    triplets,
                    frequencies,
                    sigma_object)
            elif isinstance(sigma_object, np.ndarray):
                assert sigma_object.shape == g.shape[1:]
                phono3c.triplets_integration_weights_with_asigma(
                    g,
                    frequency_points,
                    triplets,
                    frequencies,
                    sigma_object)

        else:
            if isinstance(sigma_object, np.ndarray):
                assert sigma_object.shape == g.shape[1:]
            else:
                sigma_object = np.ones(g.shape[1:]) * sigma_object

            for i, tp in enumerate(triplets):
                f1s = frequencies[tp[1]]
                f2s = frequencies[tp[2]]
                for j, k in list(np.ndindex((num_band, num_band))):
                    f1 = f1s[j]
                    f2 = f2s[k]
                    s = sigma_object[i, :, j, k]
                    g0 = gaussian(-frequency_points + f1 + f2, s)
                    g[0, i, :, j, k] = g0
                    g1 = gaussian(-frequency_points + f1 - f2, s)
                    g[1, i, :, j, k] = g1
                    g2 = gaussian(-frequency_points - f1 + f2, s)
                    g[2, i, :, j, k] = g2

    else:
        if lang == 'C':
            _set_triplets_integration_weights_c(
                g,
                interaction,
                frequency_points,
                neighboring_phonons=neighboring_phonons,
                triplets_at_q=triplets)
        else:
            _set_triplets_integration_weights_py(
                g, interaction, frequency_points, triplets_at_q=triplets)

    return g

def _set_triplets_integration_weights_c(g,
                                        interaction,
                                        frequency_points,
                                        neighboring_phonons=True,
                                        triplets_at_q=None):
    import anharmonic._phono3py as phono3c
    if triplets_at_q == None:
        triplets_at_q = interaction.get_triplets_at_q()[0]
    reciprocal_lattice = np.linalg.inv(interaction.get_primitive().get_cell())
    mesh = interaction.get_mesh_numbers()
    thm = TetrahedronMethod(reciprocal_lattice, mesh=mesh)
    grid_address = interaction.get_grid_address()
    bz_map = interaction.get_bz_map()
    if neighboring_phonons:
        unique_vertices = thm.get_unique_tetrahedra_vertices()
        for i, j in zip((1, 2), (1, -1)):
            neighboring_grid_points = np.zeros(
                len(unique_vertices) * len(triplets_at_q), dtype='intc')
            phono3c.neighboring_grid_points(
                neighboring_grid_points,
                triplets_at_q[:, i].flatten(),
                j * unique_vertices,
                mesh,
                grid_address,
                bz_map)
            interaction.set_phonons(np.unique(neighboring_grid_points))

    phono3c.triplets_integration_weights(
        g,
        frequency_points,
        thm.get_tetrahedra(),
        mesh,
        triplets_at_q,
        interaction.get_phonons()[0],
        grid_address,
        bz_map)

def _set_triplets_integration_weights_py(g, interaction, frequency_points, triplets_at_q=None):
    if triplets_at_q == None:
        triplets_at_q = interaction.get_triplets_at_q()[0]
    reciprocal_lattice = np.linalg.inv(interaction.get_primitive().get_cell())
    mesh = interaction.get_mesh_numbers()
    thm = TetrahedronMethod(reciprocal_lattice, mesh=mesh)
    grid_address = interaction.get_grid_address()
    tetrahedra_vertices = get_tetrahedra_vertices(
        thm.get_tetrahedra(),
        mesh,
        triplets_at_q,
        grid_address)
    interaction.set_phonons(np.unique(tetrahedra_vertices))
    frequencies = interaction.get_phonons()[0]
    num_band = frequencies.shape[1]
    for i, vertices in enumerate(tetrahedra_vertices):
        for j, k in list(np.ndindex((num_band, num_band))):
            f1_v = frequencies[vertices[0], j]
            f2_v = frequencies[vertices[1], k]
            thm.set_tetrahedra_omegas(f1_v + f2_v)
            thm.run(frequency_points)
            g0 = thm.get_integration_weight()
            g[0, i, :, j, k] = g0
            thm.set_tetrahedra_omegas(f1_v - f2_v)
            thm.run(frequency_points)
            g1 = thm.get_integration_weight()
            g[1, i, :, j, k] = g1
            thm.set_tetrahedra_omegas(-f1_v + f2_v)
            thm.run(frequency_points)
            g2 = thm.get_integration_weight()
            g[2, i, :, j, k] = g2

def get_tetrahedra_vertices(relative_address,
                            mesh,
                            triplets_at_q,
                            grid_address):
    grid_order = [1, mesh[0], mesh[0] * mesh[1]]
    num_triplets = len(triplets_at_q)
    vertices = np.zeros((num_triplets, 2, 24, 4), dtype='intc')
    for i, tp in enumerate(triplets_at_q):
        for j, adrs_shift in enumerate(
                (relative_address, -relative_address)):
            adrs = grid_address[tp[j + 1]] + adrs_shift # The tetrahedron build at g1 and g2
            gp = np.dot(adrs % mesh, grid_order)
            vertices[i, j] = gp
    return vertices

def reduce_triplets_by_permutation_symmetry(triplets, # all triplets are stored in one array
                                            mesh,
                                            first_mapping,
                                            first_rotation,
                                            second_mapping):
    unique_triplets, triplets_map, triplet_sequence = spg.get_reduced_triplets_permute_sym(triplets,
                                                             mesh,
                                                             first_mapping,
                                                             first_rotation,
                                                             second_mapping)

    return unique_triplets, triplets_map, triplet_sequence

def reduce_pairs_by_permutation_symmetry(pairs, # all triplets are stored in one array
                                            mesh,
                                            first_mapping,
                                            first_rotation,
                                            second_mapping):
    return spg.get_reduced_pairs_permute_sym(pairs,
                                             mesh,
                                             first_mapping,
                                             first_rotation,
                                             second_mapping)


def get_grid_point_from_address(grid, mesh):
    # X runs first in XYZ
    # (*In spglib, Z first is possible with MACRO setting.)
    return ((grid[0] + mesh[0]) % mesh[0] +
            ((grid[1] + mesh[1]) % mesh[1]) * mesh[0] +
            ((grid[2] + mesh[2]) % mesh[2]) * mesh[0] * mesh[1])

def invert_grid_point(grid_point, grid_address, mesh):
    # gp --> [address] --> [-address] --> inv_gp
    address = grid_address[grid_point]
    return get_grid_point_from_address(-address, mesh)

def get_ir_grid_points(mesh, primitive, mesh_shifts=[False, False, False], is_return_map=False):
    grid_mapping_table, grid_address = spg.get_ir_reciprocal_mesh(
        mesh,
        primitive,
        is_shift=np.where(mesh_shifts, 1, 0))
    ir_grid_points = np.unique(grid_mapping_table)
    weights = np.zeros_like(grid_mapping_table)
    for g in grid_mapping_table:
        weights[g] += 1
    ir_grid_weights = weights[ir_grid_points]
    if is_return_map:
        return ir_grid_points, ir_grid_weights, grid_address, grid_mapping_table
    else:
        return ir_grid_points, ir_grid_weights, grid_address


def reduce_grid_points(mesh_divisors,
                       grid_address,
                       dense_grid_points,
                       dense_grid_weights=None,
                       coarse_mesh_shifts=None):
    divisors = np.array(mesh_divisors, dtype=int)
    if (divisors == 1).all():
        coarse_grid_points = np.array(dense_grid_points, dtype=int)
        if dense_grid_weights is not None:
            coarse_grid_weights = np.array(dense_grid_weights, dtype=int)
    else:
        grid_weights = []
        if coarse_mesh_shifts is None:
            shift = [0, 0, 0]
        else:
            shift = np.where(coarse_mesh_shifts, divisors / 2, [0, 0, 0])
        modulo = grid_address[dense_grid_points] % divisors
        condition = (modulo == shift).all(axis=1)
        coarse_grid_points = np.extract(condition, dense_grid_points)
        if dense_grid_weights is not None:
            coarse_grid_weights = np.extract(condition, dense_grid_weights)

    if dense_grid_weights is None:
        return coarse_grid_points
    else:
        return coarse_grid_points, coarse_grid_weights

def from_coarse_to_dense_grid_points(dense_mesh,
                                     mesh_divisors,
                                     coarse_grid_points,
                                     coarse_grid_address,
                                     coarse_mesh_shifts=[False, False, False]):
    shifts = np.where(coarse_mesh_shifts, 1, 0)
    dense_grid_points = []
    for cga in coarse_grid_address[coarse_grid_points]:
        dense_address = cga * mesh_divisors + shifts * (mesh_divisors / 2)
        dense_grid_points.append(get_grid_point_from_address(dense_address,
                                                             dense_mesh))
    return np.intc(dense_grid_points)

def get_coarse_ir_grid_points(primitive, mesh, mesh_divs, coarse_mesh_shifts):
    if mesh_divs is None:
        mesh_divs = [1, 1, 1]
    mesh = np.intc(mesh)
    mesh_divs = np.intc(mesh_divs)
    coarse_mesh = mesh / mesh_divs
    if coarse_mesh_shifts is None:
        coarse_mesh_shifts = [False, False, False]
    (coarse_grid_points,
     coarse_grid_weights,
     coarse_grid_address) = get_ir_grid_points(
        coarse_mesh,
        primitive,
        mesh_shifts=coarse_mesh_shifts)
    grid_points = from_coarse_to_dense_grid_points(
        mesh,
        mesh_divs,
        coarse_grid_points,
        coarse_grid_address,
        coarse_mesh_shifts=coarse_mesh_shifts)
    grid_address = get_grid_address(mesh)

    return grid_points, coarse_grid_weights, grid_address

def get_rotations_for_star(grid, mesh, kpoint_operations, no_sym = False):
    if no_sym:
        rotations = [np.eye(3, dtype=int)]
    else:
        orig_address = grid
        orbits = []
        rotations = []
        for rot in kpoint_operations:
            rot_address = np.dot(rot, orig_address) % mesh
            in_orbits = False
            for orbit in orbits:
                if (rot_address == orbit).all():
                    in_orbits = True
                    break
            if not in_orbits:
                orbits.append(rot_address)
                rotations.append(rot)
    return np.array(rotations, dtype=np.int)

def get_point_group_reciprocal_at_grid(grid, mesh, kpoint_operations, no_sym=False):
    kpt_operation = []
    if no_sym:
        kpt_operation.append(np.eye(3, dtype=int))
    else:
        for rot in kpoint_operations:
            rot_address = np.dot(rot, grid) % mesh
            rot_address -= mesh * (rot_address> (mesh/2.0+1e-5))
            if (rot_address == grid).all():
                kpt_operation.append(rot)
    return np.array(kpt_operation, dtype=np.int)

def get_kgp_index_at_grid(grid, mesh, kpoint_operations, no_sym=False, symprec=1e-5):
    kpt_operation = []
    if no_sym:
        kpt_operation.append(0)
    else:
        for i, rot in enumerate(kpoint_operations):
            diff = (np.dot(rot, grid) - grid) / np.double(mesh)
            diff -= np.rint(diff)
            if (np.abs(np.divide(diff, mesh)) < symprec).all():
                kpt_operation.append(i)
    assert len(kpt_operation) > 0
    assert len(kpoint_operations) % len(kpt_operation) == 0 # requirement for subgroup
    return np.array(kpt_operation, dtype=np.int)


def get_kpoint_group(mesh, point_operations, qpoints = [[0., 0., 0.]], is_time_reversal=True):
    return spg.get_kpoint_group(mesh,  point_operations, qpoints, is_time_reversal)


if __name__ == '__main__':
    # This checks if ir_grid_points.yaml gives correct dense grid points
    # that are converted from coase grid points by comparing with 
    # mesh.yaml.
    
    import yaml
    import sys
    
    data1 = yaml.load(open(sys.argv[1]))['ir_grid_points'] # ir_grid_points.yaml
    data2 = yaml.load(open(sys.argv[2]))['phonon'] # phonpy mesh.yaml
    
    weights1 = np.array([x['weight'] for x in data1])
    weights2 = np.array([x['weight'] for x in data2])
    print (weights1 == weights2).sum()
    q1 = np.array([x['q-point'] for x in data1])
    q2 = np.array([x['q-position'] for x in data2])
    print (q1 == q2).all(axis=1).sum()
    for d in (q1 - q2):
        print d


def get_pointgroup_operations(point_operations_real):
    exist_r_inv = False
    for rot in point_operations_real:
        if (rot + np.eye(3, dtype='intc') == 0).all():
            exist_r_inv = True
            break

    point_operations = [rot.T for rot in point_operations_real]

    if not exist_r_inv:
        point_operations += [-rot.T for rot in point_operations_real]

    return np.array(point_operations)


def get_group_summation(group_operations):
    length = len(group_operations)
    mapping = np.zeros((length, length), dtype="intc")
    for i in range(length):
        mi = group_operations[i]
        for j in range(length):
            mj = group_operations[j]
            m = np.dot(mi, mj)
            is_close = np.all(group_operations == m, axis=(1,2))
            mapping[i,j] = np.where(is_close)[0]
    return mapping

def get_group_inversion(group_operations):
    length = len(group_operations)
    mapping = np.zeros(length, dtype="intc")
    for i in range(length):
        mi = group_operations[i]
        m = np.linalg.inv(mi)
        is_close = np.all(group_operations == m, axis=(1,2))
        mapping[i] = np.where(is_close)[0]
    return mapping

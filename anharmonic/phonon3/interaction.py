import numpy as np
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC, get_smallest_vectors
from phonopy.phonon.band_structure import estimate_band_connection
from phonopy.structure.symmetry import Symmetry
from phonopy.structure.spglib import get_mappings
from phonopy.units import VaspToTHz, total_time
from anharmonic.phonon3.real_to_reciprocal import RealToReciprocal
from anharmonic.phonon3.reciprocal_to_normal import ReciprocalToNormal
from anharmonic.phonon3.triplets import get_triplets_at_q_crude, get_BZ_triplets_at_q, get_nosym_triplets_at_q, \
     get_kpoint_group, reduce_triplets_by_permutation_symmetry, get_bz_grid_address, get_triplets_at_q
from anharmonic.file_IO import write_triplets_to_hdf5, read_amplitude_from_hdf5_all, read_amplitude_from_hdf5_at_grid,\
    write_amplitude_to_hdf5_at_grid, write_amplitude_to_hdf5_all
from phonopy.phonon.group_velocity import degenerate_sets
import os, sys

def find_index(a, b): #index of b elements in a such that a[seq] = b
    return np.array([list(a).index(j) for j in b])

def get_dynamical_matrix(fc2,
                         supercell,
                         primitive,
                         nac_params=None,
                         frequency_scale_factor=None,
                         decimals=None,
                         symprec=1e-5):
    if nac_params is None:
        dm = DynamicalMatrix(
            supercell,
            primitive,
            fc2,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals,
            symprec=symprec)
    else:
        dm = DynamicalMatrixNAC(
            supercell,
            primitive,
            fc2,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals,
            symprec=symprec)
        dm.set_nac_params(nac_params)
    return dm

def set_phonon_py(grid_point,
                  phonon_done,
                  frequencies,
                  eigenvectors,
                  degeneracies,
                  grid_address,
                  mesh,
                  dynamical_matrix,
                  frequency_factor_to_THz,
                  lapack_zheev_uplo):
    gp = grid_point
    if phonon_done[gp] == 0:
        phonon_done[gp] = 1
        q = grid_address[gp].astype('double') / mesh
        dynamical_matrix.set_dynamical_matrix(q)
        dm = dynamical_matrix.get_dynamical_matrix()
        eigvals, eigvecs = np.linalg.eigh(dm, UPLO=lapack_zheev_uplo)
        eigvals = eigvals.real
        frequencies[gp] = (np.sqrt(np.abs(eigvals)) * np.sign(eigvals)
                           * frequency_factor_to_THz)
        eigenvectors[gp] = eigvecs
        degeneracies[gp] = degenerate_at_grid(frequencies[gp])

def degenerate_at_grid(frequency_at_q):
    degeneracy = np.zeros_like(frequency_at_q, dtype="intc")
    deg_sets = degenerate_sets(frequency_at_q)
    for ele in deg_sets:
        for sub_ele in ele:
            degeneracy[sub_ele]=ele[0]
    return degeneracy

def set_phonon_c(dm,
                 frequencies,
                 eigenvectors,
                 degeneracies,
                 phonon_done,
                 grid_points,
                 grid_address,
                 mesh,
                 frequency_factor_to_THz,
                 nac_q_direction,
                 lapack_zheev_uplo):
    import anharmonic._phono3py as phono3c
    svecs, multiplicity = dm.get_shortest_vectors()
    masses = np.array(dm.get_primitive().get_masses(), dtype='double')
    rec_lattice = np.array(
        np.linalg.inv(dm.get_primitive().get_cell()), dtype='double', order='C')
    if dm.is_nac():
        born = dm.get_born_effective_charges()
        nac_factor = dm.get_nac_factor()
        dielectric = dm.get_dielectric_constant()
    else:
        born = None
        nac_factor = 0
        dielectric = None
    if degeneracies == None:
        degeneracies = np.ones((len(frequencies), 3 * len(masses)), dtype="intc")
        degeneracies *= np.arange(3*len(masses))

    phono3c.phonons_at_gridpoints(
        frequencies,
        eigenvectors,
        degeneracies,
        phonon_done,
        grid_points,
        grid_address,
        np.array(mesh, dtype='intc'),
        dm.get_force_constants(),
        svecs,
        multiplicity,
        masses,
        dm.get_primitive_to_supercell_map(),
        dm.get_supercell_to_primitive_map(),
        frequency_factor_to_THz,
        born,
        dielectric,
        rec_lattice,
        nac_q_direction,
        nac_factor,
        lapack_zheev_uplo)

class Interaction:
    def __init__(self,
                 fc3,
                 supercell,
                 primitive,
                 mesh,
                 band_indices=None,
                 frequency_factor_to_THz=VaspToTHz,
                 is_nosym=False,
                 symmetrize_fc3_q=False,
                 symprec=1e-3,
                 triplet_cut_super = None,
                 triplet_cut_prim = None,
                 cutoff_frequency=None,
                 cutoff_hfrequency=None,
                 cutoff_delta = None,
                 is_triplets_dispersed=False,
                 is_read_amplitude=False,
                 is_write_amplitude = False,
                 is_triplets_permute=True,
                 lapack_zheev_uplo='L'):
        self._fc3 = fc3
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = np.intc(mesh)
        num_band = primitive.get_number_of_atoms() * 3
        if band_indices is None:
            self._band_indices = np.arange(num_band, dtype='intc')
        else:
            self._band_indices = band_indices.astype("intc")
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._symprec = symprec
        self._is_tripelts_permute=is_triplets_permute
        natom_super = supercell.get_number_of_atoms()
        natom_prim = primitive.get_number_of_atoms()
        if triplet_cut_super is None:
            self._triplet_cut_super=np.zeros((natom_super, natom_super, natom_super), dtype='bool')
        else:
            self._triplet_cut_super=triplet_cut_super

        if triplet_cut_prim is None:
            self._triplet_cut_prim=np.zeros((natom_prim, natom_prim, natom_prim), dtype='bool')
        else:
            self._triplet_cut_prim=triplet_cut_prim

        if cutoff_delta is None:
            self._cutoff_delta = 1000.0
        else:
            self._cutoff_delta = cutoff_delta

        if cutoff_frequency is None:
            self._cutoff_frequency = 0
        else:
            self._cutoff_frequency = cutoff_frequency

        if cutoff_hfrequency is None:
            self._cutoff_hfrequency = 10000.0 # THz
        elif symmetrize_fc3_q:
            print "Warning: symmetryze_fc3_q and cutoff_hfrequency are not compatible"
            self._cutoff_hfrequency = 10000.0
        else:
            self._cutoff_hfrequency = cutoff_hfrequency
        self._is_nosym = is_nosym
        self._symmetrize_fc3_q = symmetrize_fc3_q
        self._lapack_zheev_uplo = lapack_zheev_uplo
        self._symmetry = Symmetry(primitive, symprec=symprec)
        if self._is_nosym:
            self._point_group_operations = np.array([np.eye(3)],dtype="intc")
            self._kpoint_operations = get_kpoint_group(self._mesh, self._point_group_operations, is_time_reversal=False)
        else:
            self._point_group_operations = self._symmetry.get_pointgroup_operations()
            self._kpoint_operations = get_kpoint_group(self._mesh, self._point_group_operations)

        if self.is_nosym():
            grid_mapping = np.arange(np.prod(self._mesh))
            grid_mapping_rots = np.zeros(len(grid_mapping), dtype="intc")
        else:
            grid_mapping, grid_mapping_rots = get_mappings(self._mesh,
                                        self.get_point_group_operations(),
                                        qpoints=np.array([0,0,0],dtype="double"))

        self._svecs, self._multiplicity = get_smallest_vectors(self._supercell,
                                                                 self._primitive,
                                                                 self._symprec)
        self._grid_mapping = grid_mapping
        self._grid_mapping_rot = grid_mapping_rots

        self._is_read_amplitude = is_read_amplitude
        self._is_write_amplitude = is_write_amplitude
        self._grid_point = None
        self._triplets_at_q = None
        self._weights_at_q = None
        self._grid_address = None
        self._interaction_strength = None
        self._phonon_done = None
        self._frequencies = None
        self._eigenvectors = None
        self._degenerates = None
        self._grid_points = None
        self._dm = None
        self._nac_q_direction = None
        self._triplets = None
        self._triplets_mappings = None
        self._triplets_sequence = None
        self._unique_triplets = None
        self._amplitude_all = None
        self._is_dispersed = is_triplets_dispersed
        self._allocate_phonon()

    def set_is_write_amplitude(self, is_write_amplitude=False):
        self._is_write_amplitude = is_write_amplitude

    def get_is_write_amplitude(self):
        return self._is_write_amplitude

    def set_is_read_amplitude(self, is_read_amplitude=False):
        self._is_read_amplitude = is_read_amplitude

    def get_is_read_amplitude(self):
        return self._is_read_amplitude

    def get_is_dispersed(self):
        return self._is_dispersed

    def get_degeneracy(self):
        return self._degenerates

    def get_amplitude_all(self):
        return self._amplitude_all

    def set_is_disperse(self, is_disperse):
        self._is_dispersed = is_disperse

    def set_is_symmetrize_fc3_q(self, is_symmetrize_fc3q):
        self._symmetrize_fc3_q = is_symmetrize_fc3q

    def get_is_symmetrize_fc3_q(self):
        return self._symmetrize_fc3_q

    def run(self, g_skip=None, lang='C', log_level=0):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_triplets = len(self._triplets_at_q)
        self._interaction_strength = np.zeros(
            (num_triplets, len(self._band_indices), num_band, num_band),
            dtype='double')
        if self._is_dispersed:
            unitrip_indices = self._triplets_uniq_index_at_grid
            mapping = self._triplets_maping_at_grid
            triplets_sequence = self._triplet_sequence_at_grid
            if log_level>0:
                print "Triplets number to be calculated: %d/%d" %(len(unitrip_indices), len(self._interaction_strength))

            if self._is_read_amplitude:
                self._interaction_strength_reduced = \
                    read_amplitude_from_hdf5_at_grid(self._mesh, self._grid_point)
                if self._interaction_strength_reduced is not None:
                    self.set_phonons(lang=lang)
                else:
                    print "Reading amplitude in the disperse mode unsuccessfully. Reverting to writing mode!"
                    self._is_read_amplitude = False
                    self._is_write_amplitude = True

            if not self._is_read_amplitude:
                self._interaction_strength_reduced = np.zeros(
                    (len(self._triplets_uniq_index_at_grid), len(self._band_indices), num_band, num_band),
                    dtype='double')
                self._triplets_at_q_reduced = self._triplets_at_q[unitrip_indices].copy()
                if g_skip is None:
                    g_skip = np.zeros_like(self._interaction_strength_reduced, dtype="bool")


                if lang == 'C':
                    self._run_c(g_skip=g_skip)
                else:
                    self._run_py(g_skip=g_skip)

            import anharmonic._phono3py as phono3c
            if len(self._band_indices) != self._interaction_strength.shape[-1]:
                assert (triplets_sequence[:,0] == 0).all()
            phono3c.interaction_from_reduced(self._interaction_strength,
                                         self._interaction_strength_reduced.astype("double").copy(),
                                         mapping.astype("intc"),
                                         triplets_sequence.astype("byte"))
            #debugging
            # interaction_strength = self._interaction_strength.copy()
            # interaction_strength_reduced = self._interaction_strength_reduced.copy()
            # self._interaction_strength_reduced = np.zeros(
            #         (len(self._triplets_at_q), len(self._band_indices), num_band, num_band),
            #         dtype='double')
            # self._triplets_at_q_reduced = self._triplets_at_q
            # self._run_c()
            # self._interaction_strength = self._interaction_strength_reduced.copy()
            # print np.abs(self._interaction_strength_reduced - interaction_strength).max() / self._interaction_strength_reduced.max(),
            # print np.unravel_index(np.abs(self._interaction_strength_reduced - interaction_strength).argmax(), self._interaction_strength_reduced.shape)

            if self._is_write_amplitude:
                write_amplitude_to_hdf5_at_grid(self._mesh, grid=self._grid_point, amplitude=self._interaction_strength_reduced)


        else:
            if self._is_read_amplitude:
                import anharmonic._phono3py as phono3c
                phono3c.interaction_from_reduced(self._interaction_strength,
                                                 self._amplitude_all,
                                                 self._triplets_mappings[self._i].astype("intc"),
                                                 self._triplets_sequence[self._i].astype("byte"))
                self.set_phonons(lang=lang)
            else:
                map_to_unique, reverse_map = np.unique(self._triplets_mappings[self._i], return_inverse=True)
                undone_num = np.extract(self._triplets_done[map_to_unique]==0, map_to_unique)
                self._triplets_at_q_reduced = self._unique_triplets[undone_num]

                if log_level>0:
                    print "Triplets number to be calculted: %d/%d" %(len(self._triplets_at_q_reduced), len(self._interaction_strength))

                self._interaction_strength_reduced = np.zeros(
                    (len(self._triplets_at_q_reduced), len(self._band_indices), num_band, num_band),
                    dtype='double')
                if g_skip is None:
                    g_skip = np.zeros_like(self._interaction_strength_reduced, dtype="bool")
                if lang == 'C':
                    self._run_c(g_skip=g_skip)
                else:
                    self._run_py(g_skip=g_skip)
                self._amplitude_all[undone_num] = self._interaction_strength_reduced[:]
                import anharmonic._phono3py as phono3c
                phono3c.interaction_from_reduced(self._interaction_strength,
                                                 self._amplitude_all,
                                                 self._triplets_mappings[self._i].astype("intc"),
                                                 self._triplets_sequence[self._i].astype("byte"))
                self._triplets_done[undone_num] = True

        ##Added for my own purpose, please delete it
        self._criteria = 0.55
        self._lcriteria = 0.1
        is_anyq_on_bound = (np.abs(self._grid_address[self._triplets_at_q]) > self._criteria * self._mesh).any(axis=(1,2))
        is_uprocess = np.any(self._grid_address[self._triplets_at_q].sum(axis=1)!=0, axis=1)
        is_anyq_at_center = (np.abs(self._grid_address[self._triplets_at_q]) < self._lcriteria * self._mesh).any(axis=(1,2))
        self._interaction_strength[np.where(np.logical_and(is_uprocess, is_anyq_at_center))] = 0.
        self._interaction_strength[np.where(is_anyq_on_bound)] = 0.

    # @total_time.timeit
    def set_phonons(self, grid_points=None, lang = "C"):
        if lang == "C":
            self._set_phonon_c(grid_points)
        else:
            if grid_points is None:
                for i, grid_triplet in enumerate(self._triplets_at_q):
                    for gp in grid_triplet:
                        self._set_phonon_py(gp)
            else:
                for gp in enumerate(grid_points):
                    self._set_phonon_py(gp)

    def set_phonons_all(self, is_band_connection=True, lang='C', log_level=1):
        if log_level:
            print "calculate phonon frequencies of all phonon mode..."
        grid_points = np.arange(len(self._grid_address))
        if lang == "C":
            self._set_phonon_c(grid_points)
        else:
            for gp in grid_points:
                self._set_phonon_py(gp)
        if is_band_connection:
            self._set_band_connection()

    def _set_band_connection(self):
        nqpoint, nband = self._frequencies.shape
        is_non_degenerate = np.all(self._degenerates == np.arange(nband), axis=1)
        connect_done = np.zeros(nqpoint, dtype='bool')
        start = np.where(is_non_degenerate)[0][0]

        while True:
            connect_done[start] = True
            neighbors = np.argsort(np.sum(np.abs(self._grid_address - self._grid_address[start]), axis=1))
            undone = np.extract(connect_done[neighbors]==False, neighbors)

            if not (len(undone) == 0 or (is_non_degenerate[undone] == False).all()):
                new = np.extract(is_non_degenerate[undone], undone)[0]
                new_neighbors = np.argsort(np.sum(np.abs(self._grid_address - self._grid_address[new]), axis=1))
                new_neighbor_done = np.extract(connect_done[new_neighbors], new_neighbors)[0]
                bo = estimate_band_connection(self._eigenvectors[new_neighbor_done],
                                              self._eigenvectors[new],
                                              np.arange(nband))
                if bo is not None:
                    self._frequencies[new] = (self._frequencies[new])[bo]
                    self._eigenvectors[new] = (self._eigenvectors[new].T)[bo].T
                    # Since all the phonons here are non-degenerate, the step is missed
                start = new

            else:
                break

        for new in undone:
            new_neighbors = np.argsort(np.sum(np.abs(self._grid_address - self._grid_address[new]), axis=1))
            new_neighbor_done = np.extract(connect_done[new_neighbors], new_neighbors)[0]
            deg = [np.where(self._degenerates[new] == j)[0] for j in np.unique(self._degenerates[new])]
            bo = estimate_band_connection(self._eigenvectors[new_neighbor_done],
                                          self._eigenvectors[new],
                                          np.arange(nband),
                                          degenerate_sets=deg)
            if bo is not None:
                self._frequencies[new] = (self._frequencies[new])[bo]
                self._eigenvectors[new] = (self._eigenvectors[new].T)[bo].T
                self._degenerates[new] = (self._degenerates[new])[bo].copy()

    def get_interaction_strength(self):
        return self._interaction_strength

    def get_mesh_numbers(self):
        return self._mesh

    def get_phonons(self):
        return (self._frequencies,
                self._eigenvectors,
                self._phonon_done)

    def get_dynamical_matrix(self):
        return self._dm

    def get_primitive(self):
        return self._primitive

    def get_supercell(self):
        return self._supercell

    def get_point_group_operations(self):
        return self._point_group_operations

    def get_triplets_at_q(self):
        return self._triplets_at_q, self._weights_at_q

    def get_triplets_done(self):
        return self._triplets_done

    def get_triplets_sequence_at_q_disperse(self):
        return self._triplet_sequence_at_grid

    def get_triplets_mapping_at_q_disperse(self):
        return self._triplets_maping_at_grid

    def get_triplets_at_q_nu(self):
        triplet_address = self.get_triplet_address()
        triplets_N=[]
        triplets_U=[]
        for i, t in enumerate(triplet_address):
            if (np.sum(t, axis=0)==0).all():
                triplets_N.append(i)
            else:
                triplets_U.append(i)
        return (triplets_N, triplets_U)
    def get_triplet_address(self):
        return self._triplets_address
    def get_grid_address(self):
        return self._grid_address

    def get_band_indices(self):
        return self._band_indices

    def get_frequency_factor_to_THz(self):
        return self._frequency_factor_to_THz

    def get_lapack_zheev_uplo(self):
        return self._lapack_zheev_uplo

    def is_nosym(self):
        return self._is_nosym

    def get_bz_map(self):
        return self._bz_map

    def get_bz_to_pp_map(self):
        return self._bz_to_pp_map

    def get_grid_mapping(self):
        return self._grid_mapping

    def get_grid_mapping_rot(self):
        return self._grid_mapping_rot

    def get_cutoff_frequency(self):
        return self._cutoff_frequency

    def get_cutoff_hfrequency(self):
        return self._cutoff_hfrequency

    def get_triplets_mapping_at_grid(self):
        if not self._is_dispersed:
            return self._triplets_mappings[self._i]
        else:
            return self._triplets_maping_at_grid

    def get_triplets_sequence_at_grid(self):
        return self._triplets_sequence[self._i]

    @total_time.timeit
    def set_grid_point(self, grid_point, i=None, stores_triplets_map=False):
        if i==None:
            self._grid_point = grid_point
            if self._grid_points is not None:
                self._i = np.where(grid_point == self._grid_points)[0][0]
        else:
            self._i = i
            self._grid_point = self._grid_points[i]
        if self._triplets is not None:
            self._triplets_at_q = self._triplets[self._i]
            self._weights_at_q = self._weights[self._i]
            self._triplets_address = self._grid_address[self._triplets_at_q]
            if self._is_dispersed:
                self._triplets_uniq_index_at_grid = self._unique_triplets[self._i]
                self._triplets_maping_at_grid = self._triplets_mappings[self._i]
                self._triplet_sequence_at_grid = self._triplets_sequence[self._i]
        else:
            reciprocal_lattice = np.linalg.inv(self._primitive.get_cell())
            if self._is_nosym:
                (triplets_at_q,
                 weights_at_q,
                 grid_address,
                 bz_map,
                 triplets_map_at_q,
                 ir_map_at_q) = get_nosym_triplets_at_q(
                     grid_point,
                     self._mesh,
                     reciprocal_lattice,
                     stores_triplets_map=stores_triplets_map)
                if self._is_dispersed:
                    self._triplets_uniq_index_at_grid = np.arange(len(triplets_at_q), dtype="intc")
                    self._triplets_maping_at_grid = np.arange(len(triplets_at_q), dtype="intc")
                    self._triplet_sequence_at_grid = np.arange(3, dtype="intc")[np.newaxis].repeat(len(triplets_at_q))

            else:
                (triples_at_q_crude,
                 weights_at_q,
                 grid_address,
                 grid_map)=\
                    get_triplets_at_q_crude(grid_point, self._mesh, self._point_group_operations)

                (triplets_at_q,
                 weights,
                 bz_grid_address,
                 bz_map)=\
                    get_BZ_triplets_at_q(grid_point, self._mesh, reciprocal_lattice, grid_address, grid_map)

                if self._is_dispersed:
                    (unique_triplet_nums,
                     triplets_mappings,
                     triplet_sequence) = \
                        reduce_triplets_by_permutation_symmetry([triples_at_q_crude],
                                                                self._mesh,
                                                                first_mapping=self._grid_mapping,
                                                                first_rotation=self._kpoint_operations[self._grid_mapping_rot],
                                                                second_mapping=np.array([grid_map]))
                    self._triplets_uniq_index_at_grid = unique_triplet_nums
                    self._triplets_maping_at_grid = triplets_mappings[0]
                    self._triplet_sequence_at_grid = triplet_sequence[0]
            sum_qs = bz_grid_address[triplets_at_q].sum(axis=1)
            resi = sum_qs % self._mesh
            if (resi != 0).any():
                triplets = triplets_at_q[np.where(resi != 0)[0]]
                print "============= Warning =================="
                print triplets
                print sum_qs
                print "============= Warning =================="

            self._triplets_at_q = triplets_at_q
            self._weights_at_q = weights_at_q
            self._ir_map_at_q = grid_map

    def set_grid_points(self, grid_points):
        self._grid_points = grid_points
        reciprocal_lattice = np.linalg.inv(self._primitive.get_cell())
        self._triplets = []
        self._weights = []
        second_mappings = []
        self._triplets_sequence = []
        crude_triplets = []
        total_triplet_num = 0
        # unique triplets at all grid points
        self._unique_triplets = []
        self._triplets_mappings = []
        self._triplets_sequence = []
        for g, grid_point in enumerate(grid_points):
            (triples_at_q_crude,
             weights_at_q,
             grid_address,
             grid_map)=\
                get_triplets_at_q_crude(grid_point, self._mesh, self._point_group_operations)

            (triplets_at_q,
             weights,
             bz_grid_address,
             bz_map)=\
                get_BZ_triplets_at_q(grid_point, self._mesh, reciprocal_lattice, grid_address, grid_map)

            crude_triplets.append(triples_at_q_crude)
            self._triplets.append(triplets_at_q)
            self._weights.append(weights_at_q)
            total_triplet_num += len(triplets_at_q)
            second_mappings.append(grid_map)

            if self._is_dispersed:
                (unique_triplet_nums_grid,
                 triplets_mappings_grid,
                 triplet_sequence_grid) = \
                    reduce_triplets_by_permutation_symmetry([triples_at_q_crude],
                                                            self._mesh,
                                                            first_mapping=self._grid_mapping,
                                                            first_rotation=self._kpoint_operations[self._grid_mapping_rot],
                                                            second_mapping=np.array([grid_map]))
                self._unique_triplets.append(unique_triplet_nums_grid)
                self._triplets_mappings.append(triplets_mappings_grid[0])
                self._triplets_sequence.append(triplet_sequence_grid[0])

        if not self._is_dispersed:
            unique_triplet_num, triplets_mappings, triplet_sequence = reduce_triplets_by_permutation_symmetry(crude_triplets,
                                                    self._mesh,
                                                    first_mapping=self._grid_mapping,
                                                    first_rotation=self._kpoint_operations[self._grid_mapping_rot],
                                                    second_mapping=np.vstack(second_mappings))

            print "Number of total unique triplets after permutation symmetry: %d/ %d" %(len(unique_triplet_num), total_triplet_num)
            self._unique_triplets = np.vstack(self._triplets)[unique_triplet_num]
            self._triplets_done = np.zeros(len(unique_triplet_num), dtype="byte")
            self._triplets_mappings = triplets_mappings #map to the index of the triplets in the unique_triplet_num
            self._second_mappings = second_mappings
            self._triplets_sequence = triplet_sequence
            nband = 3 * self._primitive.get_number_of_atoms()
            try:
                self._amplitude_all = np.zeros((len(self._unique_triplets), nband, nband, nband), dtype="double")
            except MemoryError:
                print "A memory error occurs in allocating the whole interaction strength array"
                print "--disperse is recommended as an alternative method"
                sys.exit(1)
        if self._is_read_amplitude:
            self.read_amplitude_all()

    def read_amplitude_all(self):
        if not self._is_dispersed:
            read_amplitude_from_hdf5_all(self._amplitude_all, self._mesh, self.is_nosym())

    def release_amplitude_all(self):
        del self._amplitude_all
        self._amplitude_all = None

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             frequency_scale_factor=None,
                             decimals=None):
        self._dm = get_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals,
            symprec=self._symprec)

    def set_nac_q_direction(self, nac_q_direction=None):
        if nac_q_direction is not None:
            self._nac_q_direction = np.double(nac_q_direction)

    def _run_c(self, g_skip=None):
        import anharmonic._phono3py as phono3c
        if g_skip is None:
            g_skip = np.zeros_like(self._interaction_strength_reduced, dtype="bool")
        assert g_skip.shape == self._interaction_strength_reduced.shape
        self._set_phonon_c()
        masses = np.double(self._primitive.get_masses())
        p2s = np.intc(self._primitive.get_primitive_to_supercell_map())
        s2p = np.intc(self._primitive.get_supercell_to_primitive_map())
        atc=np.intc(self._triplet_cut_super) # int type
        atc_prim = np.intc(self._triplet_cut_prim) # int type
        phono3c.interaction(self._interaction_strength_reduced,
                            self._frequencies,
                            self._eigenvectors,
                            self._triplets_at_q_reduced.copy(),
                            self._grid_address,
                            self._mesh,
                            self._fc3,
                            atc,
                            atc_prim,
                            g_skip,
                            self._svecs,
                            self._multiplicity,
                            np.double(masses),
                            p2s,
                            s2p,
                            self._band_indices,
                            self._symmetrize_fc3_q,
                            self._cutoff_frequency,
                            self._cutoff_hfrequency,
                            self._cutoff_delta)
        phono3c.interaction_degeneracy_grid(self._interaction_strength_reduced,
                                            self._degenerates,
                                            self._triplets_at_q_reduced.astype('intc').copy(),
                                            self._band_indices.astype('intc'))
        # from itertools import permutations
        # interaction = np.zeros((6,) + self._interaction_strength_reduced.shape, dtype='double')
        # for i, permute in enumerate(permutations((0,1,2))):
        #     interaction0 = np.zeros_like(self._interaction_strength_reduced)
        #     new = ''.join(np.array(list('ijk'))[list(permute)])
        #     phono3c.interaction(interaction0,
        #                         self._frequencies,
        #                         self._eigenvectors,
        #                         self._triplets_at_q_reduced[:, permute].copy(),
        #                         self._grid_address,
        #                         self._mesh,
        #                         self._fc3,
        #                         atc,
        #                         np.zeros_like(g_skip),
        #                         svecs,
        #                         multiplicity,
        #                         np.double(masses),
        #                         p2s,
        #                         s2p,
        #                         self._band_indices,
        #                         False,
        #                         self._cutoff_frequency,
        #                         self._cutoff_hfrequency,
        #                         self._cutoff_delta)
        #     interaction[i,:] = np.einsum("N%s->Nijk"%new, interaction0)
        # diff = np.abs(interaction - interaction[0])
        # print np.unravel_index(diff.argmax(), diff.shape), diff.max()


    def _set_phonon_c(self, grid_points=None):
        import anharmonic._phono3py as phono3c

        svecs, multiplicity = self._dm.get_shortest_vectors()
        masses = np.double(self._dm.get_primitive().get_masses())
        rec_lattice = np.double(
            np.linalg.inv(self._dm.get_primitive().get_cell())).copy()
        if self._dm.is_nac():
            born = self._dm.get_born_effective_charges()
            nac_factor = self._dm.get_nac_factor()
            dielectric = self._dm.get_dielectric_constant()
        else:
            born = None
            nac_factor = 0
            dielectric = None
        if grid_points == None:
            phono3c.phonon_triplets(self._frequencies,
                                    self._eigenvectors,
                                    self._degenerates,
                                    self._phonon_done,
                                    self._triplets_at_q,
                                    self._grid_address,
                                    self._mesh,
                                    self._dm.get_force_constants(),
                                    svecs,
                                    multiplicity,
                                    masses,
                                    self._dm.get_primitive_to_supercell_map(),
                                    self._dm.get_supercell_to_primitive_map(),
                                    self._frequency_factor_to_THz,
                                    born,
                                    dielectric,
                                    rec_lattice,
                                    self._nac_q_direction,
                                    nac_factor,
                                    self._lapack_zheev_uplo)
        else:
            set_phonon_c(self._dm,
                         self._frequencies,
                         self._eigenvectors,
                         self._degenerates,
                         self._phonon_done,
                         grid_points.astype("intc").copy(),
                         self._grid_address,
                         self._mesh,
                         self._frequency_factor_to_THz,
                         self._nac_q_direction,
                         self._lapack_zheev_uplo)

    def _run_py(self, g_skip=None):
        if g_skip is None:
            g_skip = np.zeros_like(self._interaction_strength_reduced, dtype="bool")
        else:
            assert g_skip.shape == self._interaction_strength_reduced.shape
        r2r = RealToReciprocal(self._fc3,
                               self._supercell,
                               self._primitive,
                               self._mesh,
                               symprec=self._symprec,
                               atom_triplet_cut=self._triplet_cut_super)

        r2n = ReciprocalToNormal(self._primitive,
                                 self._frequencies,
                                 self._eigenvectors,
                                 cutoff_frequency=self._cutoff_frequency,
                                 cutoff_hfrequency=self._cutoff_hfrequency,
                                 cutoff_delta=self._cutoff_delta)

        for i, grid_triplet in enumerate(self._triplets_at_q_reduced):
            print "%d / %d" % (i + 1, len(self._triplets_at_q_reduced))
            r2r.run(self._grid_address[grid_triplet], self._symmetrize_fc3_q)
            fc3_reciprocal = r2r.get_fc3_reciprocal()
            for gp in grid_triplet:
                self._set_phonon_py(gp)
            r2n.run(fc3_reciprocal, grid_triplet, g_skip = g_skip[i])
            self._interaction_strength_reduced[i] = r2n.get_reciprocal_to_normal()

    def _set_phonon_py(self, grid_point):
        set_phonon_py(grid_point,
                      self._phonon_done,
                      self._frequencies,
                      self._eigenvectors,
                      self._degenerates,
                      self._grid_address,
                      self._mesh,
                      self._dm,
                      self._frequency_factor_to_THz,
                      self._lapack_zheev_uplo)

    def _allocate_phonon(self):
        primitive_lattice = np.linalg.inv(self._primitive.get_cell())
        self._grid_address, self._bz_map, self._bz_to_pp_map = get_bz_grid_address(
            self._mesh, primitive_lattice, with_boundary=True, is_bz_map_to_pp=True)
        num_band = self._primitive.get_number_of_atoms() * 3
        num_grid = len(self._grid_address)
        self._phonon_done = np.zeros(num_grid, dtype='byte')
        self._frequencies = np.zeros((num_grid, num_band), dtype='double')
        self._degenerates = np.zeros((num_grid, num_band), dtype="intc")
        self._eigenvectors = np.zeros((num_grid, num_band, num_band),
                                      dtype='complex128')


    def write_amplitude_all(self):
        if self.get_is_write_amplitude():
            if self.get_amplitude_all() is not None:
                write_amplitude_to_hdf5_all(self.get_amplitude_all(),
                                            self._mesh,
                                            is_nosym=self.is_nosym())
            self.set_is_read_amplitude(True)
            self.set_is_write_amplitude(False)
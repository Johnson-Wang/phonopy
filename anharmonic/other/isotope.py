import numpy as np
from anharmonic.phonon3.interaction import get_dynamical_matrix,  set_phonon_py
from anharmonic.phonon3.interaction import set_phonon_c
from anharmonic.phonon3.triplets import get_bz_grid_address
from anharmonic.phonon3.imag_self_energy import gaussian, occupation
from phonopy.structure.tetrahedron_method import TetrahedronMethod
from phonopy.phonon.tetrahedron_mesh import get_tetrahedra_frequencies
import phonopy.structure.spglib as spg
from phonopy.structure.symmetry import Symmetry
from phonopy.units import VaspToTHz
from phonopy.structure.atoms import isotope_data

def get_mass_variances(primitive):
    symbols = primitive.get_chemical_symbols()
    mass_variances = []
    for s in symbols:
        masses = np.array([x[1] for x in isotope_data[s]])
        fractions = np.array([x[2] for x in isotope_data[s]])
        m_ave = np.dot(masses, fractions)
        g = np.dot(fractions, (1 - masses / m_ave) ** 2)
        mass_variances.append(g)

    return np.array(mass_variances, dtype='double')

class CollisionIso:
    def __init__(self,
                 mesh,
                 primitive,
                 mass_variances=None, # length of list is num_atom.
                 band_indices=None,
                 sigma=None,
                 frequency_factor_to_THz=VaspToTHz,
                 temperatures=None,
                 symprec=1e-5,
                 cutoff_frequency=None,
                 is_write_collision=False,
                 is_read_collision=False,
                 lapack_zheev_uplo='L',
                 is_nosym=False):
        self._mesh = np.array(mesh, dtype='intc')
        if mass_variances is None:
            self._mass_variances = get_mass_variances(primitive)
        else:
            self._mass_variances = np.array(mass_variances, dtype='double')
        self._primitive = primitive
        self._band_indices = band_indices
        self._sigma = sigma
        self._symprec = symprec
        if cutoff_frequency is None:
            self._cutoff_frequency = 1e-4
        else:
            self._cutoff_frequency = cutoff_frequency
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._lapack_zheev_uplo = lapack_zheev_uplo
        self._temps = temperatures
        self._nac_q_direction = None
        self._grid_address = None
        self._bz_map = None
        self._grid_points2 = None
        self._phonon_done = None
        self._frequencies = None
        self._eigenvectors = None
        self._dm = None
        self._grid_point = None
        self._gamma = None
        self._tetrahedron_method = None
        self._is_extra_frequency = False
        self._is_nosym = is_nosym
        self._is_read_col = is_read_collision
        self._is_write_col = is_write_collision
        
    def set_grid_point(self,
                       grid_point,
                       grid_points2=None,
                       weights2 = None):
        "The grid_point is numbered using the bz scheme"
        self._grid_point = grid_point
        if self._grid_address is None:
            primitive_lattice = np.linalg.inv(self._primitive.get_cell())
            self._grid_address, self._bz_map, self._bz_to_pp_map = get_bz_grid_address(
                self._mesh, primitive_lattice, with_boundary=True, is_bz_map_to_pp=True)

        if grid_points2 is not None:
            self._grid_points2 = grid_points2
            self._weights2 = weights2
        elif not self._is_nosym:
            symmetry = Symmetry(self._primitive, symprec=self._symprec)
            qpoint = self._grid_address[grid_point] / np.double(self._mesh)
            mapping, grid_points = spg.get_stabilized_reciprocal_mesh(self._mesh,
                                                                      symmetry.get_pointgroup_operations(),
                                                                      is_shift=np.zeros(3, dtype='intc'),
                                                                      is_time_reversal=True,
                                                                      qpoints=[qpoint])

            grid_points2 = np.unique(mapping)
            # weight = np.zeros_like(grid_points2)
            # bz_grid_points2 = np.zeros_like(grid_points2)
            # for g, grid in enumerate(grid_points2):
            #     weight[g] = len(np.where(grid == mapping)[0])
            #     bz_grid_points2[g] = np.where(grid == self._bz_to_pp_map)[0][0]
            # self._grid_points2 = bz_grid_points2
            weight_temp = np.bincount(mapping)
            weight = weight_temp[np.nonzero(weight_temp)]
            self._grid_points2 = grid_points2
            self._weights2 = weight
        else:
            self._grid_points2 = np.arange(np.prod(self._mesh))
            self._weights2 = np.ones_like(self._grid_points2, dtype="intc")

    def set_frequencies(self,
                       frequencies=None,
                       eigenvectors=None,
                       phonons_done=None,
                       degeneracies=None,
                       occupations=None):
        num_band = self._primitive.get_number_of_atoms() * 3
        if self._band_indices is None:
            self._band_indices = np.arange(num_band, dtype='intc')
        else:
            self._band_indices = np.array(self._band_indices, dtype='intc')

        if frequencies is None:
            self._is_extra_frequency = False
            self._grid_points2 = np.arange(np.prod(self._mesh), dtype='intc')

            if self._phonon_done is None:
                self._allocate_phonon()
        else:
            self._is_extra_frequency = True
            self.set_phonons(frequencies, eigenvectors, phonons_done, degeneracies=degeneracies, occupations=occupations)

    def set_temperature(self, temperature):
        self._temp = temperature

    def get_temperature(self):
        return self._temp

    def set_sigma(self, sigma):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = float(sigma)

    def run(self, lang="C"):
        if lang=="C":
            self._run_c()
        else:
            self._run_py()

    def get_mass_variances(self):
        return self._mass_variances

    def get_gamma(self):
        if self._gamma is not None:
            return self._gamma
        else:
            n0 = self._occupations[self._grid_point]
            f0 = self._frequencies[self._grid_point]
            rnn1 = np.where(f0>self._cutoff_frequency, 1 / n0 / (n0 + 1), 0)
            return self._collision_out * rnn1 / 2.

    def get_grid_address(self):
        return self._grid_address

    def get_phonons(self):
        return self._frequencies, self._eigenvectors, self._phonon_done
    
    def set_phonons(self, frequencies, eigenvectors, phonon_done, degeneracies=None, occupations=None, dm=None):
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._phonon_done = phonon_done
        if occupations is not None:
            self._occupations = occupations
        if degeneracies is not None:
            self._degeneracies = degeneracies
        if dm is not None:
            self._dm = dm

    def set_dynamical_matrix(self,
                             fc2=None,
                             supercell=None,
                             primitive=None,
                             nac_params=None,
                             frequency_scale_factor=None,
                             decimals=None,
                             dynamical_matrix=None):
        if dynamical_matrix is not None:
            self._dm = dynamical_matrix
        else:
            self._primitive = primitive
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
            self._nac_q_direction = np.array(nac_q_direction, dtype='double')

    def _run_c(self):
        if not self._is_extra_frequency:
            self._set_phonon_c(self._grid_points2)
        if not (self._occupation_done[self._grid_points2]).all():
            self.set_occupation(self._grid_points2)
        import anharmonic._phono3py as phono3c
        self._collision_out = np.zeros(len(self._band_indices), dtype='double')
        nband = self._frequencies.shape[-1]
        self._collision_in = np.zeros((len(self._grid_points2), len(self._band_indices), nband), dtype='double')
        weights = self._weights2
        if self._sigma is None:
            self._set_integration_weights()
            phono3c.thm_isotope_strength(self._collision_in,
                                         self._grid_point,
                                         self._grid_points2.astype('intc').copy(),
                                         self._mass_variances.astype('double').copy(),
                                         self._frequencies.astype('double').copy(),
                                         self._eigenvectors.astype("complex128").copy(),
                                         self._band_indices.astype("intc").copy(),
                                         self._occupations.astype("double").copy(),
                                         np.double(self._integration_weights).copy(),
                                         self._cutoff_frequency)

            # self._collision_out = np.dot(weights, self._collision_in.sum(axis=-1))
        else:
            phono3c.isotope_strength(self._collision_in,
                                     self._grid_point,
                                     self._grid_points2.astype('intc').copy(),
                                     self._mass_variances.astype('double').copy(),
                                     self._frequencies.astype("double").copy(),
                                     self._eigenvectors.astype("complex128").copy(),
                                     self._band_indices.astype("intc").copy(),
                                     self._occupations.astype("double").copy(),
                                     self._sigma,
                                     self._cutoff_frequency)
        # considering degeneracy
        # for ubi in np.unique(self._degeneracies[self._grid_point]):
        #     deg_index = np.where(self._degeneracies[self._grid_point] == ubi)[0]
        #     ave = np.average(self._collision_in[:, deg_index], axis=1)
        #     for bi0 in deg_index:
        #         self._collision_in[:, bi0] = ave
        #     for gp2 in self._grid_points2:
        #         for ubj in np.unique(self._degeneracies[gp2]):
        #             deg_index2 = np.where(self._degeneracies[gp2] == ubj)[0]
        #             ave2 = np.average(self._collision_in[:, :, deg_index2], axis=2)
        #             for bi1 in deg_index2:
        #                 self._collision_in[:, :, bi1] = ave2
        triplets_tmp = np.array([[self._grid_point, gp2, 0] for gp2 in self._grid_points2], dtype='intc')
        phono3c.collision_degeneracy(self._collision_in,
                                          self._degeneracies.astype('intc').copy(),
                                          triplets_tmp,
                                          False)
        self._collision_in *= np.pi / 2 / np.prod(self._mesh) # unit in THz
        self._collision_out = np.dot(weights, self._collision_in.sum(axis=-1))
        # conversion: (2 * pi) ** 2 / (2 * pi) / (2 * pi)
        # The last 2pi comes from the conversion from Rad to Hz,
        # which is already defined to hold a 2pi for ph-ph scattering

    def set_integration_weights(self, integration_weights):
        self._integration_weights = integration_weights


    def _set_integration_weights(self):
        primitive_lattice = np.linalg.inv(self._primitive.get_cell())
        thm = TetrahedronMethod(primitive_lattice, mesh=self._mesh)
        num_grid_points = len(self._grid_points2)
        num_band = self._primitive.get_number_of_atoms() * 3
        self._integration_weights = np.zeros(
            (num_grid_points, len(self._band_indices), num_band), dtype='double')
        self._set_integration_weights_c(thm)

    def _set_integration_weights_c(self, thm, infinitesmal = 1e-10):
        # infinitesmal is to prevent divergence of the results
        import anharmonic._phono3py as phono3c
        unique_vertices = thm.get_unique_tetrahedra_vertices()
        neighboring_grid_points = np.zeros(
            len(unique_vertices) * len(self._grid_points2), dtype='intc')
        phono3c.neighboring_grid_points(
            neighboring_grid_points,
            self._grid_points2.astype("intc").copy(),
            unique_vertices.astype("intc").copy(),
            self._mesh.astype("intc").copy(),
            self._grid_address.astype("intc").copy(),
            self._bz_map.astype("intc").copy())
        self._set_phonon_c(np.unique(neighboring_grid_points))
        freq_points = np.array(
            self._frequencies[self._grid_point, self._band_indices] + infinitesmal,
            dtype='double', order='C')
        phono3c.integration_weights(
            self._integration_weights,
            freq_points.astype("double").copy(),
            thm.get_tetrahedra(),
            self._mesh.astype("intc").copy(),
            self._grid_points2.astype("intc").copy(),
            self._frequencies,
            self._grid_address,
            self._bz_map)
        
    def _set_integration_weights_py(self, thm, infinitesmal = 1e-10):
        grid_order = [1, self._mesh[0], self._mesh[0] * self._mesh[1]]
        for i, gp in enumerate(self._grid_points2):
            address = thm.get_tetrahedra().reshape(-1,3) + self._grid_address[gp]
            for neighbor in np.unique(np.dot(address % self._mesh, grid_order)):
                self._set_phonon_py(neighbor)
            tfreqs = get_tetrahedra_frequencies(
                gp,
                self._mesh,
                [1, self._mesh[0], self._mesh[0] * self._mesh[1]],
                self._grid_address,
                thm.get_tetrahedra(),
                np.arange(np.prod(self._mesh)),
                self._frequencies)
            
            for bi, frequencies in enumerate(tfreqs):
                thm.set_segments_omegas(frequencies)
                thm.run(self._frequencies[self._grid_point, self._band_indices] + infinitesmal)
                iw = thm.get_integration_weight()
                self._integration_weights[i, :, bi] = iw

    def _run_py(self):
        for gp in self._grid_points2:
            self._set_phonon_py(gp)
        self.set_occupation(self._grid_points2)
        mass_v = np.array([[m] * 3 for m in self._mass_variances],
                          dtype='double').flatten()
        self._gamma = np.zeros(len(self._band_indices), dtype="double")
        for bi in self._band_indices:
            vec0 = self._eigenvectors[self._grid_point][:, bi].conj()
            f0 = self._frequencies[self._grid_point][bi]
            ti_sum = 0.0
            for i, gp in enumerate(self._grid_points2):
                for j, (f, vec) in enumerate(
                        zip(self._frequencies[gp], self._eigenvectors[gp].T)):
                    if f < self._cutoff_frequency:
                        continue
                    ti_sum_band = np.sum(np.abs(vec * vec0) ** 2 * mass_v)
                    # ti_sum += ti_sum_band * gaussian(f0 - f, self._sigma)
                    ti_sum += f * ti_sum_band * gaussian(f0 - f, self._sigma) * (self._occupations[gp,:,j] +1)
            # t_inv.append(np.pi / 2 / np.prod(self._mesh) * f0 ** 2 * ti_sum)
            occu0=self._occupations[self._grid_point, :, bi]
            self._gamma[:,bi] = (np.pi / 2 / np.prod(self._mesh) * f0  * ti_sum/(occu0+1)) / 2 * (2 * np.pi) / (2 * np.pi)
        # The first pi/2 is the coefficient, the second 2pi comes from the unit of omega ( not f) while the third comes
        #from the definition of gamma, which is already defined to hold a 2pi for ph-ph scattering

    def set_occupation(self, grid_points=None, occupations=None,occupations_done=None):
        num_band = self._primitive.get_number_of_atoms() * 3
        if occupations is not None:
            self._occupations = occupations
            self._occupation_done = occupations_done
        elif grid_points is not None:
            for gp in grid_points:
                if self._occupation_done[gp]:
                    continue
                assert self._phonon_done[gp] == 1
                for b in np.arange(num_band):
                    if self._frequencies[gp, b] < self._cutoff_frequency:
                        continue
                    self._occupations[gp,b] = occupation(self._frequencies[gp,b], self._temp)
                self._occupation_done[gp] = 1


    def _set_phonon_c(self, grid_points):
        set_phonon_c(self._dm,
                     self._frequencies,
                     self._eigenvectors,
                     self._degeneracies,
                     self._phonon_done,
                     grid_points,
                     self._grid_address,
                     self._mesh,
                     self._frequency_factor_to_THz,
                     self._nac_q_direction,
                     self._lapack_zheev_uplo)

    def _set_phonon_py(self, grid_point):
        set_phonon_py(grid_point,
                      self._phonon_done,
                      self._frequencies,
                      self._eigenvectors,
                      self._degeneracies,
                      self._grid_address,
                      self._mesh,
                      self._dm,
                      self._frequency_factor_to_THz,                  
                      self._lapack_zheev_uplo)

    def _allocate_phonon(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_grid = len(self._grid_address)
        self._phonon_done = np.zeros(num_grid, dtype='byte')
        self._frequencies = np.zeros((num_grid, num_band), dtype='double')
        self._eigenvectors = np.zeros((num_grid, num_band, num_band),
                                      dtype='complex128')
        self._occupation_done = np.zeros(num_grid, dtype='byte')
        self._occupations = np.zeros((num_grid, num_band), dtype="double")
        self._degeneracies = (np.zeros_like(self._frequencies) + np.arange(num_band)).astype("intc")

        

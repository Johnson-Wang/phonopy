__author__ = 'xinjiang'
import numpy as np
import sys
from anharmonic.phonon3.imag_self_energy import gaussian
from anharmonic.phonon3.imag_self_energy import occupation
from anharmonic.file_IO import  read_collision_all_from_hdf5, write_collision_to_hdf5_all,\
    read_collision_at_grid_from_hdf5, write_collision_to_hdf5_at_grid
from phonopy.units import THz, Hbar, EV, AMU, Angstrom
from anharmonic.phonon3.triplets import get_kgp_index_at_grid
from triplets import get_triplets_integration_weights
from anharmonic.other.isotope import CollisionIso # isotope scattering
from phonopy.phonon.group_velocity import get_group_velocity
class Collision():
    def __init__(self,
                 interaction,
                 sigmas=np.array([0.1]),
                 temperatures=None,
                 lang="C",
                 is_tetrahedron_method=False,
                 mass_variances=None,
                 length=None,
                 gv_delta_q = None,
                 is_adaptive_sigma=False,
                 write=False,
                 read=False,
                 cutoff_frequency = 1e-4): #unit: THz
        self._pp = interaction
        self._sigmas = sigmas
        self._temperatures = temperatures
        self._mesh = self._pp.get_mesh_numbers()
        self._is_nosym = self._pp._is_nosym
        if interaction._is_nosym:
            self._point_operations = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype="intc")
        else:
            self._point_operations = self._pp.get_point_group_operations()
        self._kpoint_operations = self._pp._kpoint_operations
        self._is_dispersed=self._pp.get_is_dispersed()
        self._cutoff_frequency = cutoff_frequency
        self._grid_point = None
        self._grid_point_triplets = None
        self._triplet_weights = None
        self._is_thm = is_tetrahedron_method
        self._lang = lang
        self._frequencies_all = None
        self._eigenvectors_all = None
        self._band_indices = None
        self._fc3_normal_squared = None
        self._occupations_all = None
        self._sigma =None
        self._temperature = None
        self._isigma = None
        self._itemp = None

        self._write_col=write
        self._read_col=read
        self._is_adaptive_sigma =is_adaptive_sigma
        self._is_on_iteration = False
        self._asigma = None
        self._gamma_all=None
        self._cutoff_frequency = cutoff_frequency
        self._irr_mapping_indices = None
        self._collision_in = None
        self._collision_out = None
        self._collision_in_all = None
        self._is_read_error = None
        self._unit_conversion =  np.pi / 4 * ((Hbar * EV) ** 3 # Hbar * EV : hbar
                                 * EV ** 2 / Angstrom ** 6
                                 / (2 * np.pi * THz) ** 3
                                 / AMU ** 3
                                 / (Hbar * EV) ** 2
                                 / (2 * np.pi * THz) ** 2
                                 / np.prod(self._mesh))
        self._init_grids_properties()
        self._length = length
        self._gv_delta_q = gv_delta_q
        if mass_variances is not None:
            self._collision_iso = \
                CollisionIso(self._mesh,
                             self._pp._primitive,
                             mass_variances,
                             self._band_indices,
                             self._sigma)
            self._collision_iso.set_dynamical_matrix(dynamical_matrix=self._pp.get_dynamical_matrix())
        else:
            self._collision_iso = None

    def _init_grids_properties(self):
        num_grids_all, num_band = self._pp._frequencies.shape
        num_temp = len(self._temperatures)
        num_sigma = len(self._sigmas)
        self._occupations_all = np.zeros((num_grids_all, num_temp, num_band), dtype="double")
        self._collision_out_all = np.zeros((num_sigma, num_grids_all, num_temp, num_band), dtype="double")
        self._gamma_all = np.zeros_like(self._collision_out_all)
        self._n_done = np.zeros((num_grids_all, num_temp), dtype="bool")


    def set_grid(self, grid_point):
        if grid_point is None:
            self._grid_point = None
        else:
            self._pp.set_grid_point(grid_point)
            (self._grid_point_triplets,
             self._triplet_weights) = self._pp.get_triplets_at_q()

            self._grid_point = self._grid_point_triplets[0, 0]
            self._grid_address = self._pp.get_grid_address()
            self._qpoint = self._grid_address[self._grid_point] / np.double(self._mesh)
            if  self._pp._unique_triplets is not None:
                self._inv_rot_sum = self._pp.get_2inv_rot_sum_at_grid()
                self._kpg_at_q_index = self._pp.get_kpg_at_qs_index()
            else:
                kpg_index = get_kgp_index_at_grid(self._grid_address[grid_point], self._mesh, self._kpoint_operations)
                self._inv_rot_sum = self._kpoint_operations[kpg_index].sum(axis=0)
                self._kpg_at_q_index = kpg_index

            self._fc3_normal_squared = None
            if self._collision_iso is not None:
                grid_points2 = self.get_grid_point_triplets()[:,1]
                weights2 = self.get_triplets_weights()
                self._collision_iso.set_grid_point(grid_point, grid_points2=grid_points2, weights2=weights2)



    def set_pp_grid_points_all(self, grid_points):
        self._pp.set_grid_points(grid_points)

    def set_grids(self, grid_points):
        self._pp._grid_points = grid_points
        if not self._is_dispersed:
            if self._pp._unique_triplets == None:
                self._pp.set_grid_points(grid_points)
            bi = self._pp._band_indices
            nband = 3 * self._pp._primitive.get_number_of_atoms()
            luniq = len(self._pp._unique_triplets)
            try:
                self._collision_in_all = np.zeros((luniq,3, len(bi), nband), dtype="double")
            except MemoryError:
                print "A memory error occurs in allocating the whole collision array"
                print "--disperse is recommended as a possible solution"
                sys.exit(1)
            self._collision_done = np.zeros(luniq, dtype="bool")



    def get_occupation(self):
        return self._occupations_all

    def get_grid_point_triplets(self):
        return self._grid_point_triplets

    def get_triplets_weights(self):
        return self._triplet_weights

    def get_collision_out(self):
        if self._cutoff_frequency is None:
            return self._collision_out
        else: # Averaging imag-self-energies by degenerate bands
            deg_sets = self._degeneracy_all[self._grid_point]
            cout = np.zeros_like(self._collision_out)
            for i, dset in enumerate(deg_sets):
                cout[i] = np.average(self._collision_out[np.where(deg_sets == dset)])
            return cout

    def get_collision_in(self):
        return self._collision_in

    def read_collision_all(self, log_level=0, is_adaptive_sigma=None):
        if is_adaptive_sigma == None:
            is_adaptive_sigma = self._is_adaptive_sigma
        is_error = read_collision_all_from_hdf5(self._collision_in_all,
                                                      self._mesh,
                                                      self._sigma,
                                                      self._temperature,
                                                      is_adaptive_sigma=is_adaptive_sigma,
                                                      log_level=log_level,
                                                      is_nosym=self._pp._is_nosym)

        if is_error:
            self.set_write_collision(True)
        else:
            if self._is_adaptive_sigma and (not self._is_on_iteration):
                self._collision_done[:] = False
            else:
                self._collision_done[:] = True

    def read_collision_at_grid(self, grid, log_level=0, is_adaptive_sigma=None):
        if is_adaptive_sigma == None:
            is_adaptive_sigma = self._is_adaptive_sigma
        is_error = read_collision_at_grid_from_hdf5(self._collision_in,
                                                    self._mesh,
                                                    grid,
                                                    self._sigma,
                                                    self._temperature,
                                                    is_adaptive_sigma=is_adaptive_sigma,
                                                    log_level=log_level,
                                                    is_nosym=self._pp._is_nosym)
        self._is_read_error = is_error
        if self._is_read_error:
            self.set_read_collision(False)
            self.set_write_collision(True)

    def write_collision_at_grid(self, grid, log_level=0, is_adaptive_sigma=None):
        if is_adaptive_sigma == None:
            is_adaptive_sigma = self._is_adaptive_sigma
        write_collision_to_hdf5_at_grid(self._collision_in,
                                         self._mesh,
                                         grid,
                                         self._sigma,
                                         self._temperature,
                                         is_adaptive_sigma=is_adaptive_sigma,
                                         log_level=log_level,
                                         is_nosym=self._pp._is_nosym)
        if self._is_read_error: # Condition: read argument is invoked
            self.set_read_collision(True)
            self.set_write_collision(False)

    def write_collision_all(self, log_level=0, is_adaptive_sigma=None):
        if is_adaptive_sigma == None:
            is_adaptive_sigma = self._is_adaptive_sigma
        if self._write_col:
            write_collision_to_hdf5_all(self._collision_in_all,
                                        self._mesh,
                                        self._sigma,
                                        self._temperature,
                                        is_adaptive_sigma = is_adaptive_sigma,
                                        log_level=log_level,
                                        is_nosym=self._pp._is_nosym)

    def set_temperature(self, temperature):
        for i, t in enumerate(self._temperatures):
            if temperature == t:
                self._itemp = i
                break
        self._temperature=temperature

    def set_is_adaptive_sigma(self, is_adaptive_sigma=False):
        self._is_adaptive_sigma = is_adaptive_sigma

    def set_is_on_iteration(self, is_on_iteration=True):
        self._is_on_iteration = is_on_iteration


    def set_sigma(self, sigma):
        for i, s in enumerate(self._sigmas):
            if s == sigma:
                self._isigma = i
                break
        self._sigma = sigma
        if self._sigma is None:
            self._is_thm = True

    def set_asigma(self):
        if self._is_adaptive_sigma:
            if (self._read_col or self._write_col) and (not self._is_dispersed): # collision_all case
                triplets = self._triplets_reduced
            elif not self._read_col: #
                triplets = self._grid_point_triplets
            else:
                return
            triplet_gammas = self._gamma_all[self._isigma, triplets, self._itemp]
            gt = (triplet_gammas * 2 * np.pi) ** 2 / (2 * np.log(2)) # 2pi comes from the definition of gamma
            self._asigma = np.sqrt(gt[:,0, :, np.newaxis, np.newaxis] +
                                      gt[:,1, np.newaxis, :, np.newaxis] +
                                      gt[:,2, np.newaxis, np.newaxis, :])
        elif self._read_col:
            nband = self._pp._primitive.get_number_of_atoms() * 3
            self._asigma = np.zeros((0,nband, nband, nband), dtype="double")


    def set_write_collision(self, is_write_col):
        self._write_col = is_write_col

    def set_read_collision(self, is_read_col):
        self._read_col = is_read_col

    def get_write_collision(self):
        return self._write_col

    def get_read_collision(self):
        return self._read_col

    def get_is_dispersed(self):
        return self._is_dispersed

    def get_collision_out_all(self):
        return self._collision_out_all

    def run_interaction_at_grid_point(self):
        if self._read_col and ((not self._is_adaptive_sigma) or self._is_on_iteration):
            nband = self._pp._primitive.get_number_of_atoms() * 3
            self._pp.set_phonons(lang=self._lang)
            self._fc3_normal_squared = np.zeros((0,nband, nband, nband), dtype="double")
        else:
            self._pp.run(lang=self._lang)
            self._fc3_normal_squared = self._pp.get_interaction_strength()

        (self._frequencies_all,
        self._eigenvectors_all) = self._pp.get_phonons()[:2]
        self._degeneracy_all = self._pp.get_degeneracy()
        self._band_indices = self._pp.get_band_indices()
        if self._collision_iso:
            self._collision_iso.set_frequencies(frequencies=self._frequencies_all,
                                                eigenvectors=self._eigenvectors_all,
                                                phonons_done=self._pp._phonon_done,
                                                degeneracies=self._degeneracy_all)


    def reduce_triplets(self):
        num_triplets = len(self._pp.get_triplets_at_q()[0])
        num_band0 = self._fc3_normal_squared.shape[1]
        num_band = self._fc3_normal_squared.shape[2]
        self._collision_out = None
        self._collision_in = np.zeros((num_triplets, num_band0, num_band), dtype="double")


        if not self._is_dispersed:
            self._triplets_mapping = self._pp.get_triplets_mapping_at_grid()
            self.extract_undone_triplets()
            self._collision_in_reduced = np.zeros((len(self._undone_uniq_index),
                                               3,
                                               num_band0,
                                               num_band), dtype="double")
            if self._pp.get_interaction_strength() is not None:
                self._fc3_normal_squared_reduced= self._pp.get_interaction_strength()[self._undone_triplet_index]
            else:
                self._fc3_normal_squared_reduced = np.zeros((len(self._undone_triplet_index),
                                                             num_band0,
                                                             num_band,
                                                             num_band), dtype="double")

            self._triplets_reduced = self._pp.get_triplets_at_q()[0][self._undone_triplet_index]
            assert (self._pp._phonon_done[self._triplets_reduced] == True).all()
            self._frequencies_reduced = self._frequencies_all[self._triplets_reduced]
            self._degeneracy_reduced = self._degeneracy_all[self._triplets_reduced]
            # self.set_grid_points_occupation()
            self._occu_reduced = self._occupations_all[self._triplets_reduced, self._itemp]

    def set_grid_points_occupation(self, grid_points=None):
        if grid_points is None:
            grid_points = self._grid_point_triplets
        uniq_grids = np.unique(grid_points)
        for g in uniq_grids:
            if not self._n_done[g, self._itemp]:
                n = occupation(self._frequencies_all[g], self._temperature)
                self._occupations_all[g, self._itemp] = n
                self._n_done[g, self._itemp] = True
        if self._collision_iso is not None:
            self._collision_iso.set_occupation(occupations=self._occupations_all[:,self._itemp],
                                               occupations_done=self._n_done[:,self._itemp])

    def extract_undone_triplets(self):
        undone_num = []
        index_reduced = []
        for i,m in enumerate(self._triplets_mapping):
            if m in undone_num:
                continue
            if not self._collision_done[m]:
                undone_num.append(m)
                index_reduced.append(i)
        self._undone_uniq_index = undone_num
        self._undone_triplet_index = index_reduced

    def broadcast_collision_out(self):
        grid_map = self._pp.get_grid_mapping()
        bz_to_pp_map = self._pp.get_bz_to_pp_map()
        bz_to_irred_map = grid_map[bz_to_pp_map]
        equiv_pos = np.where(bz_to_irred_map == self._grid_point)
        cm_out = self.get_collision_out()
        self._collision_out_all[self._isigma, equiv_pos, self._itemp] = cm_out
        n = self.get_occupation()[self._grid_point, self._itemp]
        nn1 = n * (n + 1)
        is_pass = self._frequencies_all[self._grid_point] < self._cutoff_frequency
        self._gamma_all[self._isigma, equiv_pos, self._itemp] = \
            cm_out * np.where(is_pass, 0, 1 / nn1) / 2.

    def run(self):
        self.set_integration_weights()
        if self._is_dispersed and self._read_col:
            self.read_collision_at_grid(self._grid_point)
        if self._lang=="C":
            self.run_c()
        else:
            self.run_py()
        if self._is_dispersed and self._write_col:
            self.write_collision_at_grid(self._grid_point)
        self._collision_in *= self._unit_conversion # unit in THz
        summation = np.sum(self._collision_in, axis=-1)
        self._collision_out = np.dot(self._triplet_weights, summation) / 2.0
        if self._collision_iso is not None:
            self._collision_iso.run()
            self._collision_out += self._collision_iso._collision_out
            self._collision_in += self._collision_iso._collision_in

        if self._length is not None:
            bnd_collision_unit = 100. / 1e-6 / THz / (2 * np.pi) # unit in THz, unit of length is micron
            dm = self._pp.get_dynamical_matrix()
            gv = get_group_velocity(self._qpoint,
                                    dm,
                                    symmetry=self._pp._symmetry,
                                    q_length=self._gv_delta_q,
                                    frequency_factor_to_THz=self._pp.get_frequency_factor_to_THz())
            gv = np.sqrt(np.sum(gv ** 2, axis=-1))
            n = self._occupations_all[self._grid_point, self._itemp]
            cout_bnd = gv / self._length * n * (n + 1)
            self._collision_out += cout_bnd * bnd_collision_unit
        self.broadcast_collision_out()

    def run_py(self):
        for i, triplet in enumerate(self._grid_point_triplets):
            occu = occupation(self._frequencies_all[triplet], self._temperature)
            freq = self._frequencies_all[triplet]
            for (j,k,l) in np.ndindex(self._fc3_normal_squared.shape[1:]):
                if self._asigma:
                    sigma=self._asigma[i,j,k,l]
                else:
                    sigma = self._sigma
                f0 = freq[0, j]; f1 = freq[1, k]; f2 = freq[2, l]
                if (f0<self._cutoff_frequency or f1 < self._cutoff_frequency or f2 < self._cutoff_frequency):
                    continue
                n0 = occu[0, j]; n1 = occu[1, k];  n2 = occu[2, l]
                delta1 = gaussian(f0+f1-f2, sigma)
                delta2 = gaussian(f0+f2-f1, sigma)
                delta3 = gaussian(f0-f1-f2, sigma)
                self._collision_in[i,j,k] += self._fc3_normal_squared[i,j,k,l] * ((n0 + 1) * n1 * n2 * delta3 +
                                                                                      (n1 + 1) * n0 * n2 * delta2 +
                                                                                      (n2 + 1) * n0 * n1 * delta1)

    def set_integration_weights(self):
        f_points = self._frequencies_all[self._grid_point][self._band_indices]
        if self._asigma is not None:
            sigma_object = self._asigma
        else:
            sigma_object = self._sigma
        if (self._read_col or self._write_col) and (not self._is_dispersed): # case for reading  collision all
            self._g = get_triplets_integration_weights(
                    self._pp,
                    np.array(f_points, dtype='double'),
                    sigma_object,
                    triplets = self._triplets_reduced)
        elif not self._read_col:
            self._g = get_triplets_integration_weights(
                self._pp,
                np.array(f_points, dtype='double'),
                sigma_object,
                triplets = self._grid_point_triplets)

    def run_c(self):
        import anharmonic._phono3py as phono3c
        freq = self._frequencies_all[self._grid_point_triplets]

        if (self._read_col or self._write_col) and (not self._is_dispersed):
            phono3c.collision_all_permute(self._collision_in_reduced,
                                                self._fc3_normal_squared_reduced.copy(),
                                                self._occu_reduced.copy(),
                                                self._frequencies_reduced.copy(),
                                                self._g.copy(),
                                                self._cutoff_frequency)
            phono3c.collision_degeneracy(self._collision_in_reduced,
                                               np.intc(self._degeneracy_reduced).copy())
            self._collision_in_all[self._undone_uniq_index] = self._collision_in_reduced[:]
            self._collision_done[self._undone_uniq_index] = True

            phono3c.collision_from_reduced(self._collision_in,
                                                 self._collision_in_all,
                                                 self._pp.get_triplets_mapping_at_grid().astype("intc"),
                                                 self._pp.get_triplets_sequence_at_grid().astype("byte"))
        elif not self._read_col:
            phono3c.collision(self._collision_in,
                              self._fc3_normal_squared.copy(),
                              freq.copy(),
                              self._g.copy(),
                              self._temperature,
                              self._cutoff_frequency)




    def calculate_collision(self,grid_point, sigma=None, temperature=None):
        "On behalf of the limited memory, each time only one temperature and one sigma is calculated"
        if sigma is not None:
            self.set_sigma(sigma)
        if temperature is not None:
            self.set_temperature(temperature)
        self.set_grid(grid_point)
        self.run_interaction_at_grid_point()
        self.set_grid_points_occupation()
        self.reduce_triplets()
        self.set_asigma()
        self.run()

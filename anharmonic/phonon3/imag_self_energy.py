import numpy as np
from phonopy.units import THzToEv, Kb, Hbar, EV, Angstrom, THz, AMU
from phonopy.phonon.group_velocity import degenerate_sets
from triplets import get_triplets_integration_weights

def gaussian(x, sigma):
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-x**2 / 2 / sigma**2)

def occupation(x, t):
    return 1.0 / (np.exp(THzToEv * x / (Kb * t)) - 1) 
    

class ImagSelfEnergy:
    def __init__(self,
                 interaction,
                 is_nu=False,
                 is_thm=False,
                 grid_point=None,
                 fpoints=None,
                 temperature=None,
                 sigma=None,
                 cutoff_lifetime=1e-4, # inseconds
                 lang='C'):
        self._interaction = interaction
        self._cutoff_delta = interaction._cutoff_delta
        self._is_nu=is_nu
        self._is_thm = is_thm
        self.set_temperature(temperature)
        self.set_fpoints(fpoints)
        self.set_grid_point(grid_point=grid_point)
        self.set_sigma(sigma)
        self._cutoff_gamma = 0.5 / cutoff_lifetime / THz
        self._lang = lang
        self._asigma=None # adaptive sigma
        self._imag_self_energy = None
        self._imag_self_energy_N = None
        self._imag_self_energy_U = None
        self._fc3_normal_squared = None
        self._frequencies = None
        self._grid_point_triplets = None
        self._triplet_weights = None
        self._band_indices = None
        self._unit_conversion = None
        self._cutoff_frequency = interaction.get_cutoff_frequency()
        self._cutoff_hfrequency = interaction.get_cutoff_hfrequency()

    def set_pp_grid_points_all(self, grid_points):
        self._interaction.set_grid_points(grid_points)


    def set_nu_properties(self):
        self._imag_self_energy_N=np.zeros_like(self._imag_self_energy)
        self._imag_self_energy_U=np.zeros_like(self._imag_self_energy)
        self._grid_point_triplets_N=self._grid_point_triplets[self._triplets_index_N]
        self._grid_point_triplets_U=self._grid_point_triplets[self._triplets_index_U]
        self._fc3_normal_squared_N=self._fc3_normal_squared[self._triplets_index_N]
        self._fc3_normal_squared_U=self._fc3_normal_squared[self._triplets_index_U]
        self._triplet_weights_N=self._triplet_weights[self._triplets_index_N]
        self._triplet_weights_U=self._triplet_weights[self._triplets_index_U]

    def run(self):
        if self._fc3_normal_squared is None:        
            self.run_interaction()
        self.set_integration_weights()
        num_band0 = self._fc3_normal_squared.shape[1]

        if not self._is_thm:
            if self._fpoints is None:
                self._imag_self_energy = np.zeros(num_band0, dtype='double')
                if self._is_nu:
                    self.set_nu_properties()
                if self._lang == 'C':
                    self._run_c_with_band_indices()
                else:
                    self._run_py_with_band_indices()
            else:
                self._imag_self_energy = np.zeros((len(self._fpoints), num_band0),
                                                  dtype='double')
                if self._lang == 'C':
                    self._run_c_with_fpoints()
                else:
                    self._run_py_with_fpoints()
        else:
            if self._fpoints is None:
                self._imag_self_energy = np.zeros(num_band0, dtype='double')
                self._run_thm_with_band_indices()
            else:
                self._imag_self_energy = np.zeros(
                    (len(self._fpoints), num_band0), dtype='double')
                self._run_thm_with_frequency_points()

    def _run_thm_with_band_indices(self):
        if self._g is not None:
            if self._lang == 'C':
                self._run_thm_c_with_band_indices()
            else:
                self._run_thm_py_with_band_indices()
        else:
            if self._lang == 'C':
                self._run_c_with_band_indices()
            else:
                self._run_py_with_band_indices()

    def _run_thm_py_with_band_indices(self):
        if self._temperature > 0:
            self._ise_thm_with_band_indices()
        else:
            self._ise_thm_with_band_indices_0K()

    def _ise_thm_with_band_indices(self):
        freqs = self._frequencies[self._grid_point_triplets[:, [1, 2]]]
        freqs = np.where(freqs > self._cutoff_frequency, freqs, 1)
        n = occupation(freqs, self._temperature)
        for i, (tp, w, interaction) in enumerate(zip(self._grid_point_triplets,
                                                     self._triplet_weights,
                                                     self._interaction)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                f1 = self._frequencies[tp[1]][j]
                f2 = self._frequencies[tp[2]][k]
                if (f1 > self._cutoff_frequency and
                    f2 > self._cutoff_frequency):
                    n2 = n[i, 0, j]
                    n3 = n[i, 1, k]
                    g1 = self._g[0, i, :, j, k]
                    g2_g3 = self._g[1, i, :, j, k] # g2 - g3
                    self._imag_self_energy[:] += (
                        (n2 + n3 + 1) * g1 +
                        (n2 - n3) * (g2_g3)) * interaction[:, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _ise_thm_with_band_indices_0K(self):
        for i, (w, interaction) in enumerate(zip(self._triplet_weights,
                                                 self._interaction)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                g1 = self._g[0, i, :, j, k]
                self._imag_self_energy[:] += g1 * interaction[:, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _run_thm_with_frequency_points(self):
        if self._g is not None:
            if self._lang == 'C':
                self._run_thm_c_with_frequency_points()
            else:
                self._run_thm_py_with_frequency_points()
        else:
            if self._lang == 'C':
                self._run_c_with_fpoints()
            else:
                self._run_py_with_fpoints()

    def _run_thm_py_with_frequency_points(self):
        if self._temperature > 0:
            self._ise_thm_with_frequency_points()
        else:
            self._ise_thm_with_frequency_points_0K()

    def _ise_thm_with_frequency_points(self):
        for i, (tp, w, interaction) in enumerate(zip(self._grid_point_triplets,
                                                     self._triplet_weights,
                                                     self._interaction)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                f1 = self._frequencies[tp[1]][j]
                f2 = self._frequencies[tp[2]][k]
                if (f1 > self._cutoff_frequency and
                    f2 > self._cutoff_frequency):
                    n2 = occupation(f1, self._temperature)
                    n3 = occupation(f2, self._temperature)
                    g1 = self._g[0, i, :, j, k]
                    g2_g3 = self._g[1, i, :, j, k] # g2 - g3
                    for l in range(len(interaction)):
                        self._imag_self_energy[:, l] += (
                            (n2 + n3 + 1) * g1 +
                            (n2 - n3) * (g2_g3)) * interaction[l, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _ise_thm_with_frequency_points_0K(self):
        for i, (w, interaction) in enumerate(zip(self._triplet_weights,
                                                 self._interaction)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                g1 = self._g[0, i, :, j, k]
                for l in range(len(interaction)):
                    self._imag_self_energy[:, l] += g1 * interaction[l, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _run_thm_c_with_frequency_points(self):
        import anharmonic._phono3py as phono3c
        g = np.zeros((2,) + self._interaction.shape, dtype='double')
        ise_at_f = np.zeros(self._imag_self_energy.shape[1], dtype='double')
        for i in range(len(self._fpoints)):
            for j in range(g.shape[2]):
                g[:, :, j, :, :] = self._g[:, :, i, :, :]
            phono3c.thm_imag_self_energy(ise_at_f,
                                         self._interaction,
                                         self._grid_point_triplets,
                                         self._triplet_weights,
                                         self._frequencies,
                                         self._temperature,
                                         g,
                                         self._unit_conversion,
                                         self._cutoff_frequency)
            self._imag_self_energy[i] = ise_at_f

    def _run_thm_c_with_band_indices(self):
        import anharmonic._phono3py as phono3c
        phono3c.thm_imag_self_energy(self._imag_self_energy,
                                     self._interaction.get_interaction_strength(),
                                     self._grid_point_triplets,
                                     self._triplet_weights,
                                     self._frequencies,
                                     self._temperature,
                                     self._g,
                                     self._unit_conversion,
                                     self._cutoff_frequency)

    def set_integration_weights(self, scattering_event_class=None):
        if self._fpoints is None:
            f_points = self._frequencies[self._grid_point][self._band_indices]
        else:
            f_points = self._fpoints

        self._g = get_triplets_integration_weights(
            self._interaction,
            np.array(f_points, dtype='double'),
            self._sigma)

        if scattering_event_class == 1:
            self._g[0] = 0
        elif scattering_event_class == 2:
            self._g[1] = 0
            self._g[2] = 0

    def run_interaction(self, is_triplets_dispersed=False, log_level=0):
        self._interaction.run(lang=self._lang,
                              log_level=log_level)
        self._fc3_normal_squared = self._interaction.get_interaction_strength()
        (self._frequencies,
         self._eigenvectors) = self._interaction.get_phonons()[:2]
        self._band_indices = self._interaction.get_band_indices()
        
        mesh = self._interaction.get_mesh_numbers()
        num_grid = np.prod(mesh)

        # Unit to THz of Gamma
        self._unit_conversion = ((Hbar * EV) ** 3 / 36 / 8
                                 * EV ** 2 / Angstrom ** 6
                                 / (2 * np.pi * THz) ** 3
                                 / AMU ** 3
                                 * 18 * np.pi / (Hbar * EV) ** 2
                                 / (2 * np.pi * THz) ** 2
                                 / num_grid)

    def get_imag_self_energy(self):
        if self._cutoff_frequency is None:
            return self._imag_self_energy
        else: # Averaging imag-self-energies by degenerate bands
            imag_se = np.zeros_like(self._imag_self_energy)
            freqs = self._frequencies[self._grid_point]
            deg_sets = degenerate_sets(freqs) # such like [[0,1], [2], [3,4,5]]
            for dset in deg_sets:
                bi_set = []
                for i, bi in enumerate(self._band_indices):
                    if bi in dset:
                        bi_set.append(i)
                if len(bi_set) > 0:
                    for i in bi_set:
                        if self._fpoints is None:
                            imag_se[i] = (self._imag_self_energy[bi_set].sum() /
                                          len(bi_set))
                        else:
                            imag_se[:, i] = (
                                self._imag_self_energy[:, bi_set].sum(axis=1) /
                                len(bi_set))
            return imag_se

    def get_imag_self_energy_N(self):
        if self._cutoff_frequency is None:
            return self._imag_self_energy_N
        else: # Averaging imag-self-energies by degenerate bands
            imag_se = np.zeros_like(self._imag_self_energy_N)
            freqs = self._frequencies[self._grid_point]
            deg_sets = degenerate_sets(freqs) # such like [[0,1], [2], [3,4,5]]
            for dset in deg_sets:
                bi_set = []
                for i, bi in enumerate(self._band_indices):
                    if bi in dset:
                        bi_set.append(i)
                if len(bi_set) > 0:
                    for i in bi_set:
                        if self._fpoints is None:
                            imag_se[i] = (self._imag_self_energy_N[bi_set].sum() /
                                          len(bi_set))
                        else:
                            imag_se[:, i] = (
                                self._imag_self_energy_N[:, bi_set].sum(axis=1) /
                                len(bi_set))
            return imag_se

    def get_imag_self_energy_U(self):
        if self._cutoff_frequency is None:
            return self._imag_self_energy_U
        else: # Averaging imag-self-energies by degenerate bands
            imag_se = np.zeros_like(self._imag_self_energy_U)
            freqs = self._frequencies[self._grid_point]
            deg_sets = degenerate_sets(freqs) # such like [[0,1], [2], [3,4,5]]
            for dset in deg_sets:
                bi_set = []
                for i, bi in enumerate(self._band_indices):
                    if bi in dset:
                        bi_set.append(i)
                if len(bi_set) > 0:
                    for i in bi_set:
                        if self._fpoints is None:
                            imag_se[i] = (self._imag_self_energy_U[bi_set].sum() /
                                          len(bi_set))
                        else:
                            imag_se[:, i] = (
                                self._imag_self_energy_U[:, bi_set].sum(axis=1) /
                                len(bi_set))
            return imag_se
            
    def get_phonon_at_grid_point(self):
        return (self._frequencies[self._grid_point],
                self._eigenvectors[self._grid_point])

    def set_grid_point(self, grid_point=None, i=None):
        if grid_point is None:
            self._grid_point = None
        else:
            self._interaction.set_grid_point(grid_point, i)
            self._fc3_normal_squared = None
            (self._grid_point_triplets,
             self._triplet_weights) = self._interaction.get_triplets_at_q()
            if self._is_nu:
                (self._triplets_index_N, self._triplets_index_U) = self._interaction.get_triplets_at_q_nu()
            self._grid_point = self._grid_point_triplets[0, 0]

        
    def set_sigma(self, sigma):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = float(sigma)
            self._asigma = np.ones((len(self._grid_point_triplets),)+ self._fc3_normal_squared.shape[1:], dtype="float") * sigma

    def set_adaptive_sigma(self, triplet_indices, gamma):
        self._asigma=np.zeros_like(self._fc3_normal_squared)
        assert triplet_indices.max()+1 == len(gamma)
        assert len(triplet_indices) == len(self._asigma), "length of triplet indices: %d, length of asigma:%d" %(len(triplet_indices), len(self._asigma))
        gamma2 = (gamma * 2 * np.pi) ** 2 / (2 * np.log(2)) # Gamma_THz = 2 pi Gamma, Gamma = sqrt(2*ln2) sigma
        b0=self._band_indices
        gt = gamma2[triplet_indices]
        self._asigma[:] = np.sqrt(gt[:,0, :, np.newaxis, np.newaxis] + gt[:,1, np.newaxis, :, np.newaxis] + gt[:,2, np.newaxis, np.newaxis, :])
        # for i, (g0, g1, g2) in enumerate(triplet_indices):
        #     self._asigma[i] = np.sqrt((gamma2[g0,b0].reshape(-1, 1, 1) + gamma2[g1].reshape(1,-1,1) + gamma2[g2].reshape(1,1,-1)))

    def set_fpoints(self, fpoints):
        if fpoints is None:
            self._fpoints = None
        else:
            self._fpoints = np.double(fpoints)

    def set_temperature(self, temperature):
        if temperature is None:
            self._temperature = None
        else:
            self._temperature = float(temperature)
        
    def _run_c_with_band_indices(self):
        import anharmonic._phono3py as phono3c
        phono3c.imag_self_energy_at_bands(self._imag_self_energy,
                                          self._fc3_normal_squared,
                                          self._grid_point_triplets,
                                          self._triplet_weights,
                                          self._frequencies,
                                          self._band_indices,
                                          self._temperature,
                                          self._asigma,
                                          self._unit_conversion,
                                          self._cutoff_delta,
                                          self._cutoff_frequency,
                                          self._cutoff_hfrequency,
                                          self._cutoff_gamma)
        if self._is_nu:
            if len(self._triplets_index_N):
                phono3c.imag_self_energy_at_bands(self._imag_self_energy_N,
                                                  self._fc3_normal_squared_N,
                                                  self._grid_point_triplets_N,
                                                  self._triplet_weights_N,
                                                  self._frequencies,
                                                  self._band_indices,
                                                  self._temperature,
                                                  self._asigma,
                                                  self._unit_conversion,
                                                  self._cutoff_delta,
                                                  self._cutoff_frequency,
                                                  self._cutoff_hfrequency,
                                                  self._cutoff_gamma)
            if len(self._triplets_index_U):
                phono3c.imag_self_energy_at_bands(self._imag_self_energy_U,
                                                  self._fc3_normal_squared_U,
                                                  self._grid_point_triplets_U,
                                                  self._triplet_weights_U,
                                                  self._frequencies,
                                                  self._band_indices,
                                                  self._temperature,
                                                  self._asigma,
                                                  self._unit_conversion,
                                                  self._cutoff_delta,
                                                  self._cutoff_frequency,
                                                  self._cutoff_hfrequency,
                                                  self._cutoff_gamma)

    def _run_c_with_fpoints(self):
        import anharmonic._phono3py as phono3c
        for i, fpoint in enumerate(self._fpoints):
            phono3c.imag_self_energy(self._imag_self_energy[i],
                                     self._fc3_normal_squared,
                                     self._grid_point_triplets,
                                     self._triplet_weights,
                                     self._frequencies,
                                     fpoint,
                                     self._temperature,
                                     self._asigma,
                                     self._unit_conversion,
                                     self._cutoff_delta,
                                     self._cutoff_frequency,
                                     self._cutoff_gamma)

    def _run_py_with_band_indices(self):
        for i, (triplet, w, interaction) in enumerate(
            zip(self._grid_point_triplets,
                self._triplet_weights,
                self._fc3_normal_squared)):
            print "%d / %d" % (i + 1, len(self._grid_point_triplets))

            freqs = self._frequencies[triplet]
            for j, bi in enumerate(self._band_indices):
                if freqs[bi] > self._cutoff_hfrequency:
                    continue
                if self._temperature > 0:
                    self._imag_self_energy[j] += (
                        self._imag_self_energy_at_bands(
                            j, bi, freqs, interaction, w))
                else:
                    self._imag_self_energy[j] += (
                        self._imag_self_energy_at_bands_0K(
                            j, bi, freqs, interaction, w))

        self._imag_self_energy *= self._unit_conversion

    def _imag_self_energy_at_bands(self, i, bi, freqs, interaction, weight):
        sum_g = 0
        for j, k in list(np.ndindex(interaction.shape[1:])):
            sigma=self._asigma[i,j,k] #dimension of sigma is the same as the interaction
            if (freqs[1][j] > self._cutoff_frequency and
                freqs[2][k] > self._cutoff_frequency and
                sigma > self._cutoff_gamma):
                n2 = occupation(freqs[1][j], self._temperature)
                n3 = occupation(freqs[2][k], self._temperature)

                g1 = gaussian(freqs[0, bi] - freqs[1, j] - freqs[2, k],
                              sigma)
                g2 = gaussian(freqs[0, bi] + freqs[1, j] - freqs[2, k],
                              sigma)
                g3 = gaussian(freqs[0, bi] - freqs[1, j] + freqs[2, k],
                              sigma)
                sum_g += ((n2 + n3 + 1) * g1 +
                          (n2 - n3) * (g2 - g3)) * interaction[i, j, k] * weight
        return sum_g

    def _imag_self_energy_at_bands_0K(self, i, bi, freqs, interaction, weight):
        sum_g = 0
        for (j, k) in list(np.ndindex(interaction.shape[1:])):
            sigma=self._asigma[i,j,k]
            if sigma > self._cutoff_gamma:
                g1 = gaussian(freqs[0, bi] - freqs[1, j] - freqs[2, k],
                              sigma)
                sum_g += g1 * interaction[i, j, k] * weight

        return sum_g


    def _run_py_with_fpoints(self):
        for i, (triplet, w, interaction) in enumerate(
            zip(self._grid_point_triplets,
                self._triplet_weights,
                self._fc3_normal_squared)):
            print "%d / %d" % (i + 1, len(self._grid_point_triplets))

            # freqs[2, num_band]
            freqs = self._frequencies[triplet[1:]]
            if self._temperature > 0:
                self._imag_self_energy_with_fpoints(freqs, interaction, w)
            else:
                self._imag_self_energy_with_fpoints_0K(freqs, interaction, w)

        self._imag_self_energy *= self._unit_conversion

    def _imag_self_energy_with_fpoints(self, freqs, interaction, weight):
        for (i, j, k) in list(np.ndindex(interaction.shape)):
            sigma=self._asigma[i,j,k]
            if (freqs[0][j] > self._cutoff_frequency and
                freqs[1][k] > self._cutoff_frequency and
                sigma > self._cutoff_gamma):
                n2 = occupation(freqs[0][j], self._temperature)
                n3 = occupation(freqs[1][k], self._temperature)
                g1 = gaussian(self._fpoints - freqs[0][j] - freqs[1][k],
                              sigma)
                g2 = gaussian(self._fpoints + freqs[0][j] - freqs[1][k],
                              sigma)
                g3 = gaussian(self._fpoints - freqs[0][j] + freqs[1][k],
                              sigma)
                self._imag_self_energy[:, i] += (
                    (n2 + n3 + 1) * g1 +
                    (n2 - n3) * (g2 - g3)) * interaction[i, j, k] * weight

    def _imag_self_energy_with_fpoints_0K(self, freqs, interaction, weight):
        for (i, j, k) in list(np.ndindex(interaction.shape)):
            sigma=self._asigma[i,j,k]
            if sigma > self._cutoff_gamma:
                g1 = gaussian(self._fpoints - freqs[0][j] - freqs[1][k],
                              sigma)
                self._imag_self_energy[:, i] += g1 * interaction[i, j, k] * weight
        
class MatrixContribution:
    def __init__(self,
                 interaction,
                 grid_point=None,
                 fpoints=None,
                 sigma=None,
                 lang='C'):
        self._interaction = interaction
        self._cutoff_delta = interaction._cutoff_delta
        self.set_fpoints(fpoints)
        self.set_grid_point(grid_point=grid_point)
        self.set_sigma(sigma)
        self._lang = lang
        self._asigma=None # adaptive sigma
        self._matrix_contribution = None
        self._fc3_normal_squared = None
        self._frequencies = None
        self._grid_point_triplets = None
        self._triplet_weights = None
        self._band_indices = None
        self._unit_conversion = None
        self._cutoff_frequency = interaction.get_cutoff_frequency()

    def run(self):
        if self._fc3_normal_squared is None:
            self.run_interaction()

        if self._fpoints is not None:
            self._matrix_contribution = np.zeros(len(self._fpoints),dtype='double')
            self._run_py_with_fpoints()

    def run_interaction(self):

        self._interaction.run(lang=self._lang)
        self._fc3_normal_squared = self._interaction.get_interaction_strength()
        (self._frequencies,
         self._eigenvectors) = self._interaction.get_phonons()[:2]
        self._band_indices = self._interaction.get_band_indices()

        mesh = self._interaction.get_mesh_numbers()
        num_grid = np.prod(mesh)

        self._unit_conversion = ((Hbar * EV) ** 3 / 36 / 8
                                 * EV ** 2 / Angstrom ** 6
                                 / (2 * np.pi * THz) ** 3
                                 / AMU ** 3
                                 * 18 * np.pi / (Hbar * EV) ** 2
                                 / (2 * np.pi * THz) ** 2
                                 / num_grid)

    def get_imag_self_energy(self):
        return self._matrix_contribution


    def get_phonon_at_grid_point(self):
        return (self._frequencies[self._grid_point],
                self._eigenvectors[self._grid_point])

    def set_grid_point(self, grid_point=None):
        if grid_point is None:
            self._grid_point = None
        else:
            self._interaction.set_grid_point(grid_point)
            self._fc3_normal_squared = None
            (self._grid_point_triplets,
             self._triplet_weights) = self._interaction.get_triplets_at_q()
            self._grid_point = self._grid_point_triplets[0, 0]


    def set_sigma(self, sigma):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = float(sigma)


    def set_fpoints(self, fpoints):
        if fpoints is None:
            self._fpoints = None
        else:
            self._fpoints = np.double(fpoints)


    def _run_py_with_fpoints(self):
        num_band0 = self._fc3_normal_squared.shape[1]
        matrix_band  = np.zeros(num_band0, dtype="double")
        freqs = self._frequencies[self._grid_point]
        for i, (triplet, w, interaction) in enumerate(
            zip(self._grid_point_triplets,
                self._triplet_weights,
                self._fc3_normal_squared)):
            print "%d / %d" % (i + 1, len(self._grid_point_triplets))

            matrix_band += np.sum(interaction, axis=(1,2)) * w

        for i, f in enumerate(self._fpoints):
            self._matrix_contribution[i] = np.sum(gaussian(freqs-f, sigma=self._sigma) * matrix_band)

        self._matrix_contribution *= self._unit_conversion


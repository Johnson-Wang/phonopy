import numpy as np
from anharmonic.file_IO import write_decay_channels
from phonopy.units import THzToEv, Kb, VaspToTHz, Hbar, EV, Angstrom, THz, AMU
from triplets import get_triplets_integration_weights
from anharmonic.phonon3.imag_self_energy import occupation,gaussian

class DecayChannel:
    def __init__(self,
                 interaction,
                 nu=None,
                 scattering_class=None,
                 grid_point=None,
                 fpoints=None,
                 temperature=None,
                 sigma=0.1,
                 is_thm = False,
                 lang='C'):
        self._interaction = interaction
        self._is_nosym=interaction._is_nosym
        self._mesh=interaction.get_mesh_numbers()
        self._num_atom=interaction.get_primitive().get_number_of_atoms()
        (self._frequencies,
         self._eigenvectors) = interaction.get_phonons()[:2]
        self._band_indices = interaction.get_band_indices()
        self._is_thm = is_thm
        self._freq_factor_to_THz=VaspToTHz/interaction.get_frequency_factor_to_THz()
        mesh = interaction.get_mesh_numbers()
        num_grid = np.prod(mesh)
        self._nu=nu
        self._scattering_class = scattering_class

        self.set_sigma(sigma)
        self.set_temperature(temperature)
        self.set_fpoints(fpoints)
        self.set_grid_point(grid_point=grid_point)
        self._g = None
        self._lang = lang
        self._imag_self_energy = None
        self._fc3_normal_squared = None
        self._frequencies = None
        self._grid_point_triplets = None
        self._triplet_weights = None
        self._band_indices = interaction.get_band_indices()
        self._unit_conversion = None
        self._cutoff_frequency = interaction.get_cutoff_frequency()

    def set_sigma(self,sigma):
        if sigma is None:
            self._sigma=None
        else:
            self._sigma=float(sigma)

    def set_temperature(self, temperature):
        if temperature is None:
            self._temperature = None
        else:
            self._temperature = float(temperature)

    def set_fpoints(self, fpoints):
        if fpoints is None:
            self._fpoints = None
        else:
            self._fpoints = np.double(fpoints)

    def run_interaction(self, log_level=0):
        self.set_phonons(lang=self._lang)
        self.set_integration_weights(scattering_event_class=self._scattering_class,
                                     is_triplet_symmetry=self._interaction._symmetrize_fc3_q)
        if self._interaction._is_dispersed:
            uniq, index = np.unique(self._interaction._triplets_maping_at_grid, return_index=True)
            g_skip = (np.abs(self._g[:, index]).sum(axis=0) < 1e-8)
        else:
            g_skip = (np.abs(self._g).sum(axis=0) < 1e-8)
        self._interaction.run(g_skip=g_skip,
                              lang=self._lang,
                              log_level=log_level)
        self._fc3_normal_squared = self._interaction.get_interaction_strength()
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

    def set_phonons(self, lang="C"):
        self._interaction.set_phonons(lang=lang)
        (self._frequencies,
         self._eigenvectors) = self._interaction.get_phonons()[:2]

    def set_grid_point(self, grid_point=None):
        if grid_point is None:
            self._grid_point = None
        else:
            self._interaction.set_grid_point(grid_point)
            self._grid_points=self._interaction.get_grid_address()
            self._fc3_normal_squared = None
            (self._grid_point_triplets,
             self._triplet_weights) = self._interaction.get_triplets_at_q()
            self._grid_point = self._grid_point_triplets[0, 0]

    def _print_log(self, message):
        print message

    def _run_c(self):
        import anharmonic._phono3py as phono3c
        if self._g is None:
            phono3c.decay_channel(self._decay_channels,
                                  self._fc3_normal_squared,
                                  self._frequencies_at_q,
                                  self._fpoints,
                                  self._cutoff_frequency,
                                  float(self._temperature),
                                  float(self._sigma))
        else:
            phono3c.decay_channel_thm(self._decay_channels,
                                      self._fc3_normal_squared,
                                      self._frequencies_at_q,
                                      self._fpoints,
                                      self._g,
                                      self._cutoff_frequency,
                                      float(self._temperature))

    def _run_py(self):
        for i, (triplet, w, interaction) in enumerate(
            zip(self._grid_point_triplets,
                self._triplet_weights,
                self._fc3_normal_squared)):
            print "%d / %d" % (i + 1, len(self._grid_point_triplets))

            freqs = self._frequencies[triplet]
            for j, bi in enumerate(self._band_indices):
                if self._temperature > 0:
                    self._decay_channels_at_bands(
                        i, j,bi, freqs, interaction, w)
                else:
                    self._decay_channels_at_bands_0K(
                        i,j, bi, freqs, interaction, w)

    def _decay_channels_at_bands(self, i,p, bi, freqs, interaction, weight):
        for (j, k) in list(np.ndindex(interaction.shape[1:])):
            if (freqs[1][j] > self._cutoff_frequency and
                freqs[2][k] > self._cutoff_frequency):
                n2 = occupation(freqs[1][j], self._temperature)
                n3 = occupation(freqs[2][k], self._temperature)
                g1 = gaussian(freqs[0, bi] - freqs[1, j] - freqs[2, k],
                              self._sigma)
                g2 = gaussian(freqs[0, bi] + freqs[1, j] - freqs[2, k],
                              self._sigma)
                g3 = gaussian(freqs[0, bi] - freqs[1, j] + freqs[2, k],
                              self._sigma)
                self._decay_channels[i,j,k] += ((n2 + n3 + 1) * g1 +
                          (n2 - n3) * (g2 - g3)) * interaction[p, j, k]

    def _decay_channels_at_bands_0K(self, i, p, bi, freqs, interaction, weight):
        for (j, k) in list(np.ndindex(interaction.shape[1:])):
            g1 = gaussian(freqs[0, bi] - freqs[1, j] - freqs[2, k],
                          self._sigma)
            self._decay_channels[i,j,k] += g1 * interaction[p, j, k]

    def get_decay_channels(self, filename=None):
        self._print_log("---- Decay channels ----\n")
        if self._temperature==None:
            self._print_log("Temperature: 0K\n")
            self._temperature=0
        else:
            self._print_log("Temperature: %10.3fK\n" % self._temperature)
        freqs=self._frequencies[self._grid_point]
        self._fpoints = np.array([freqs[x] for x in self._band_indices])
        self._frequencies_at_q=self._frequencies[self._grid_point_triplets]
        if self._temperature==None:
            self._temperature = -1
        self._decay_channels = np.zeros((len(self._grid_point_triplets),
                                   self._num_atom*3,
                                   self._num_atom*3), dtype="double")

        if self._fc3_normal_squared is None:
            self.run_interaction()
        if self._lang =="C":
            self._run_c()
        else:
            self._run_py()
        self._decay_channels *= self._unit_conversion / len(self._fpoints)
        filename = write_decay_channels(self._decay_channels,
                                        self._fc3_normal_squared,
                                        self._frequencies_at_q,
                                        self._grid_point_triplets,
                                        self._triplet_weights,
                                        self._grid_points,
                                        self._mesh,
                                        self._band_indices,
                                        self._fpoints,
                                        self._grid_point,
                                        is_nosym=self._is_nosym,
                                        filename=filename)

        decay_channels_sum = np.array(
            [d.sum() * w for d, w in
             zip(self._decay_channels, self._triplet_weights)]).sum()

        self._print_log("FWHM: %f\n" % (decay_channels_sum * 2))
        self._print_log( "Decay channels are written into %s.\n" % filename)

    def set_integration_weights(self, scattering_event_class=None, is_triplet_symmetry=True):
        f_points = self._fpoints
        num_band = self._frequencies.shape[-1]
        if self._nu is not None:
            self._g = np.zeros(
                (3, len(self._grid_point_triplets), len(f_points), num_band, num_band),
                dtype='double')
            grid_addresses = self._interaction.get_grid_address()
            triplets_sum = grid_addresses[self._grid_point_triplets].sum(axis=1)
            is_normal_process = np.alltrue(triplets_sum == 0, axis=-1)
            if self._nu == "N":
                triplets = self._grid_point_triplets[np.where(is_normal_process)]
                g_reduced = get_triplets_integration_weights(
                    self._interaction,
                    f_points,
                    self._sigma,
                    band_indices=self._band_indices,
                    triplets=triplets,
                    is_triplet_symmetry = is_triplet_symmetry)
                self._g[:, np.where(is_normal_process)[0]] = g_reduced
            elif self._nu == "U":
                triplets = self._grid_point_triplets[np.where(is_normal_process == False)]
                g_reduced = get_triplets_integration_weights(
                    self._interaction,
                    f_points,
                    self._sigma,
                    triplets=triplets,
                    band_indices=self._band_indices,
                    is_triplet_symmetry = is_triplet_symmetry)
                self._g[:, np.where(is_normal_process == False)[0]] = g_reduced
        else:
            self._g = get_triplets_integration_weights(
                self._interaction,
                f_points,
                self._sigma,
                band_indices=self._band_indices,
                is_triplet_symmetry = is_triplet_symmetry)
        if scattering_event_class == 1:
            self._g[0] = 0
        elif scattering_event_class == 2:
            self._g[1] = 0
            self._g[2] = 0
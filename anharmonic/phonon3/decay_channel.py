import numpy as np
from anharmonic.file_IO import write_decay_channels
from phonopy.units import THzToEv, Kb, VaspToTHz, Hbar, EV, Angstrom, THz, AMU
from anharmonic.phonon3.imag_self_energy import occupation,gaussian

class DecayChannel:
    def __init__(self,
                 interaction,
                 grid_point=None,
                 fpoints=None,
                 temperature=None,
                 sigma=0.1,
                 lang='C'):
        self._interaction = interaction
        self._is_nosym=interaction._is_nosym
        self._mesh=interaction.get_mesh_numbers()
        self._num_atom=interaction.get_primitive().get_number_of_atoms()
        (self._frequencies,
         self._eigenvectors) = interaction.get_phonons()[:2]
        self._band_indices = interaction.get_band_indices()

        self._freq_factor_to_THz=VaspToTHz/interaction.get_frequency_factor_to_THz()
        mesh = interaction.get_mesh_numbers()
        num_grid = np.prod(mesh)

        self.set_sigma(sigma)
        self.set_temperature(temperature)
        self.set_fpoints(fpoints)
        self.set_grid_point(grid_point=grid_point)

        self._lang = lang
        self._imag_self_energy = None
        self._fc3_normal_squared = None
        self._frequencies = None
        self._grid_point_triplets = None
        self._triplet_weights = None
        self._band_indices = None
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

    def set_temperature(self, temperature):
        if temperature is None:
            self._temperature = None
        else:
            self._temperature = float(temperature)

    def run_interaction(self, read_amplitude=False):
        self._interaction.run_at_sigma_and_temp(read_amplitude=read_amplitude, lang=self._lang)
        self._fc3_normal_squared = self._interaction.get_interaction_strength()
        (self._frequencies,
         self._eigenvectors) = self._interaction.get_phonons()[:2]
        self._frequencies *= self._freq_factor_to_THz # unit in THz
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
        phono3c.decay_channel(self._decay_channels,
                              self._fc3_normal_squared,
                              self._frequencies_at_q,
                              self._fpoints,
                              self._cutoff_frequency,
                              float(self._temperature),
                              float(self._sigma))

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
        if self._lang =="C":
            self._run_c()
        else:
            self._run_py()
        # self._run_py()
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


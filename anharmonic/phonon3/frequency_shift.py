import numpy as np
from phonopy.units import THzToEv, Kb, VaspToTHz, Hbar, EV, Angstrom, THz, AMU
from phonopy.phonon.group_velocity import degenerate_sets
from anharmonic.phonon3.imag_self_energy import occupation

class FrequencyShift:
    def __init__(self,
                 interaction,
                 grid_point=None,
                 fpoints=None,
                 temperature=None,
                 epsilon=0.1,
                 lang='C'):
        self._interaction = interaction
        self.set_epsilon(epsilon)
        self.set_temperature(temperature)
        self.set_fpoints(fpoints)
        self.set_grid_point(grid_point=grid_point)

        self._lang = lang
        self._frequency_ = None
        self._fc3_normal_squared = None
        self._frequencies = None
        self._grid_point_triplets = None
        self._triplet_weights = None
        self._band_indices = None
        self._unit_conversion = None
        self._cutoff_frequency = interaction.get_cutoff_frequency()

        self._frequency_shifts = None

    def run(self):
        if self._fc3_normal_squared is None:        
            self.run_interaction()

        num_band0 = self._fc3_normal_squared.shape[1]
        if self._fpoints is None:
            self._frequency_shifts = np.zeros(num_band0, dtype='double')
            # if self._lang == 'C':
            #     self._run_c_with_band_indices()
            # else:
            self._run_py_with_band_indices()
        else:
            self._frequency_shifts = np.zeros((len(self._fpoints), num_band0),
                                              dtype='double')
            # if self._lang == 'C':
            #     self._run_c_with_fpoints()
            # else:
            self._run_py_with_fpoints()

    def run_interaction(self):
        self._interaction.run_at_sigma_and_temp(lang=self._lang)
        self._fc3_normal_squared = self._interaction.get_interaction_strength()
        (self._frequencies,
         self._eigenvectors) = self._interaction.get_phonons()[:2]
        (self._grid_point_triplets,
         self._triplet_weights) = self._interaction.get_triplets_at_q()
        self._band_indices = self._interaction.get_band_indices()
        
        mesh = self._interaction.get_mesh_numbers()
        num_grid = np.prod(mesh)

        # Unit to THz of Gamma
        self._unit_conversion = ((Hbar * EV) ** 3 / 36 / 8
                                 * EV ** 2 / Angstrom ** 6
                                 / (2 * np.pi * THz) ** 3
                                 / AMU ** 3
                                 * 18 / (Hbar * EV) ** 2
                                 / (2 * np.pi * THz) ** 2
                                 / num_grid)

    def get_frequency_shift(self):
        if self._cutoff_frequency is None:
            return self._frequency_shifts
        else: # Averaging frequency shifts by degenerate bands
            shifts = np.zeros_like(self._frequency_shifts)
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
                            shifts[i] = (self._frequency_shifts[bi_set].sum() /
                                         len(bi_set))
                        else:
                            shifts[:, i] = (
                                self._frequency_shifts[:, bi_set].sum(axis=1) /
                                len(bi_set))
            return shifts

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
        
    def set_epsilon(self, epsilon):
        if epsilon is None:
            self._epsilon = None
        else:
            self._epsilon = float(epsilon)

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
        
    # def _run_c_with_band_indices(self):
    #     import anharmonic._phono3py as phono3c
    #     phono3c.imag_self_energy_at_bands(self._imag_self_energy,
    #                                       self._fc3_normal_squared,
    #                                       self._grid_point_triplets,
    #                                       self._triplet_weights,
    #                                       self._frequencies,
    #                                       self._band_indices,
    #                                       self._temperature,
    #                                       self._sigma,
    #                                       self._unit_conversion,
    #                                       self._cutoff_frequency)

    # def _run_c_with_fpoints(self):
    #     import anharmonic._phono3py as phono3c
    #     for i, fpoint in enumerate(self._fpoints):
    #         phono3c.imag_self_energy(self._imag_self_energy[i],
    #                                  self._fc3_normal_squared,
    #                                  self._grid_point_triplets,
    #                                  self._triplet_weights,
    #                                  self._frequencies,
    #                                  fpoint,
    #                                  self._temperature,
    #                                  self._sigma,
    #                                  self._unit_conversion,
    #                                  self._cutoff_frequency)

    def _run_py_with_band_indices(self):
        for i, (triplet, w, interaction) in enumerate(
            zip(self._grid_point_triplets,
                self._triplet_weights,
                self._fc3_normal_squared)):

            freqs = self._frequencies[triplet]
            for j, bi in enumerate(self._band_indices):
                if self._temperature > 0:
                    self._frequency_shifts[j] += (
                        self._frequency_shifts_at_bands(
                            j, bi, freqs, interaction, w))
                else:
                    self._frequency_shifts[j] += (
                        self._frequency_shifts_at_bands_0K(
                            j, bi, freqs, interaction, w))

        self._frequency_shifts *= self._unit_conversion

    def _frequency_shifts_at_bands(self, i, bi, freqs, interaction, weight):
        sum_d = 0
        for (j, k) in list(np.ndindex(interaction.shape[1:])):
            if (freqs[1, j] > self._cutoff_frequency and
                freqs[2, k] > self._cutoff_frequency):
                d = 0.0
                n2 = occupation(freqs[1, j], self._temperature)
                n3 = occupation(freqs[2, k], self._temperature)
                f1 = freqs[0, bi] + freqs[1, j] + freqs[2, k]
                f2 = freqs[0, bi] - freqs[1, j] - freqs[2, k]
                f3 = freqs[0, bi] - freqs[1, j] + freqs[2, k]
                f4 = freqs[0, bi] + freqs[1, j] - freqs[2, k]

                # if abs(f1) > self._epsilon:
                #     d -= (n2 + n3 + 1) / f1
                # if abs(f2) > self._epsilon:
                #     d += (n2 + n3 + 1) / f2
                # if abs(f3) > self._epsilon:
                #     d -= (n2 - n3) / f3
                # if abs(f4) > self._epsilon:
                #     d += (n2 - n3) / f4
                d -= (n2 + n3 + 1) * f1 / (f1 ** 2 + self._epsilon ** 2)
                d += (n2 + n3 + 1) * f2 / (f2 ** 2 + self._epsilon ** 2)
                d -= (n2 - n3) * f3 / (f3 ** 2 + self._epsilon ** 2)
                d += (n2 - n3) * f4 / (f4 ** 2 + self._epsilon ** 2)

                sum_d += d * interaction[i, j, k] * weight
        return sum_d

    def _frequency_shifts_at_bands_0K(self, i, bi, freqs, interaction, weight):
        sum_d = 0
        for (j, k) in list(np.ndindex(interaction.shape[1:])):
            if (freqs[1, j] > self._cutoff_frequency and
                freqs[2, k] > self._cutoff_frequency):
                d = 0.0
                f1 = freqs[0, bi] + freqs[1, j] + freqs[2, k]
                f2 = freqs[0, bi] - freqs[1, j] - freqs[2, k]

                # if abs(f1) > self._epsilon:
                #     d -= 1.0 / f1
                # if abs(f2) > self._epsilon:
                #     d += 1.0 / f2
                d -= 1.0 * f1 / (f1 ** 2 + self._epsilon ** 2)
                d += 1.0 * f2 / (f2 ** 2 + self._epsilon ** 2)

                sum_d += d * interaction[i, j, k] * weight
        return sum_d

    def _run_py_with_fpoints(self):
        for i, (triplet, w, interaction) in enumerate(
            zip(self._grid_point_triplets,
                self._triplet_weights,
                self._fc3_normal_squared)):

            # freqs[2, num_band]
            freqs = self._frequencies[triplet[1:]]
            if self._temperature > 0:
                self._frequency_shifts_with_fpoints(freqs, interaction, w)
            else:
                self._frequency_shifts_with_fpoints_0K(freqs, interaction, w)

        self._frequency_shifts *= self._unit_conversion

    def _frequency_shifts_with_fpoints(self, freqs, interaction, weight):
        for (i, j, k) in list(np.ndindex(interaction.shape)):
            if (freqs[0, j] > self._cutoff_frequency and
                freqs[1, k] > self._cutoff_frequency):

                d = np.zeros(len(self._fpoints), dtype='double')
                n2 = occupation(freqs[0, j], self._temperature)
                n3 = occupation(freqs[1, k], self._temperature)
                f1s = self._fpoints + freqs[0, j] + freqs[1, k]
                f2s = self._fpoints - freqs[0, j] - freqs[1, k]
                f3s = self._fpoints - freqs[0, j] + freqs[1, k]
                f4s = self._fpoints + freqs[0, j] - freqs[1, k]

                for l, (f1, f2, f3, f4) in enumerate(
                    zip(f1s, f2s, f3s, f4s)):
                    if abs(f1) > self._epsilon:
                        d[l] -= (n2 + n3 + 1) / f1
                    if abs(f2) > self._epsilon:
                        d[l] += (n2 + n3 + 1) / f2
                    if abs(f3) > self._epsilon:
                        d[l] -= (n2 - n3) / f3
                    if abs(f4) > self._epsilon:
                        d[l] += (n2 - n3) / f4

                d *= interaction[i, j, k] * weight 
                self._frequency_shifts[:, i] += d

    def _frequency_shifts_with_fpoints_0K(self, freqs, interaction, weight):
        for (i, j, k) in list(np.ndindex(interaction.shape)):
            if (freqs[0][j] > self._cutoff_frequency and
                freqs[1][k] > self._cutoff_frequency):
                d = np.zeros(len(self._fpoints), dtype='double')
                f1s = self._fpoints + freqs[0, j] + freqs[1, k]
                f2s = self._fpoints - freqs[0, j] - freqs[1, k]

                for l, (f1, f2) in enumerate(zip(f1s, f2s)):
                    if abs(f1) > self._epsilon:
                        d[l] -= 1.0 / f1
                    if abs(f2) > self._epsilon:
                        d[l] += 1.0 / f2

                d *= interaction[i, j, k] * weight 
                self._frequency_shifts[:, i] += d

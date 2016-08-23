import numpy as np
from phonopy.cui.settings import Settings, ConfParser, fracval
import sys
class Phono3pySettings(Settings):
    def __init__(self):
        Settings.__init__(self)
        self._band_indices = None
        self._coarse_mesh_shifts = None
        self._cutoff_pair = None
        self._cutoff_delta = None
        self._cutoff_triplet = None
        self._cutoff_radius = None
        self._cutoff_frequency = 1e-4
        self._cutoff_hfrequency = None
        self._cutoff_lifetime = 1e-4 # in second
        self._diff_kappa = 1e-5 # relative
        self._diff_gamma = 1e-5 # relative
        self._grid_points = None
        self._ion_clamped = False
        self._is_ite=False
        self._is_ite_cg = False
        self._is_ise=False
        self._is_bterta = False
        self._is_linewidth = False
        self._nu = None
        self._is_isotope = False
        self._is_frequency_shift = False
        self._max_ite = None
        self._mass_variances = None
        self._max_freepath = None
        self._mesh_divisors = None
        self._multiple_sigmas = None
        self._adaptive_sigma_step = 0
        self._no_kappa_stars = False
        self._read_amplitude = False
        self._read_gamma = False
        self._supercell_matrix_extra = None
        self._temperatures = None
        self._write_amplitude = False
        self._write_gamma = False
        self._write_triplet = False
        self._read_fc2 = False
        self._read_fc3 = False
        self._wstep = None
        self._length = None
        self._read_fc2_extra=False

        
    def set_band_indices(self, band_indices):
        self._band_indices = band_indices

    def get_band_indices(self):
        return self._band_indices

    def set_coarse_mesh_shifts(self, coarse_mesh_shifts):
        self._coarse_mesh_shifts = coarse_mesh_shifts

    def get_coarse_mesh_shifts(self):
        return self._coarse_mesh_shifts

    def set_cutoff_pair(self, cutoff_pair):
        self._cutoff_pair = cutoff_pair

    def get_cutoff_pair(self):
        return self._cutoff_pair

    def set_cutoff_triplet(self, cutoff_triplet):
        self._cutoff_triplet = cutoff_triplet

    def get_cutoff_triplet(self):
        return self._cutoff_triplet

    def set_cutoff_radius(self, cutoff_radius):
        self._cutoff_radius = cutoff_radius

    def get_cutoff_radius(self):
        return self._cutoff_radius

    def set_cutoff_delta(self, cutoff_delta):
        self._cutoff_delta = cutoff_delta

    def get_cutoff_delta(self):
        return self._cutoff_delta

    def set_cutoff_frequency(self, cutoff_frequency):
        self._cutoff_frequency = cutoff_frequency

    def get_cutoff_frequency(self):
        return self._cutoff_frequency

    def set_cutoff_hfrequency(self, cutoff_hfrequency):
        self._cutoff_hfrequency = cutoff_hfrequency

    def get_cutoff_hfrequency(self):
        return self._cutoff_hfrequency

    def set_cutoff_lifetime(self, cutoff_lifetime):
        self._cutoff_lifetime = cutoff_lifetime

    def get_cutoff_lifetime(self):
        return self._cutoff_lifetime

    def set_diff_kappa(self, diff_kappa):
        self._diff_kappa = diff_kappa

    def get_diff_kappa(self):
        return self._diff_kappa

    def set_diff_gamma(self, diff_gamma):
        self._diff_gamma = diff_gamma

    def get_diff_gamma(self):
        return self._diff_gamma

    def set_grid_points(self, grid_points):
        self._grid_points = grid_points

    def get_grid_points(self):
        return self._grid_points

    def set_ion_clamped(self, ion_clamped):
        self._ion_clamped = ion_clamped

    def get_ion_clamped(self):
        return self._ion_clamped

    def set_is_ite(self,is_ite):
        self._is_ite= is_ite

    def get_is_ite(self):
        return self._is_ite

    def set_is_ite_cg(self,is_ite_cg):
        self._is_ite_cg= is_ite_cg

    def get_is_ite_cg(self):
        return self._is_ite_cg

    def set_is_ise(self,is_ise):
        self._is_ise= is_ise

    def get_is_ise(self):
        return self._is_ise

    def set_is_bterta(self, is_bterta):
        self._is_bterta = is_bterta

    def get_is_bterta(self):
        return self._is_bterta

    def set_is_linewidth(self, is_linewidth):
        self._is_linewidth = is_linewidth

    def get_is_linewidth(self):
        return self._is_linewidth

    def set_nu(self, nu):
        self._nu = nu

    def get_nu(self):
        return self._nu

    def set_is_isotope(self,isotope):
        self._is_isotope = isotope

    def get_is_isotope(self):
        return self._is_isotope

    def set_read_fc2(self, read_fc2):
        self._read_fc2 = read_fc2

    def get_read_fc2(self):
        return self._read_fc2

    def set_read_fc2_extra(self, read_fc2_extra):
        self._read_fc2_extra = read_fc2_extra

    def get_read_fc2_extra(self):
        return self._read_fc2_extra

    def set_read_fc3(self, read_fc3):
        self._read_fc3 = read_fc3

    def get_read_fc3(self):
        return self._read_fc3

    def set_is_frequency_shift(self, is_frequency_shift):
        self._is_frequency_shift = is_frequency_shift

    def get_is_frequency_shift(self):
        return self._is_frequency_shift

    def set_mass_variances(self, mass_variances):
        self._mass_variances = mass_variances

    def get_mass_variances(self):
        return self._mass_variances

    def set_max_ite(self, max_ite):
        self._max_ite = max_ite

    def get_max_ite(self):
        return self._max_ite

    def set_length(self, length):
        self._length = length

    def get_length(self):
        return self._length

    def set_max_freepath(self, max_freepath):
        self._max_freepath = max_freepath

    def get_max_freepath(self):
        return self._max_freepath

    def set_mesh_divisors(self, mesh_divisors):
        self._mesh_divisors = mesh_divisors

    def get_mesh_divisors(self):
        return self._mesh_divisors

    def set_multiple_sigmas(self, multiple_sigmas):
        self._multiple_sigmas = multiple_sigmas

    def get_multiple_sigmas(self):
        return self._multiple_sigmas

    def set_adaptive_sigma_step(self, adaptive_sigma_step):
        self._adaptive_sigma_step=adaptive_sigma_step

    def get_adaptive_sigma_step(self):
        return self._adaptive_sigma_step

    def set_kappa_write_step(self, wstep):
        self._wstep=wstep

    def get_kappa_write_step(self):
        return self._wstep

    def set_no_kappa_stars(self, no_kappa_stars):
        self._no_kappa_stars = no_kappa_stars

    def get_no_kappa_stars(self):
        return self._no_kappa_stars

    def set_read_gamma(self, read_gamma):
        self._read_gamma = read_gamma

    def get_read_gamma(self):
        return self._read_gamma

    def set_read_amplitude(self, read_amplitude):
        self._read_amplitude = read_amplitude

    def get_read_amplitude(self):
        return self._read_amplitude

    def set_supercell_matrix_extra(self, matrix):
        self._supercell_matrix_extra = matrix

    def get_supercell_matrix_extra(self):
        return self._supercell_matrix_extra

    def set_temperatures(self, temperatures):
        if temperatures is not None:
            self._temperatures = temperatures
        else:
            self._temperatures = np.arange(np.int(self.get_min_temperature()),
                                           np.int(self.get_max_temperature())+1,
                                           np.int(self.get_temperature_step()))

    def get_temperatures(self):
        return self._temperatures

    def set_write_amplitude(self, write_amplitude):
        self._write_amplitude = write_amplitude

    def get_write_amplitude(self):
        return self._write_amplitude

    def set_write_gamma(self, write_gamma):
        self._write_gamma = write_gamma

    def get_write_gamma(self):
        return self._write_gamma

    def set_write_triplet(self, write_triplet):
        self._write_triplet = write_triplet

    def get_write_triplet(self):
        return self._write_triplet

class PhonompySettings(Settings):
    def __init__(self):
        Settings.__init__(self)
        self._band_indices = None
        self._coarse_mesh_shifts = None
        self._grid_points = None
        self._is_bterta = False
        self._is_linewidth = False
        self._no_kappa_stars = False
        self._read_gamma = False
        self._supercell_matrix_extra = None
        self._temperature = None
        self._write_gamma = False
        self._read_fc2 = False
        self._read_fc2_extra=False
        self._mesh_shift = [False, False, False]


    def set_band_indices(self, band_indices):
        self._band_indices = band_indices

    def get_band_indices(self):
        return self._band_indices

    def set_mesh_shift(self, mesh_shift):
        self._mesh_shift = mesh_shift

    def get_mesh_shift(self):
        return self._mesh_shift

    def set_grid_points(self, grid_points):
        self._grid_points = grid_points

    def get_grid_points(self):
        return self._grid_points

    def set_is_bterta(self, is_bterta):
        self._is_bterta = is_bterta

    def get_is_bterta(self):
        return self._is_bterta

    def set_is_linewidth(self, is_linewidth):
        self._is_linewidth = is_linewidth

    def get_is_linewidth(self):
        return self._is_linewidth

    def set_read_fc2(self, read_fc2):
        self._read_fc2 = read_fc2

    def get_read_fc2(self):
        return self._read_fc2

    def set_read_fc2_extra(self, read_fc2_extra):
        self._read_fc2_extra = read_fc2_extra

    def get_read_fc2_extra(self):
        return self._read_fc2_extra

    def set_read_fc3(self, read_fc3):
        self._read_fc3 = read_fc3

    def get_read_fc3(self):
        return self._read_fc3

    def set_no_kappa_stars(self, no_kappa_stars):
        self._no_kappa_stars = no_kappa_stars

    def get_no_kappa_stars(self):
        return self._no_kappa_stars

    def set_read_gamma(self, read_gamma):
        self._read_gamma = read_gamma

    def get_read_gamma(self):
        return self._read_gamma

    def set_supercell_matrix_extra(self, matrix):
        self._supercell_matrix_extra = matrix

    def get_supercell_matrix_extra(self):
        return self._supercell_matrix_extra

    def set_temperature(self, temperature):
        self._temperature = temperature

    def get_temperature(self):
        return self._temperature

    def set_write_gamma(self, write_gamma):
        self._write_gamma = write_gamma

    def get_write_gamma(self):
        return self._write_gamma


class Phono3pyConfParser(ConfParser):
    def __init__(self, filename=None, options=None, option_list=None):
        ConfParser.__init__(self, filename, options, option_list)
        self._read_options()
        self._parse_conf()
        self._settings = Phono3pySettings()
        self._set_settings()

    def _read_options(self):
        for opt in self._option_list:
            if opt.dest == 'supercell_dimension_extra':
                if self._options.supercell_dimension_extra is not None:
                    self._confs['dim_extra'] = self._options.supercell_dimension_extra

            if opt.dest == 'band_indices':
                if self._options.band_indices is not None:
                    self._confs['band_indices'] = self._options.band_indices

            if opt.dest == 'cutoff_pair':
                if self._options.cutoff_pair is not None:
                    self._confs['cutoff_pair'] = \
                        self._options.cutoff_pair

            if opt.dest == 'cutoff_triplet':
                if self._options.cutoff_triplet is not None:
                    self._confs['cutoff_triplet'] = \
                        self._options.cutoff_triplet

            if opt.dest == "diff_kappa":
                if self._options.diff_kappa is not None:
                    self._confs['diff_kappa'] = self._options.diff_kappa

            if opt.dest == "diff_gamma":
                if self._options.diff_gamma is not None:
                    self._confs['diff_gamma'] = self._options.diff_gamma

            if opt.dest == 'cutoff_radius':
                if self._options.cutoff_radius is not None:
                    self._confs['cutoff_radius'] = \
                        self._options.cutoff_radius

            if opt.dest == "cutoff_delta":
                if self._options.cutoff_delta is not None:
                    self._confs["cutoff_delta"]= self._options.cutoff_delta

            if opt.dest == 'cutoff_frequency':
                if self._options.cutoff_frequency is not None:
                    self._confs['cutoff_frequency'] = self._options.cutoff_frequency

            if opt.dest == 'cutoff_hfrequency':
                if self._options.cutoff_hfrequency is not None:
                    self._confs['cutoff_hfrequency'] = self._options.cutoff_hfrequency

            if opt.dest == 'cutoff_lifetime':
                if self._options.cutoff_lifetime is not None:
                    self._confs['cutoff_lifetime'] = self._options.cutoff_lifetime

            if opt.dest == 'grid_points':
                if self._options.grid_points is not None:
                    self._confs['grid_points'] = self._options.grid_points

            if opt.dest == 'ion_clamped':
                if self._options.ion_clamped:
                    self._confs['ion_clamped'] = '.true.'

            if opt.dest == 'is_ite':
                if self._options.is_ite:
                    self._confs['ite']='.true.'

            if opt.dest == "length":
                if self._options.length is not None:
                    self._confs['length'] = self._options.length

            if opt.dest == 'is_ite_cg':
                if self._options.is_ite_cg:
                    self._confs['ite_cg']='.true.'

            if opt.dest == 'is_ise':
                if self._options.is_ise:
                    self._confs['ise']='.true.'

            if opt.dest == 'is_bterta':
                if self._options.is_bterta:
                    self._confs['bterta'] = '.true.'

            if opt.dest == 'is_linewidth':
                if self._options.is_linewidth:
                    self._confs['linewidth'] = '.true.'

            if opt.dest == 'nu':
                if self._options.nu:
                    self._confs['nu']= self._options.nu

            if opt.dest == 'read_fc2':
                if self._options.read_fc2:
                    self._confs['read_fc2']='.true.'

            if opt.dest == 'read_fc2_extra':
                if self._options.read_fc2_extra:
                    self._confs['read_fc2_extra']='.true.'

            if opt.dest == 'read_fc3':
                if self._options.read_fc3:
                    self._confs['read_fc3']='.true.'

            if opt.dest == 'is_frequency_shift':
                if self._options.is_frequency_shift:
                    self._confs['frequency_shift'] = '.true.'

            if opt.dest == "max_ite":
                if self._options.max_ite is not None:
                    self._confs['max_ite']=self._options.max_ite

            if opt.dest == 'mass_variances':
                if self._options.mass_variances is not None:
                    self._confs['mass_variances'] = self._options.mass_variances

            if opt.dest == 'max_freepath':
                if self._options.max_freepath is not None:
                    self._confs['max_freepath'] = self._options.max_freepath

            if opt.dest == 'mesh_divisors':
                if self._options.mesh_divisors is not None:
                    self._confs['mesh_divisors'] = self._options.mesh_divisors

            if opt.dest == 'multiple_sigmas':
                if self._options.multiple_sigmas is not None:
                    self._confs['multiple_sigmas'] = self._options.multiple_sigmas

            if opt.dest == "adaptive_sigma_step":
                if self._options.adaptive_sigma_step is not None:
                    self._confs['adaptive_sigma_step'] = self._options.adaptive_sigma_step

            if opt.dest == 'wstep':
                if self._options.wstep is not None:
                    self._confs['kappa_write_step'] = self._options.wstep

            if opt.dest == 'no_kappa_stars':
                if self._options.no_kappa_stars:
                    self._confs['no_kappa_stars'] = '.true.'

            if opt.dest == 'read_amplitude':
                if self._options.read_amplitude:
                    self._confs['read_amplitude'] = '.true.'

            if opt.dest == 'read_gamma':
                if self._options.read_gamma:
                    self._confs['read_gamma'] = '.true.'

            if opt.dest == 'temperatures':
                if self._options.temperatures is not None:
                    self._confs['temperatures'] = self._options.temperatures

            if opt.dest == 'write_amplitude':
                if self._options.write_amplitude:
                    self._confs['write_amplitude'] = '.true.'

            if opt.dest == 'write_gamma':
                if self._options.write_gamma:
                    self._confs['write_gamma'] = '.true.'

            if opt.dest == "write_triplet":
                if self._options.write_triplet:
                    self._confs["write_triplet"] = '.true.'

    def _parse_conf(self):
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == 'dim_extra':
                matrix = [ int(x) for x in confs['dim_extra'].replace(",", " ").split() ]
                if len(matrix) == 9:
                    matrix = np.array(matrix).reshape(3, 3)
                elif len(matrix) == 3:
                    matrix = np.diag(matrix)
                else:
                    self.setting_error("Number of elements of dim2 has to be 3 or 9.")

                if matrix.shape == (3, 3):
                    if np.linalg.det(matrix) < 1:
                        self.setting_error('Determinant of supercell matrix has to be positive.')
                    else:
                        self.set_parameter('dim_extra', matrix)

            if conf_key == 'band_indices':
                vals = []
                for sum_set in confs['band_indices'].split(','):
                    vals.append([int(x) - 1 for x in sum_set.split()])
                self.set_parameter('band_indices', vals)

            if conf_key == 'cutoff_pair':
                cutpair=[ float(x) for x in confs['cutoff_pair'].replace(","," ").split()]
                self.set_parameter('cutoff_pair',cutpair)

            if conf_key == 'cutoff_triplet':
                cutfc3 = [ float(x) for x in confs['cutoff_triplet'].replace(","," ").split()]
                self.set_parameter('cutoff_triplet',cutfc3)

            if conf_key == 'cutoff_radius':
                cutfc3q = [ float(x) for x in confs['cutoff_radius'].replace(","," ").split()]
                self.set_parameter('cutoff_radius',cutfc3q)

            if conf_key == "cutoff_delta":
                self.set_parameter("cutoff_delta", float(confs["cutoff_delta"]))

            if conf_key == 'cutoff_frequency':
                self.set_parameter('cutoff_frequency', confs['cutoff_frequency'])

            if conf_key == 'cutoff_hfrequency':
                self.set_parameter('cutoff_hfrequency', confs['cutoff_hfrequency'])

            if conf_key == 'cutoff_lifetime':
                self.set_parameter('cutoff_lifetime', confs['cutoff_lifetime'])

            if conf_key == "diff_kappa":
                self.set_parameter("diff_kappa", confs['diff_kappa'])

            if conf_key == "diff_gamma":
                self.set_parameter("diff_gamma", confs['diff_gamma'])

            if conf_key == 'grid_points':
                vals = [int(x) for x in confs['grid_points'].replace(",", " ").split()]
                self.set_parameter('grid_points', vals)

            if conf_key == 'ion_clamped':
                if confs['ion_clamped'] == '.true.':
                    self.set_parameter('ion_clamped', True)

            if conf_key == "ite":
                if confs['ite'] == ".true.":
                    self.set_parameter('is_ite',True)

            if conf_key == "ite_cg":
                if confs['ite_cg'] == ".true.":
                    self.set_parameter('is_ite_cg',True)

            if conf_key == "length":
                self.set_parameter('length', confs['length'])

            if conf_key == "ise":
                if confs['ise'] == ".true.":
                    self.set_parameter('is_ise',True)

            if conf_key == 'bterta':
                if confs['bterta'] == '.true.':
                    self.set_parameter('is_bterta', True)

            if conf_key == 'linewidth':
                if confs['linewidth'] == '.true.':
                    self.set_parameter('is_linewidth', True)

            if conf_key == 'nu':
                if confs['nu'] is not None:
                    nu  = confs['nu'].strip().upper()[0]
                    if nu == "N" or nu == "U":
                        self.set_parameter('nu',nu)

            if conf_key == 'read_fc2':
                if confs['read_fc2']== '.true.':
                    self.set_parameter('read_fc2',True)

            if conf_key == 'read_fc2_extra':
                if confs['read_fc2_extra']== '.true.':
                    self.set_parameter('read_fc2_extra',True)

            if conf_key == 'read_fc3':
                if confs['read_fc3']== '.true.':
                    self.set_parameter('read_fc3',True)

            if conf_key == 'frequency_shift':
                if confs['frequency_shift'] == '.true.':
                    self.set_parameter('is_frequency_shift', True)

            if conf_key == 'mass_variances':
                vals = [fracval(x) for x in confs['mass_variances'].split()]
                if len(vals) < 1:
                    self.setting_error("Mass variance parameters are incorrectly set.")
                else:
                    self.set_parameter('mass_variances', vals)

            if conf_key == "max_ite":
                self.set_parameter("max_ite", int(confs['max_ite']))

            if conf_key == 'max_freepath':
                self.set_parameter('max_freepath', float(confs['max_freepath']))

            if conf_key == 'mesh_divisors':
                vals = [x for x in confs['mesh_divisors'].split()]
                if len(vals) == 3:
                    self.set_parameter('mesh_divisors', [int(x) for x in vals])
                elif len(vals) == 6:
                    divs = [int(x) for x in vals[:3]]
                    is_shift = [x.lower() == 't' for x in vals[3:]]
                    for i in range(3):
                        if is_shift[i] and (divs[i] % 2 != 0):
                            is_shift[i] = False
                            self.setting_error("Coarse grid shift along the " +
                                               ["first", "second", "third"][i] +
                                               " axis is not allowed.")
                    self.set_parameter('mesh_divisors', divs + is_shift)
                else:
                    self.setting_error("Mesh divisors are incorrectly set.")

            if conf_key == 'multiple_sigmas':
                vals = [fracval(x) for x in confs['multiple_sigmas'].replace(",", " ").split()]
                if len(vals) < 1:
                    self.setting_error("Mutiple sigmas are incorrectly set.")
                else:
                    self.set_parameter('multiple_sigmas', vals)

            if conf_key == "adaptive_sigma_step":
                self.set_parameter("adaptive_sigma_step", int(confs['adaptive_sigma_step']))

            if conf_key == 'kappa_write_step':
                self.set_parameter('kappa_write_step', int(confs['kappa_write_step']))

            if conf_key == 'no_kappa_stars':
                if confs['no_kappa_stars'] == '.true.':
                    self.set_parameter('no_kappa_stars', True)

            if conf_key == 'read_amplitude':
                if confs['read_amplitude'] == '.true.':
                    self.set_parameter('read_amplitude', True)

            if conf_key == 'read_gamma':
                if confs['read_gamma'] == '.true.':
                    self.set_parameter('read_gamma', True)

            if conf_key == 'temperatures':
                vals = [fracval(x) for x in confs['temperatures'].replace(",", " ").split()]
                if len(vals) < 1:
                    self.setting_error("Temperatures are incorrectly set.")
                else:
                    self.set_parameter('temperatures', vals)

            if conf_key == 'write_amplitude':
                if confs['write_amplitude'] == '.true.':
                    self.set_parameter('write_amplitude', True)

            if conf_key == 'write_gamma':
                if confs['write_gamma'] == '.true.':
                    self.set_parameter('write_gamma', True)

            if conf_key == "write_triplet":
                if confs['write_triplet'] == '.true.':
                    self.set_parameter('write_triplet', True)


    def _set_settings(self):
        ConfParser.set_settings(self)
        params = self._parameters

        # Supercell size for fc2
        if params.has_key('dim_extra'):
            self._settings.set_supercell_matrix_extra(params['dim_extra'])

        # Sets of band indices that are summed
        if params.has_key('band_indices'):
            self._settings.set_band_indices(params['band_indices'])

        # Cutoff distance between pairs of displaced atoms used for supercell
        # creation with displacements and making third-order force constants
        if params.has_key('cutoff_pair'):
            self._settings.set_cutoff_pair(
                params['cutoff_pair'])

        # Cutoff distance of third-order force constants. Elements where any
        # pair of atoms has larger distance than cut-off distance are set zero.
        if params.has_key('cutoff_triplet'):
            self._settings.set_cutoff_triplet(params['cutoff_triplet'])

        # Cutoff distance of reciprocal third-order force constants.
        if params.has_key('cutoff_radius'):
            self._settings.set_cutoff_radius(params['cutoff_radius'])

        #Cutoff sigmas for gaussian smearing.
        if params.has_key("cutoff_delta"):
            self._settings.set_cutoff_delta(params["cutoff_delta"])

        # Phonon modes larger than this frequency are ignored.
        if params.has_key('cutoff_hfrequency'):
            self._settings.set_cutoff_hfrequency(params['cutoff_hfrequency'])

        # Phonon modes smaller than this frequency are ignored.
        if params.has_key('cutoff_frequency'):
            self._settings.set_cutoff_frequency(params['cutoff_frequency'])

        # Cutoff lifetime used for thermal conductivity calculation
        if params.has_key('cutoff_lifetime'):
            self._settings.set_cutoff_lifetime(params['cutoff_lifetime'])

        # different lifetime (relative) used for iterative method
        if params.has_key('diff_kappa'):
            self._settings.set_diff_kappa(params['diff_kappa'])

        # Grid points
        if params.has_key('grid_points'):
            self._settings.set_grid_points(params['grid_points'])

        # Atoms are clamped under applied strain in Gruneisen parameter calculation
        if params.has_key('ion_clamped'):
            self._settings.set_ion_clamped(params['ion_clamped'])

        #Solve the Boltzmann Transport Equation Iteratively
        if params.has_key("is_ite"):
            self._settings.set_is_ite(params['is_ite'])

        #Solve the Boltzmann Transport Equation Iteratively using the conjugate gradient method
        if params.has_key("is_ite_cg"):
            self._settings.set_is_ite_cg(params['is_ite_cg'])

        #Get the image self energy for a specific grid point and band index
        if params.has_key("is_ise"):
            self._settings.set_is_ise(params['is_ise'])

        #The maximum iteration steps for the iterative method
        if params.has_key("max_ite"):
            self._settings.set_max_ite(params['max_ite'])

        #The sample length considering phonon boundary scattering
        if params.has_key("length"):
            self._settings.set_length(params['length'])

        # Calculate thermal conductivity in BTE-RTA
        if params.has_key('is_bterta'):
            self._settings.set_is_bterta(params['is_bterta'])

        # Calculate linewidths
        if params.has_key('is_linewidth'):
            self._settings.set_is_linewidth(params['is_linewidth'])
        # Tell Normal and Umklapp process apart
        if params.has_key('nu'):
            self._settings.set_nu(params['nu'])

        if params.has_key("read_fc2"):
            self._settings.set_read_fc2(params['read_fc2'])

        if params.has_key("read_fc2_extra"):
            self._settings.set_read_fc2_extra(params['read_fc2_extra'])

        if params.has_key("read_fc3"):
            self._settings.set_read_fc3(params['read_fc3'])

        # Calculate frequency_shifts
        if params.has_key('is_frequency_shift'):
            self._settings.set_is_frequency_shift(params['is_frequency_shift'])

        # Mass variance parameters
        if params.has_key('mass_variances'):
            self._settings.set_mass_variances(params['mass_variances'])

        # Maximum mean free path
        if params.has_key('max_freepath'):
            self._settings.set_max_freepath(params['max_freepath'])

        # Divisors for mesh numbers
        if params.has_key('mesh_divisors'):
            self._settings.set_mesh_divisors(params['mesh_divisors'][:3])
            if len(params['mesh_divisors']) > 3:
                self._settings.set_coarse_mesh_shifts(
                    params['mesh_divisors'][3:])

        # Multiple sigmas
        if params.has_key('multiple_sigmas'):
            self._settings.set_multiple_sigmas(params['multiple_sigmas'])

        #Adaptive smearing factor
        if params.has_key('adaptive_sigma_step'):
            if params['adaptive_sigma_step'] <= 0:
                print "Error! Step cannot be negative or 0"
                sys.exit(1)
            self._settings.set_adaptive_sigma_step(params['adaptive_sigma_step'])

        if params.has_key('kappa_write_step'):
            self._settings.set_kappa_write_step(params['kappa_write_step'])

        # Read phonon-phonon interaction amplitudes from hdf5
        if params.has_key('read_amplitude'):
            self._settings.set_read_amplitude(params['read_amplitude'])

        # Read gammas from hdf5
        if params.has_key('read_gamma'):
            self._settings.set_read_gamma(params['read_gamma'])
            
        # Sum partial kappa at q-stars
        if params.has_key('no_kappa_stars'):
            self._settings.set_no_kappa_stars(params['no_kappa_stars'])

        # Temperatures
        if params.has_key('temperatures'):
            self._settings.set_temperatures(params['temperatures'])

        # Write phonon-phonon interaction amplitudes to hdf5
        if params.has_key('write_amplitude'):
            self._settings.set_write_amplitude(params['write_amplitude'])

        # Write gamma to hdf5
        if params.has_key('write_gamma'):
            self._settings.set_write_gamma(params['write_gamma'])

        #write triplets to a file
        if params.has_key("write_triplet"):
            self._settings.set_write_triplet(params['write_triplet'])

class PhonompyConfParser(ConfParser):
    def __init__(self, filename=None, options=None, option_list=None):
        ConfParser.__init__(self, filename, options, option_list)
        self._read_options()
        self._parse_conf()
        self._settings = PhonompySettings()
        self._set_settings()

    def _read_options(self):
        for opt in self._option_list:
            if opt.dest == 'supercell_dimension_extra':
                if self._options.supercell_dimension_extra is not None:
                    self._confs['dim_extra'] = self._options.supercell_dimension_extra

            if opt.dest == 'band_indices':
                if self._options.band_indices is not None:
                    self._confs['band_indices'] = self._options.band_indices


            if opt.dest == 'grid_points':
                if self._options.grid_points is not None:
                    self._confs['grid_points'] = self._options.grid_points

            if opt.dest == 'is_bterta':
                if self._options.is_bterta:
                    self._confs['bterta'] = '.true.'

            if opt.dest == 'is_linewidth':
                if self._options.is_linewidth:
                    self._confs['linewidth'] = '.true.'

            if opt.dest == 'read_fc2':
                if self._options.read_fc2:
                    self._confs['read_fc2']='.true.'

            if opt.dest == 'read_fc2_extra':
                if self._options.read_fc2_extra:
                    self._confs['read_fc2_extra']='.true.'

            if opt.dest == 'no_kappa_stars':
                if self._options.no_kappa_stars:
                    self._confs['no_kappa_stars'] = '.true.'

            if opt.dest == 'read_gamma':
                if self._options.read_gamma:
                    self._confs['read_gamma'] = '.true.'

            if opt.dest == 'temperature':
                if self._options.temperature is not None:
                    self._confs['temperature'] = self._options.temperature

            if opt.dest == 'write_gamma':
                if self._options.write_gamma:
                    self._confs['write_gamma'] = '.true.'

            if opt.dest == "mesh_shift":
                if self._options.mesh_shift is not None:
                    self._confs['mesh_shift'] = self._options.mesh_shift

    def _parse_conf(self):
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == 'dim_extra':
                matrix = [ int(x) for x in confs['dim_extra'].replace(",", " ").split() ]
                if len(matrix) == 9:
                    matrix = np.array(matrix).reshape(3, 3)
                elif len(matrix) == 3:
                    matrix = np.diag(matrix)
                else:
                    self.setting_error("Number of elements of dim2 has to be 3 or 9.")

                if matrix.shape == (3, 3):
                    if np.linalg.det(matrix) < 1:
                        self.setting_error('Determinant of supercell matrix has to be positive.')
                    else:
                        self.set_parameter('dim_extra', matrix)

            if conf_key == "mesh_shift":
                vals = [fracval(x) for x in confs['mesh_shift'].replace(",", " ").split()]
                for i in range(3-len(vals)):
                    vals += [0.0] # Correction for incomplete shift set
                self.set_parameter('mesh_shift', vals[:3])

            if conf_key == 'band_indices':
                vals = []
                for sum_set in confs['band_indices'].replace(",", " ").split(','):
                    vals.append([int(x) - 1 for x in sum_set.replace(",", " ").split()])
                self.set_parameter('band_indices', vals)

            if conf_key == 'grid_points':
                vals = [int(x) for x in confs['grid_points'].replace(",", " ").split()]
                self.set_parameter('grid_points', vals)

            if conf_key == 'bterta':
                if confs['bterta'] == '.true.':
                    self.set_parameter('is_bterta', True)

            if conf_key == 'linewidth':
                if confs['linewidth'] == '.true.':
                    self.set_parameter('is_linewidth', True)

            if conf_key == 'read_fc2':
                if confs['read_fc2']== '.true.':
                    self.set_parameter('read_fc2',True)

            if conf_key == 'read_fc2_extra':
                if confs['read_fc2_extra']== '.true.':
                    self.set_parameter('read_fc2_extra',True)

            if conf_key == 'max_freepath':
                self.set_parameter('max_freepath', float(confs['max_freepath']))

            if conf_key == 'no_kappa_stars':
                if confs['no_kappa_stars'] == '.true.':
                    self.set_parameter('no_kappa_stars', True)

            if conf_key == 'read_gamma':
                if confs['read_gamma'] == '.true.':
                    self.set_parameter('read_gamma', True)

            if conf_key == 'temperature':
                vals = confs['temperature']
                self.set_parameter('temperature', vals)

            if conf_key == 'write_gamma':
                if confs['write_gamma'] == '.true.':
                    self.set_parameter('write_gamma', True)


    def _set_settings(self):
        ConfParser.set_settings(self)
        params = self._parameters

        # Supercell size for fc2
        if params.has_key('dim_extra'):
            self._settings.set_supercell_matrix_extra(params['dim_extra'])

        # Sets of band indices that are summed
        if params.has_key('band_indices'):
            self._settings.set_band_indices(params['band_indices'])

        if params.has_key('mesh_shift'):
            self._settings.set_mesh_shift(params['mesh_shift'])

        # Grid points
        if params.has_key('grid_points'):
            self._settings.set_grid_points(params['grid_points'])

        # Calculate thermal conductivity in BTE-RTA
        if params.has_key('is_bterta'):
            self._settings.set_is_bterta(params['is_bterta'])

        # Calculate linewidths
        if params.has_key('is_linewidth'):
            self._settings.set_is_linewidth(params['is_linewidth'])
        # Tell Normal and Umklapp process apart

        if params.has_key("read_fc2"):
            self._settings.set_read_fc2(params['read_fc2'])

        if params.has_key("read_fc2_extra"):
            self._settings.set_read_fc2_extra(params['read_fc2_extra'])

        # Read gammas from hdf5
        if params.has_key('read_gamma'):
            self._settings.set_read_gamma(params['read_gamma'])

        # Sum partial kappa at q-stars
        if params.has_key('no_kappa_stars'):
            self._settings.set_no_kappa_stars(params['no_kappa_stars'])

        # Temperatures
        if params.has_key('temperature'):
            self._settings.set_temperature(params['temperature'])

        # Write gamma to hdf5
        if params.has_key('write_gamma'):
            self._settings.set_write_gamma(params['write_gamma'])

        


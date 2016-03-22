import numpy as np
from anharmonic.phonon3.interaction import get_dynamical_matrix,  set_phonon_py
from anharmonic.phonon3.interaction import set_phonon_c
# from anharmonic.phonon3.triplets import get_bz_grid_address
from anharmonic.phonon3.imag_self_energy import gaussian
from anharmonic.phonon3.triplets import get_grid_address
# from phonopy.structure.tetrahedron_method import TetrahedronMethod
# from phonopy.phonon.tetrahedron_mesh import get_tetrahedra_frequencies
# from phonopy.structure.grid_points import GridPoints
import phonopy.structure.spglib as spg
from phonopy.units import VaspToTHz

class Isotope:
    def __init__(self,
                 mesh,
                 mass_variances, # length of list is num_atom.
                 primitive=None,
                 band_indices=None,
                 sigma=None,
                 frequency_factor_to_THz=VaspToTHz,
                 symprec=1e-5,
                 cutoff_frequency=None,
                 lapack_zheev_uplo='L'):
        self._mesh = np.array(mesh, dtype='intc')
        self._mass_variances = np.array(mass_variances, dtype='double')
        self._primitive = primitive
        self._band_indices = band_indices
        self._sigma = sigma
        self._symprec = symprec
        if cutoff_frequency is None:
            self._cutoff_frequency = 0
        else:
            self._cutoff_frequency = cutoff_frequency
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._lapack_zheev_uplo = lapack_zheev_uplo
        self._nac_q_direction = None
        
        self._grid_address = None
        self._bz_map = None
        self._grid_points = None
        self._phonon_done = None
        self._frequencies = None
        self._eigenvectors = None
        self._dm = None
        self._grid_point = None
        self._gamma = None
        self._tetrahedron_method = None
        
    def set_grid_point(self, grid_point):
        self._grid_point = grid_point
        num_band = self._primitive.get_number_of_atoms() * 3
        if self._band_indices is None:
            self._band_indices = np.arange(num_band, dtype='intc')
        else:
            self._band_indices = np.array(self._band_indices, dtype='intc')
            
        self._grid_points = np.arange(np.prod(self._mesh), dtype='intc')
        
        if self._grid_address is None:
            primitive_lattice = np.linalg.inv(self._primitive.get_cell())
            self._grid_address = get_grid_address(self._mesh)
            # self._grid_address, self._bz_map = get_bz_grid_address(
            #     self._mesh, primitive_lattice, with_boundary=True)
        
        if self._phonon_done is None:
            self._allocate_phonon()

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

    def get_gamma(self):
        return self._gamma
        
    def get_grid_address(self):
        return self._grid_address

    def get_phonons(self):
        return self._frequencies, self._eigenvectors, self._phonon_done
    
    def set_phonons(self, frequencies, eigenvectors, phonon_done, dm=None):
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._phonon_done = phonon_done
        if dm is not None:
            self._dm = dm

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             frequency_scale_factor=None,
                             decimals=None):
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
        self._set_phonon_c(self._grid_points)
        import anharmonic._phono3py as phono3c
        gamma = np.zeros(len(self._band_indices), dtype='double')
        phono3c.isotope_strength(gamma,
                                 self._grid_point,
                                 self._mass_variances,
                                 self._frequencies,
                                 self._eigenvectors,
                                 self._band_indices,
                                 np.prod(self._mesh),
                                 self._sigma,
                                 self._cutoff_frequency)
            
        self._gamma = np.pi / 2 / np.prod(self._mesh) * gamma

    def _run_py(self):
        for gp in self._grid_points:
            self._set_phonon_py(gp)
            
        mass_v = np.array([[m] * 3 for m in self._mass_variances],
                          dtype='double').flatten()
        t_inv = []
        for bi in self._band_indices:
            vec0 = self._eigenvectors[self._grid_point][:, bi].conj()
            f0 = self._frequencies[self._grid_point][bi]
            ti_sum = 0.0
            for i, gp in enumerate(self._grid_points):
                for j, (f, vec) in enumerate(
                        zip(self._frequencies[i], self._eigenvectors[i].T)):
                    if f < self._cutoff_frequency:
                        continue
                    ti_sum_band = np.sum(np.abs(vec * vec0) ** 2 * mass_v)
                    ti_sum += ti_sum_band * gaussian(f0 - f, self._sigma)
            t_inv.append(np.pi / 2 / np.prod(self._mesh) * f0 ** 2 * ti_sum)

        self._gamma = np.array(t_inv, dtype='double') / 2
            
    def _set_phonon_c(self, grid_points):
        set_phonon_c(self._dm,
                     self._frequencies,
                     self._eigenvectors,
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
        

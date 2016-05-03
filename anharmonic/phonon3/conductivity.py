import numpy as np
import sys
from phonopy.structure.spglib import relocate_BZ_grid_address, get_mappings
from phonopy.phonon.group_velocity import get_group_velocity
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.units import THzToEv, EV, THz, Angstrom, kb_J
from phonopy.phonon.thermal_properties import mode_cv as get_mode_cv
from anharmonic.phonon3.triplets import get_grid_address, reduce_grid_points, get_ir_grid_points,\
    from_coarse_to_dense_grid_points, get_kpoint_group, get_group_summation, get_group_inversion
from anharmonic.other.isotope import CollisionIso
from phonopy.structure.tetrahedron_method import TetrahedronMethod, TriagonalMethod
unit_to_WmK = ((THz * Angstrom) ** 2 / (Angstrom ** 3) * EV / THz /
               (2 * np.pi)) # 2pi comes from definition of lifetime.
ite_unit_to_WmK = kb_J / Angstrom ** 3 * ((THz * Angstrom) * THzToEv * EV / kb_J) ** 2  / THz/ (2 * np.pi)
# 2pi comes from the definition of tau

def get_degenerate_property(degeneracy_at_q, property_at_q):
    assert len(degeneracy_at_q) == len(property_at_q)
    prop = np.zeros_like(property_at_q)
    for iband in np.unique(degeneracy_at_q):
        deg_states = np.where(degeneracy_at_q == iband)
        prop[deg_states] = np.average(property_at_q[deg_states], axis=0)
    return prop

class Conductivity:
    def __init__(self,
                 interaction,
                 symmetry,
                 grid_points=None,
                 temperatures=np.arange(0, 1001, 10, dtype='double'),
                 sigmas=[],
                 is_isotope=False,
                 mass_variances=None,
                 mesh_divisors=None,
                 coarse_mesh_shifts=None,
                 boundary_mfp=None, # in micrometre
                 no_kappa_stars=False,
                 gv_delta_q=None, # finite difference for group veolocity
                 log_level=0,
                 write_tecplot=False):
        self._pp = interaction
        self._collision = None # has to be set derived class
        self._no_kappa_stars = no_kappa_stars
        self._gv_delta_q = gv_delta_q
        self._log_level = log_level
        self._sigmas = sigmas
        self._temperatures=temperatures
        self._primitive = self._pp.get_primitive()
        self._dm = self._pp.get_dynamical_matrix()
        self._frequency_factor_to_THz = self._pp.get_frequency_factor_to_THz()
        self._cutoff_frequency = self._pp.get_cutoff_frequency()
        self._boundary_mfp = boundary_mfp
        self._write_tecplot = write_tecplot
        self._mesh = None
        self._mesh_divisors = None
        self._coarse_mesh = None
        self._coarse_mesh_shifts = None
        self._dim = 3
        self._set_mesh_numbers(mesh_divisors=mesh_divisors,
                               coarse_mesh_shifts=coarse_mesh_shifts)
        self._symmetry = symmetry
        self._bz_grid_address=None
        self._bz_to_pp_map = None
        self._coarse_mesh_shifts = None
        if self._no_kappa_stars:
            self._kpoint_operations = np.array([np.eye(3)], dtype="intc")
            self._mappings = np.arange(np.prod(self._mesh))
            self._rot_mappings = np.zeros(len(self._mappings), dtype="intc")
        else:
            self._kpoint_operations = get_kpoint_group(self._mesh, self._pp.get_point_group_operations())
            (self._mappings, self._rot_mappings) =get_mappings(self._mesh,
                                            self._pp.get_point_group_operations(),
                                            qpoints=np.array([0,0,0],dtype="double"))
        self._kpoint_group_sum_map = get_group_summation(self._kpoint_operations)
        self._kpoint_group_inv_map = get_group_inversion(self._kpoint_operations)
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        self._rotations_cartesian = np.array(
            [similarity_transformation(rec_lat, r)
             for r in self._kpoint_operations], dtype='double')

        self._grid_points = None
        self._grid_weights = None
        self._grid_address = None
        self._ir_grid_points = None
        self._ir_grid_weights = None

        self._kappa = None
        self._mode_kappa = None
        self._gamma = None
        self._read_gamma = False
        self._read_gamma_iso = False
        self._frequencies = None
        self._gv = None
        self._gamma_iso = None
        volume = self._primitive.get_volume()
        self._conversion_factor = unit_to_WmK / volume
        self._kappa_factor = ite_unit_to_WmK / volume
        self._irr_index_mapping=np.where(np.unique(self._mappings)-self._mappings.reshape(-1,1)==0)[1]

        self._isotope = None
        self._mass_variances = None
        self._is_isotope = is_isotope
        if mass_variances is not None:
            self._is_isotope = True
        if self._is_isotope:
            self._set_isotope(mass_variances)
        self._grid_point_count = None
        self._degeneracies = None
        self._set_grid_properties(grid_points)
        self._bz_to_pp_map = self._pp.get_bz_to_pp_map()

    def get_mesh_divisors(self):
        return self._mesh_divisors

    def get_mesh_numbers(self):
        return self._mesh

    def get_group_velocities(self):
        return self._gv

    def get_mode_heat_capacities(self):
        pass

    def get_frequencies(self):
        return self._frequencies[self._grid_points]
        
    def get_qpoints(self):
        return self._qpoints
            
    def get_grid_points(self):
        return self._grid_points

    def get_grid_weights(self):
        return self._grid_weights
            
    def get_temperatures(self):
        return self._temperatures

    def set_temperature(self, i, temperature = None):
        self._itemp = i
        if self._temperatures is not None:
            self._temp = self._temperatures[i]
        else:
            self._temp = temperature

    def set_sigma(self, s, sigma = None):
        self._isigma = s
        if self._sigmas is not None:
            self._sigma = self._sigmas[s]
        else:
            self._sigma = sigma

    def set_temperatures(self, temperatures):
        self._temperatures = temperatures
        self._allocate_values()

    def set_gamma(self, gamma):
        self._gamma = gamma
        self._read_gamma = True
        
    def set_gamma_isotope(self, gamma_iso):
        self._gamma_iso = gamma_iso
        self._read_gamma_iso = True

    def get_gamma(self):
        return self._gamma
        
    def get_gamma_isotope(self):
        return self._gamma_iso
        
    def get_kappa(self):
        return self._kappa

    def get_mode_kappa(self):
        return self._mode_kappa

    def get_sigmas(self):
        return self._sigmas

    def get_grid_point_count(self):
        return self._grid_point_count

    def set_pp_grid_points_all(self):
        self._pp.set_grid_points(self._grid_points)

    def _run_at_grid_point(self):
        """This has to be implemented in the derived class"""
        pass

    def _allocate_values(self):
        """This has to be implemented in the derived class"""
        pass

    def _calculate_kappa(self):
        """This has to be implemented in the derived class"""
        pass

    def _get_cv(self, freqs):

        cv = np.zeros((len(self._temperatures), len(freqs)), dtype='double')
        # T/freq has to be large enough to avoid divergence.
        # Otherwise just set 0.
        for i, f in enumerate(freqs):
            finite_t = (self._temperatures > f / 100)
            if f > self._cutoff_frequency:
                cv[:, i] = np.where(
                    finite_t, get_mode_cv(
                        np.where(finite_t, self._temperatures, 10000),
                        f * THzToEv), 0)
        return cv

    def _set_grid_properties(self, grid_points):
        self._grid_address = self._pp.get_grid_address()

        if grid_points is not None: # Specify grid points
            self._grid_points = reduce_grid_points(
                self._mesh_divisors,
                self._grid_address,
                grid_points,
                coarse_mesh_shifts=self._coarse_mesh_shifts)
            (self._ir_grid_points,
             self._ir_grid_weights) = self._get_ir_grid_points()
        elif self._no_kappa_stars: # All grid points
            coarse_grid_address = get_grid_address(self._coarse_mesh)
            coarse_grid_points = np.arange(np.prod(self._coarse_mesh),
                                           dtype='intc')
            self._grid_points = from_coarse_to_dense_grid_points(
                self._mesh,
                self._mesh_divisors,
                coarse_grid_points,
                coarse_grid_address,
                coarse_mesh_shifts=self._coarse_mesh_shifts)
            self._grid_weights = np.ones(len(self._grid_points), dtype='intc')
            self._ir_grid_points = self._grid_points
            self._ir_grid_weights = self._grid_weights
        else: # Automatic sampling
            self._grid_points, self._grid_weights = self._get_ir_grid_points()
            self._ir_grid_points = self._grid_points
            self._ir_grid_weights = self._grid_weights

        self._qpoints = np.array(self._grid_address[self._grid_points] /
                                 self._mesh.astype('double'),
                                 dtype='double', order='C')

        self._grid_point_count = 0
        self._pp.set_phonons(self._grid_points)
        self._frequencies = self._pp.get_phonons()[0][self._grid_points]
        self._degeneracies = self._pp.get_degeneracy()[self._grid_points]
        if self._write_tecplot:
            self._dim = np.count_nonzero(np.array(self._mesh)>1)
            #only dim=2 or 3 are implemented
            assert self._dim>1, "The dimention of the given system is %d, but only dim=2 or 3 are implemented" %self._dim
            self.set_bz_grid_points()
            self.tetrahedra_specify()

    def _set_gamma_isotope_at_sigmas(self, i):
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                print "Calculating Gamma of ph-isotope with",
                if sigma is None:
                    print "tetrahedron method"
                else:
                    print "sigma=%s" % sigma
            pp_freqs, pp_eigvecs, pp_phonon_done = self._pp.get_phonons()
            self._isotope.set_sigma(sigma)
            self._isotope.set_phonons(pp_freqs,
                                      pp_eigvecs,
                                      pp_phonon_done,
                                      dm=self._dm)
            gp = self._grid_points[i]
            self._isotope.set_grid_point(gp)
            self._isotope.run()
            self._gamma_iso[j, i] = self._isotope.get_gamma()

    def _set_mesh_numbers(self, mesh_divisors=None, coarse_mesh_shifts=None):
        self._mesh = self._pp.get_mesh_numbers()

        if mesh_divisors is None:
            self._mesh_divisors = np.array([1, 1, 1], dtype='intc')
        else:
            self._mesh_divisors = []
            for i, (m, n) in enumerate(zip(self._mesh, mesh_divisors)):
                if m % n == 0:
                    self._mesh_divisors.append(n)
                else:
                    self._mesh_divisors.append(1)
                    print ("Mesh number %d for the " +
                           ["first", "second", "third"][i] + 
                           " axis is not dividable by divisor %d.") % (m, n)
            self._mesh_divisors = np.array(self._mesh_divisors, dtype='intc')
            if coarse_mesh_shifts is None:
                self._coarse_mesh_shifts = [False, False, False]
            else:
                self._coarse_mesh_shifts = coarse_mesh_shifts
            for i in range(3):
                if (self._coarse_mesh_shifts[i] and
                    (self._mesh_divisors[i] % 2 != 0)):
                    print ("Coarse grid along " +
                           ["first", "second", "third"][i] + 
                           " axis can not be shifted. Set False.")
                    self._coarse_mesh_shifts[i] = False

        self._coarse_mesh = self._mesh / self._mesh_divisors

        if self._log_level:
            print ("Lifetime sampling mesh: [ %d %d %d ]" %
                   tuple(self._mesh / self._mesh_divisors))

    def _get_ir_grid_points(self):
        if self._coarse_mesh_shifts is None:
            mesh_shifts = [False, False, False]
        else:
            mesh_shifts = self._coarse_mesh_shifts
        (coarse_grid_points,
         coarse_grid_weights,
         coarse_grid_address) = get_ir_grid_points(
            self._coarse_mesh,
            self._primitive,
            mesh_shifts=mesh_shifts)
        grid_points = from_coarse_to_dense_grid_points(
            self._mesh,
            self._mesh_divisors,
            coarse_grid_points,
            coarse_grid_address,
            coarse_mesh_shifts=self._coarse_mesh_shifts)
        grid_weights = coarse_grid_weights

        assert grid_weights.sum() == np.prod(self._mesh /
                                             self._mesh_divisors)

        return grid_points, grid_weights

    def set_bz_grid_points(self):
        self._bz_grid_address, self._bz_map , self._bz_to_pp_map= \
            relocate_BZ_grid_address(self._grid_address,
                                     self._mesh,
                                     np.linalg.inv(self._primitive.get_cell()),
                                     is_bz_map_to_orig=True)

    def tetrahedra_specify(self, lang="C"):
        reciprocal_lattice = np.linalg.inv(self._primitive.get_cell())
        mesh=self._mesh
        if self._dim == 3:
            self._tetra=TetrahedronMethod(reciprocal_lattice, mesh=mesh)
            relative_address = self._tetra.get_tetrahedra()
            if lang=="C":
                self._unique_vertices = self.get_unique_tetrahedra_C(relative_address)
            else:
                self._unique_vertices = self.get_unique_tetrahedra_py(relative_address)
        elif self._dim == 2:
            self._tri = TriagonalMethod(reciprocal_lattice, mesh=mesh)
            relative_address = self._tri.get_triagonal()
            if lang == "C":
                self._unique_vertices = self.get_unique_triagonal_C(relative_address)
            else:
                self._unique_vertices = self.get_unique_triagonal_py(relative_address)

    def _set_isotope(self, mass_variances):
        if mass_variances is True:
            mv = None
        else:
            mv = mass_variances
        self._isotope = CollisionIso(
            self._mesh,
            self._primitive,
            mass_variances=mv,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            symprec=self._symmetry.get_symmetry_tolerance(),
            cutoff_frequency=self._cutoff_frequency,
            lapack_zheev_uplo=self._pp.get_lapack_zheev_uplo())
        self._mass_variances = self._isotope.get_mass_variances()
        
    def _set_gv(self, i):
        # Group velocity [num_freqs, 3]
        gv  = self._get_gv(self._qpoints[i])
        self._gv[i] = gv
        # if self._degeneracies is not None:
        #     deg = self._degeneracies[i]
        #     self._gv[i] = get_degenerate_property(deg, gv)

    def _get_gv(self, q):
        return get_group_velocity(
            q,
            self._dm,
            symmetry=self._symmetry,
            q_length=self._gv_delta_q,
            frequency_factor_to_THz=self._frequency_factor_to_THz)

    def _get_main_diagonal(self, i, j, k):
        num_band = self._primitive.get_number_of_atoms() * 3
        main_diagonal = self._gamma[j, k, i].copy()
        if self._gamma_iso is not None:
            main_diagonal += self._gamma_iso[j, i]
        if self._boundary_mfp is not None:
            main_diagonal += self._get_boundary_scattering(i)

        # if self._boundary_mfp is not None:
        #     for l in range(num_band):
        #         # Acoustic modes at Gamma are avoided.
        #         if i == 0 and l < 3:
        #             continue
        #         gv_norm = np.linalg.norm(self._gv[i, l])
        #         mean_free_path = (gv_norm * Angstrom * 1e6 /
        #                           (4 * np.pi * main_diagonal[l]))
        #         if mean_free_path > self._boundary_mfp:
        #             main_diagonal[l] = (
        #                 gv_norm / (4 * np.pi * self._boundary_mfp))
                    
        return main_diagonal
                        
    def _get_boundary_scattering(self, i):
        num_band = self._primitive.get_number_of_atoms() * 3
        g_boundary = np.zeros(num_band, dtype='double')
        for l in range(num_band):
            g_boundary[l] = (np.linalg.norm(self._gv[i, l]) * Angstrom * 1e6 /
                             (4 * np.pi * self._boundary_mfp))
        return g_boundary
        
    def _show_log_header(self, i):
        if self._log_level:
            gp = self._grid_points[i]
            print ("======================= Grid point %d (%d/%d) "
                   "=======================" %
                   (gp, i + 1, len(self._grid_points)))
            print "q-point: (%5.2f %5.2f %5.2f)" % tuple(self._qpoints[i])
            if self._boundary_mfp is not None:
                if self._boundary_mfp > 1000:
                    print ("Boundary mean free path (millimetre): %.3f" %
                           (self._boundary_mfp / 1000.0))
                else:
                    print ("Boundary mean free path (micrometre): %.5f" %
                       self._boundary_mfp)
            if self._is_isotope:
                print "Mass variance parameters:",
                print ("%5.2e " * len(self._mass_variances)) % tuple(
                    self._mass_variances)
                        
    def get_unique_tetrahedra_C(self, relative_address):
        import phonopy._spglib as spg
        num_expect=np.prod(self._mesh)*6
        unique_vertices = np.zeros((num_expect, 4), dtype="intc")
        number_of_unique=spg.unique_tetrahedra(unique_vertices,
                                               self._bz_grid_address,
                                               self._bz_map,
                                               relative_address,
                                               self._mesh)
        return unique_vertices[:number_of_unique]

    def get_unique_triagonal_C(self, relative_address):
        import phonopy._spglib as spg
        num_expect=np.prod(self._mesh)*6
        unique_vertices = np.zeros((num_expect, 3), dtype="intc")
        number_of_unique=spg.unique_tetrahedra(unique_vertices,
                                              self._bz_grid_address,
                                              self._bz_map,
                                              relative_address,
                                              self._mesh)
        return unique_vertices[:number_of_unique]

    def get_unique_triagonal_py(self, relative_address):
        assert np.count_nonzero(self._mesh>1) == 2
        # bzmesh = np.extract(self._mesh>1, self._mesh) * 2
        bzmesh = self._mesh * 2
        bz_grid_order = [1, bzmesh[0], bzmesh[0] * bzmesh[1]]
        num_grids = len(self._bz_grid_address)
        unique_vertices=[]
        for i in np.arange(num_grids):
            adrs = self._bz_grid_address[i] + relative_address
            bz_gp = np.dot(adrs % bzmesh, bz_grid_order)
            vgp = self._bz_map[bz_gp]
            for triag in vgp:
                if (triag==-1).any():
                    continue
                exists = False
                for element in unique_vertices:
                    if (triag==element).all():
                        exists = True
                        break
                if not exists:
                    unique_vertices.append(triag)
        return np.array(unique_vertices)

    def get_unique_tetrahedra_py(self, relative_address):
        bzmesh = self._mesh * 2
        bz_grid_order = [1, bzmesh[0], bzmesh[0] * bzmesh[1]]
        num_grids = len(self._bz_grid_address)
        unique_vertices=[]
        for i in np.arange(num_grids):
            adrs = self._bz_grid_address[i] + relative_address
            bz_gp = np.dot(adrs % bzmesh, bz_grid_order)
            vgp = self._bz_map[bz_gp]
            for tetra in vgp:
                if (tetra==-1).any():
                    continue
                exists = False
                for element in unique_vertices:
                    if (tetra==element).all():
                        exists = True
                        break
                if not exists:
                    unique_vertices.append(tetra)
        return np.array(unique_vertices)

    def print_kappa(self, isigma=None, itemp=None):
        temperatures = self._temperatures
        directions = ['xx', 'yy','zz','xy','yz','zx']
        if self._log_level == 0:
            directions = self._kappa
        if self._log_level==2:
            directions=directions
        if self._log_level == 0:
            directions=np.argmax(self._kappa, axis=-1)
        if self._log_level == 1:
            directions = ['xx', 'yy','zz']

        for i, sigma in enumerate(self._sigmas):
            if isigma is not None:
                if i != isigma:
                    continue
            kappa = self._kappa[i]
            band = ['band'+str(b+1) for b in range(kappa.shape[2])]
            print "----------- Thermal conductivity (W/m-k) for",
            print "sigma=%s -----------" % sigma
            for j, direction in enumerate(directions):
                print"*direction:%s" %direction
                print ("#%6s%10s" + " %9s" * len(band)) % (("T(K)","Total")+ tuple(band))
                for m,(t, k) in enumerate(zip(temperatures, kappa[...,j].sum(axis=0))):
                    if itemp is not None:
                        if m != itemp:
                            continue
                    print ("%7.1f%10.3f" + " %9.3f" * len(band)) % ((t,k.sum()) + tuple(k))
                print
        sys.stdout.flush()
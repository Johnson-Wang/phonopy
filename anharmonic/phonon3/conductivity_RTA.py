import numpy as np
import sys
from phonopy.phonon.group_velocity import get_group_velocity, degenerate_sets
from phonopy.units import THzToEv, EV, THz, Angstrom
from anharmonic.file_IO import write_kappa_to_hdf5, write_triplets, write_amplitude_to_hdf5_all, write_kappa_to_tecplot_BZ
from anharmonic.phonon3.triplets import get_pointgroup_operations
from anharmonic.phonon3.imag_self_energy import ImagSelfEnergy
from anharmonic.phonon3.conductivity import Conductivity

np.seterr(divide="ignore", invalid="ignore")
unit_to_WmK = ((THz * Angstrom) ** 2 / (Angstrom ** 3) * EV / THz /
               (2 * np.pi)) # 2pi comes from definition of lifetime.

class conductivity_RTA(Conductivity):
    def __init__(self,
                 interaction,
                 symmetry,
                 sigmas=[0.1],
                 asigma_step =1,
                 temperatures=None,
                 mesh_divisors=None,
                 coarse_mesh_shifts=None,
                 grid_points=None,
                 cutoff_lifetime=1e-4,  # in second
                 diff_kappa = 1e-3,  #  W/m-K
                 is_nu=False,  # is Normal or Umklapp
                 no_kappa_stars=False,
                 gv_delta_q=1e-4,  # finite difference for group velocity
                 log_level=0,
                 write_tecplot=False,
                 kappa_write_step=None,
                 is_thm=False,
                 filename=None):
        Conductivity.__init__(self,
                              interaction,
                              symmetry,
                              grid_points=grid_points,
                              temperatures=temperatures,
                              sigmas=sigmas,
                              mesh_divisors=mesh_divisors,
                              coarse_mesh_shifts=coarse_mesh_shifts,
                              no_kappa_stars=no_kappa_stars,
                              gv_delta_q=gv_delta_q,
                              log_level=log_level,
                              write_tecplot=write_tecplot)
        self._ise = ImagSelfEnergy(self._pp, is_nu, is_thm=is_thm, cutoff_lifetime= cutoff_lifetime)
        self._max_sigma_step=asigma_step
        self._is_asigma = False if asigma_step==1 else True
        self._sigma_iteration_step = 0
        self._is_nu=is_nu
        self._filename = filename
        if asigma_step > 1:
            if self._filename is not None:
                self._filename ="-adapt" + self._filename
            else:
                self._filename = "-adapt"
        self._cutoff_lifetime = cutoff_lifetime
        self._diff_kappa = diff_kappa
        self._is_converge = None
        if self._no_kappa_stars:
            self._kpoint_operations = np.eye(3,dtype="intc")
        else:
            self._kpoint_operations = get_pointgroup_operations(
                self._pp.get_point_group_operations())
        self._gamma_N = [None] * len(sigmas)
        self._gamma_U = [None]*len(sigmas)
        self._read_f = False
        self._read_gv = False
        self._cv = None
        self._wstep=kappa_write_step
        self._sum_num_kstar = 0
        self._scale_bar = 0
        if temperatures is not None:
            self._allocate_values()

    def get_mesh_divisors(self):
        return self._mesh_divisors

    def get_mesh_numbers(self):
        return self._mesh

    def get_group_velocities(self):
        return self._gv

    def get_mode_heat_capacities(self):
        return self._cv

    def get_frequencies(self):
        return self._frequencies

    def get_qpoints(self):
        qpoints = np.double([self._grid_address[gp].astype(float) / self._mesh
                             for gp in self._grid_points])
        return qpoints
            
    def get_grid_points(self):
        return self._grid_points

    def get_grid_weights(self):
        return self._grid_weights
            
    def set_temperatures(self, temperatures):
        self._temperatures = temperatures

    def get_temperatures(self):
        return self._temperatures

    def set_gamma(self, gamma):
        if not self._read_gamma:
            self._gamma = gamma
            self._read_gamma = True

    def set_group_velocity(self, gv):
        if not self._read_gv:
            self._gv = gv
            self._read_gv = True

    def set_frequency(self, frequencies):
        if not self._read_f:
            self._frequencies = frequencies
            for i, freq in enumerate(frequencies):
                self.set_degenerate_at_grid(i)
            self._read_f = True

    def get_kappa(self):
        return self._kappa

    def print_calculation_progress_header(self):
        self._scale_bar = 0 # used for marking the calculation progress
        if self._sigma_iteration_step==0:
            print "Calculation based on constant sigma value"
        else:
            print "Calculation for the %d iteration for sigmas" %self._sigma_iteration_step
        print "Calculation progress..."
        print "%9s%%"*10 %tuple(np.arange(1,11)*10)

    def calculate_kappa(self,write_gamma=False):
        self.set_pp_grid_points_all()
        for self._sigma_iteration_step in np.arange(self._max_sigma_step+1):
            #Avoid multiple writings
            if self._sigma_iteration_step != 0 :
                if self._pp.get_is_write_amplitude():
                    self._pp.set_is_write_amplitude(False)
                    self._pp.set_is_read_amplitude(True)
            # self._gamma_prev[:] = self._gamma[:]
            self._kappa_prev[:] = self._kappa.copy()
            self.print_calculation_progress_header()
            for i, grid_point in enumerate(self._grid_points):
                self._qpoint = (self._grid_address[grid_point].astype('double') /
                                self._mesh)
                self.print_log_information(i, grid_point)
                if not self._read_f:
                    self._frequencies[i] = self._get_phonon_c()
                    self.set_degenerate_at_grid(i)
                if (not self._read_gamma) or (self._max_sigma_step > 1):
                    if self._log_level > 0:
                        print "Number of triplets:",
                    self._ise.set_grid_point(grid_point, i)
                    if self._log_level > 0:
                        print len(self._pp.get_triplets_at_q()[0])
                        print "Calculating interaction..."
                        sys.stdout.flush()
                    log_level = self._log_level if self._sigma_iteration_step ==0 else 0
                    self._ise.run_interaction(log_level=log_level)
                    self._frequencies[i] = self._ise.get_phonon_at_grid_point()[0]
                    self.set_degenerate_at_grid(i)
                    if self._sigma_iteration_step == 0 and not self._read_gamma:
                        self._set_gamma_at_sigmas(i)
                    else:
                        self._set_gamma_at_sigmas(i, is_adapt_sigma=True)
                self._set_kappa_at_sigmas(i)
                if write_gamma:
                    self._write_gamma(i, grid_point)
            self._kappa /=  self._sum_num_kstar
            if self._pp.get_is_write_amplitude() and self._ise._interaction._amplitude_all is not None:
                write_amplitude_to_hdf5_all(self._ise._interaction._amplitude_all, self._mesh, is_nosym=self._pp.is_nosym())
            print
            if self._sigma_iteration_step==0:
                print "Thermal conductivity from constant sigma values is calculated to be"
            else:
                print "After %d iterations for sigma, the thermal conducitvity is recalculated to be" %(self._sigma_iteration_step)
            print_kappa(self._kappa, self._temperatures, self._sigmas)
            if self._wstep is not None:
                if self._sigma_iteration_step % self._wstep == 0:
                    self.write_kappa(filename="adapt-%d"%self._sigma_iteration_step)
            if self.check_sigma_convergence().all() and self._max_sigma_step>0 and self._sigma_iteration_step > 0:
                print "The iterations for adaptive sigma has converged"
                break

        if not self.check_sigma_convergence().all() and self._sigma_iteration_step>0:
            print "The iteration for sigma has ended because it has reached the maximum step"
            print "Note that the iterations has not fully converged"


    def print_log_information(self, i, grid_point):
        if self._log_level:
            print ("===================== Grid point %d (%d/%d) "
                   "=====================" %
                   (grid_point, i + 1, len(self._grid_points)))
            print "q-point: (%5.2f %5.2f %5.2f)" % tuple(self._qpoint)
            print "Lifetime cutoff (sec): %-10.3e" % self._cutoff_lifetime
            sys.stdout.flush()
        else:
            scale = 100./len(self._grid_points)
            num = np.rint(scale)
            self._scale_bar+=(scale-num)
            sys.stdout.write("="*(num+int(self._scale_bar)))
            self._scale_bar-=int(self._scale_bar)
            sys.stdout.flush()

    def check_sigma_convergence(self):
        nsigma = self._kappa.shape[0]
        ntemp = self._kappa.shape[2]
        max_kappa = np.sum(self._kappa, axis=(1,3)).max(axis=-1)
        diff_kappa=np.sum(np.abs(self._kappa - self._kappa_prev), axis=(1,3)) # sum over qpoints and bands
        for i in range(nsigma):
            for j in range(ntemp):
                diff_kappa[i,j] /= max_kappa[i,j]
        is_converge = (np.abs(diff_kappa) < self._diff_kappa)
        self._is_converge = is_converge

        return is_converge

    def _allocate_values(self):
        num_freqs = self._primitive.get_number_of_atoms() * 3
        self._kappa = np.zeros((len(self._sigmas),
                                len(self._grid_points),
                                len(self._temperatures),
                                num_freqs,
                                6), dtype='double')
        self._kappa_prev = np.zeros_like(self._kappa)
        if not self._read_gamma:
            self._gamma = np.zeros((len(self._sigmas),
                                    len(self._grid_points),
                                    len(self._temperatures),
                                    num_freqs), dtype='double')

            if self._is_nu:
                self._gamma_N = np.zeros_like(self._gamma)
                self._gamma_U = np.zeros_like(self._gamma)
        # self._gamma_prev = self._gamma.copy()
        self._is_converge = np.zeros((len(self._sigmas), len(self._temperatures), 6), dtype="bool")
        if not self._read_gv:
            self._gv = np.zeros((len(self._grid_points),
                                 num_freqs,
                                 3), dtype='double')
        self._cv = np.zeros((len(self._grid_points),
                             len(self._temperatures),
                             num_freqs), dtype='double')
        if not self._read_f:
            self._frequencies = np.zeros((len(self._grid_points),
                                          num_freqs), dtype='double')

    def _set_gamma_at_sigmas(self, i, is_adapt_sigma=False):
        freqs = self._frequencies[i]
        if is_adapt_sigma:
            bz_to_irr_map = self._irr_index_mapping[self._bz_to_pp_map]
            triplets_irre_indices = bz_to_irr_map[self._ise._grid_point_triplets]

        for j, sigma in enumerate(self._sigmas):
            if self._is_converge[j].all():
                continue
            if self._log_level > 0:
                print "Calculating Gamma with initial sigma=%s" % sigma
            if not is_adapt_sigma:
                self._ise.set_sigma(sigma)
            for k, t in enumerate(self._temperatures):
                if self._is_converge[j,k].all():
                    continue
                if is_adapt_sigma:
                    self._ise.set_adaptive_sigma(triplets_irre_indices, self._gamma[j,:,k])
                self._ise.set_temperature(t)
                self._ise.run()
                self._gamma[j, i, k] = self._ise.get_imag_self_energy()
                if self._is_nu:
                    self._gamma_N[j,i,k]=self._ise.get_imag_self_energy_N()
                    self._gamma_U[j,i,k]=self._ise.get_imag_self_energy_U()

    def get_gamma(self):
        return np.where(self._gamma< 0.5 / self._cutoff_lifetime / THz, -1, self._gamma)

    def write_kappa(self, filename):
        temperatures = self.get_temperatures()
        for i, sigma in enumerate(self._sigmas):
            kappa = self._kappa[i]
            write_kappa_to_hdf5(self._gamma[i],
                                temperatures,
                                self.get_mesh_numbers(),
                                frequency=self.get_frequencies(),
                                group_velocity=self.get_group_velocities(),
                                heat_capacity=self.get_mode_heat_capacities(),
                                kappa=kappa,
                                qpoint=self.get_qpoints(),
                                weight=self.get_grid_weights(),
                                mesh_divisors=self.get_mesh_divisors(),
                                sigma=sigma,
                                filename=filename,
                                gnu=(self._gamma_N[i],self._gamma_U[i]))
            if self._write_tecplot:
                for j,temp in enumerate(temperatures):
                    write_kappa_to_tecplot_BZ(np.where(self._gamma[i,:,j]>1e-8, self._gamma[i,:,j],0),
                                           temp,
                                           self.get_mesh_numbers(),
                                           bz_q_address=self._bz_grid_address / self.get_mesh_numbers().astype(float),
                                           tetrahedrdons=self._unique_vertices,
                                           bz_to_pp_mapping=self._bz_to_pp_map,
                                           rec_lattice=np.linalg.inv(self._primitive.get_cell()),
                                           spg_indices_mapping=self._irr_index_mapping,
                                           spg_rotation_mapping=self._rot_mappings,
                                           frequency=self.get_frequencies(),
                                           group_velocity=self.get_group_velocities(),
                                           heat_capacity=self.get_mode_heat_capacities()[:,j],
                                           kappa=kappa[:,j],
                                           weight=self.get_grid_weights(),
                                           sigma=sigma,
                                           filename=filename+"-bz")


    def _set_kappa_at_sigmas(self, i):
        freqs = self._frequencies[i]
        
        # Group velocity [num_freqs, 3]
        if not self._read_gv:
            gv = get_group_velocity(
                self._qpoint,
                self._dm,
                self._symmetry,
                q_length=self._gv_delta_q,
                frequency_factor_to_THz=self._frequency_factor_to_THz)
            self._gv[i] = gv
        # self._gv[i] = self._get_degenerate_gv(i)

        
        # Heat capacity [num_temps, num_freqs]
        cv = self._get_cv(freqs)
        self._cv[i] = cv
        num_kstar_counted = False
        try:
            import anharmonic._phono3py as phono3c
            rec_lat = np.linalg.inv(self._primitive.get_cell())
            kpt_rotations_at_q = self._get_rotations_for_star(i)
            if self._sigma_iteration_step == 0:
                self._sum_num_kstar += len(kpt_rotations_at_q)
                num_kstar_counted = True
            deg = degenerate_sets(self._frequencies[i])
            degeneracy = np.zeros(len(self._frequencies[i]), dtype="intc")
            for ele in deg:
                for sub_ele in ele:
                    degeneracy[sub_ele]=ele[0]
            for j, sigma in enumerate(self._sigmas):
                kappa_at_qs = np.zeros_like(self._kappa[j,i])
                mfp = np.zeros((len(self._temperatures), self._frequencies.shape[1], 3), dtype="double")
                for t, temp in enumerate(self._temperatures):
                    for k in range(3):
                        mfp[t,:,k] = np.where(self._gamma[j,i,t] > 0.5 / self._cutoff_lifetime / THz,
                                          self._gv[i,:,k] /  self._gamma[j,i,t],
                                          0)
                phono3c.thermal_conductivity_at_grid(kappa_at_qs,
                                                     kpt_rotations_at_q.astype("intc").copy(),
                                                     rec_lat.copy(),
                                                     cv.copy(),
                                                     mfp.copy(),
                                                     self._gv[i].copy(),
                                                     degeneracy.copy())
                self._kappa[j,i] = kappa_at_qs / 2 * self._conversion_factor

        except ImportError:
            print "Warning: kappa calculation (the final step) went wrong in the C implementation. Changing to python instead..."
            # Outer product of group velocities (v x v) [num_k*, num_freqs, 3, 3]
            gv_by_gv_tensor = self._get_gv_by_gv(i)
            if self._sigma_iteration_step == 0 and num_kstar_counted == False:
                self._sum_num_kstar += len(gv_by_gv_tensor)

            # Sum all vxv at k*
            gv_sum2 = np.zeros((6, len(freqs)), dtype='double')
            for j, vxv in enumerate(
                ([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])):
                gv_sum2[j] = gv_by_gv_tensor[:, :, vxv[0], vxv[1]].sum(axis=0)

            # Kappa
            for j, sigma in enumerate(self._sigmas):
                for k, l in list(np.ndindex(len(self._temperatures), len(freqs))):
                    if self._gamma[j, i, k, l] < 0.5 / self._cutoff_lifetime / THz:
                        continue
                    self._kappa[j, i, k, l, :] = (
                        gv_sum2[:, l] * cv[k, l] / (self._gamma[j, i, k, l] * 2) *
                        self._conversion_factor)

    def _get_degenerate_gv(self, i):
        deg_sets = degenerate_sets(self._frequencies[i])
        gv = self._gv[i]
        gvs = np.zeros_like(gv)
        for deg in deg_sets:
            gv_ave = gv[deg].sum(axis=0) / len(deg)
            for j in deg:
                gvs[j] = gv_ave
        return gvs

    def set_degenerate_at_grid(self, i):
        deg_sets = degenerate_sets(self._frequencies[i])
        for ele in deg_sets:
            for sub_ele in ele:
                self._degeneracies[i, sub_ele]=ele[0]

    def _get_rotations_for_star(self, i):
        if self._no_kappa_stars:
            rotations = np.array([np.eye(3, dtype=int)])
        else:
            grid_point = self._grid_points[i]
            rotations = self._kpoint_operations[self._rot_mappings[np.where(self._mappings==grid_point)]]
            if self._grid_weights is not None:
                assert len(rotations) == self._grid_weights[i], \
                    "Num rotations %d, weight %d" % (len(rotations), self._grid_weights[i])
        return rotations

    def _get_phonon_c(self):
        import anharmonic._phono3py as phono3c

        dm = self._dm
        svecs, multiplicity = dm.get_shortest_vectors()
        masses = np.double(dm.get_primitive().get_masses())
        rec_lattice = np.double(
            np.linalg.inv(dm.get_primitive().get_cell())).copy()
        if dm.is_nac():
            born = dm.get_born_effective_charges()
            nac_factor = dm.get_nac_factor()
            dielectric = dm.get_dielectric_constant()
        else:
            born = None
            nac_factor = 0
            dielectric = None
        uplo = self._pp.get_lapack_zheev_uplo()
        num_freqs = len(masses) * 3
        frequencies = np.zeros(num_freqs, dtype='double')
        eigenvectors = np.zeros((num_freqs, num_freqs), dtype='complex128')

        phono3c.phonon(frequencies,
                       eigenvectors,
                       np.double(self._qpoint),
                       dm.get_force_constants(),
                       svecs,
                       multiplicity,
                       masses,
                       dm.get_primitive_to_supercell_map(),
                       dm.get_supercell_to_primitive_map(),
                       self._frequency_factor_to_THz,
                       born,
                       dielectric,
                       rec_lattice,
                       None,
                       nac_factor,
                       uplo)
        # dm.set_dynamical_matrix(self._qpoint)
        # dynmat = dm.get_dynamical_matrix()
        # eigvals = np.linalg.eigvalsh(dynmat).real
        # frequencies = (np.sqrt(np.abs(eigvals)) * np.sign(eigvals) *
        #                self._frequency_factor_to_THz)

        return frequencies

    def _show_log(self,
                  grid_point,
                  frequencies,
                  group_velocity,
                  rotations,
                  rotations_cartesian):
        print "----- Partial kappa at grid address %d -----" % grid_point
        print "Frequency, projected group velocity (x, y, z), norm at k-stars",
        if self._gv_delta_q is None:
            print
        else:
            print " (dq=%3.1e)" % self._gv_delta_q
        q = self._grid_address[grid_point].astype(float) / self._mesh
        for i, (rot, rot_c) in enumerate(zip(rotations, rotations_cartesian)):
            q_rot = np.dot(rot, q)
            q_rot -= np.rint(q_rot)
            print " k*%-2d (%5.2f %5.2f %5.2f)" % ((i + 1,) + tuple(q_rot))
            for f, v in zip(frequencies, np.dot(rot_c, group_velocity.T).T):
                print "%8.3f   (%8.3f %8.3f %8.3f) %8.3f" % (
                    f, v[0], v[1], v[2], np.linalg.norm(v))

        print
    def print_kappa(self):
        temperatures = self.get_temperatures()
        if self._log_level==2:
            directions=['xx', 'yy','zz','xy','yz','zx']
        else:
            directions=['xx']
        for i, sigma in enumerate(self._sigmas):
            kappa = self._kappa[i]
            band = ['band'+str(b+1) for b in range(kappa.shape[2])]
            print "----------- Thermal conductivity (W/m-k) for",
            print "sigma=%s -----------" % sigma
            for j, direction in enumerate(directions):
                print"*direction:%s" %direction
                print ("#%6s%10s" + " %9s" * len(band)) % (("T(K)","Total")+ tuple(band))
                for t, k in zip(temperatures, kappa[...,j].sum(axis=0)):
                    print ("%7.1f%10.3f" + " %9.3f" * len(band)) % ((t,k.sum()) + tuple(k))
                print
        sys.stdout.flush()

    def _write_gamma(self, i, grid_point):
        for j, sigma in enumerate(self._sigmas):
            write_kappa_to_hdf5(
                self._gamma[j, i],
                self._temperatures,
                self._mesh,
                frequency=self._frequencies[i],
                group_velocity=self._gv[i],
                heat_capacity=self._cv[i],
                kappa=self._kappa[j, i],
                mesh_divisors=self._mesh_divisors,
                grid_point=grid_point,
                sigma=sigma,
                filename=self._filename)

    def _write_triplets(self, grid_point):
        triplets, weights = self._pp.get_triplets_at_q()
        grid_address = self._pp.get_grid_address()
        write_triplets(triplets,
                       weights,
                       self._mesh,
                       grid_address,
                       grid_point=grid_point,
                       filename=self._filename)


def print_kappa(kappas, temperatures, sigmas, log_level=1):
    if log_level==2:
        directions=['xx', 'yy','zz','xy','yz','zx']
    elif log_level == 1:
        directions = ['xx', 'yy', 'zz']
    else:
        directions=['xx']
    for i, sigma in enumerate(sigmas):
        kappa = kappas[i]
        band = ['band'+str(b+1) for b in range(kappa.shape[2])]
        print "----------- Thermal conductivity (W/m-k) for",
        print "sigma=%s -----------" % sigma
        for j, direction in enumerate(directions):
            print"*direction:%s" %direction
            print ("#%6s%10s" + " %9s" * len(band)) % (("T(K)","Total")+ tuple(band))
            for t, k in zip(temperatures, kappa[...,j].sum(axis=0)):
                print ("%7.1f%10.3f" + " %9.3f" * len(band)) % ((t,k.sum()) + tuple(k))
            print
    sys.stdout.flush()
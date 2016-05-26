import numpy as np
import sys
from anharmonic.phonon3.imag_self_energy import occupation
from anharmonic.phonon3.collision import Collision
from phonopy.phonon.group_velocity import degenerate_sets
from phonopy.harmonic.force_constants import similarity_transformation
from anharmonic.phonon3.triplets import get_rotations_for_star
from anharmonic.phonon3.conductivity import Conductivity

np.seterr(divide="ignore")
class conductivity_ITE(Conductivity):
    def __init__(self,
                 interaction,
                 symmetry=None,
                 sigmas=[0.2],
                 grid_points=None,
                 temperatures=None,
                 max_ite= None,
                 adaptive_sigma_step = 0,
                 no_kappa_stars=False,
                 diff_kappa = 1e-4, # relative
                 gv_delta_q = None,
                 mass_variances = None,
                 length = None,
                 write_gamma=False,
                 read_gamma=False,
                 read_col = False,
                 write_col = False,
                 log_level=1,
                 is_thm=False,
                 filename="ite",
                 lang="C",
                 is_precondition = True,
                 write_tecplot=False):
        self._pp = interaction
        self._log_level = log_level
        self._read_gamma = read_gamma
        self._write_gamma = write_gamma
        self._lang=lang
        self._is_adaptive_sigma = False if adaptive_sigma_step == 0 else True
        self._max_asigma_step = adaptive_sigma_step
        Conductivity.__init__(self,
                              interaction,
                              symmetry,
                              grid_points=grid_points,
                              temperatures=temperatures,
                              sigmas=sigmas,
                              no_kappa_stars=no_kappa_stars,
                              gv_delta_q=gv_delta_q,
                              log_level=log_level,
                              write_tecplot=write_tecplot)
        self._scale_bar = 0
        self._itemp = None # index of temperature
        self._temp  = None
        self._is_precondition = is_precondition
        self._diff_kappa = diff_kappa
        self._collision = Collision(interaction,
                                   sigmas=self._sigmas,
                                   temperatures=self._temperatures,
                                   is_adaptive_sigma=self._is_adaptive_sigma,
                                   mass_variances=mass_variances,
                                   length=length,
                                   write=write_col,
                                   read=read_col,
                                   is_tetrahedron_method=is_thm,
                                   cutoff_frequency=self._cutoff_frequency)
        self._is_write_col = write_col
        self._is_read_col = read_col
        self.ite_init()
        self._ite_num = max_ite
        self._ite_step = 0
        if filename is not None:
            self._filename = filename
        else:
            self._filename = "ite"

    def __iter__(self):
        return self

    def next(self):
        if (self._is_converge ==True).all():
            print "All calculations have converged"
            raise StopIteration
        elif self._ite_step == self._ite_num:
            non_converge=np.where(self._is_converge ==False)
            sigma_nc=self._sigmas[non_converge[0]] # sigma value for the non-converged cases
            temp_nc=self._temperatures[non_converge[1]]
            print "Calculations for"
            for s, t in zip(sigma_nc, temp_nc):
                print "sigma=%s, temperature=%f" %(s,t)
            print "have not converged, but it has reached the maximum convergence steps "
            raise StopIteration
        for i in range(len(self._sigmas)):
            self.set_sigma(i)
            for j in range(len(self._temperatures)):
                self.set_temperature(j)
                self.set_collision()
                self.run_at_sigma_and_temp()
        self.check_convergence()
        self.renew()
        self._ite_step += 1
        return self

    def set_is_write_col(self, is_write_col=False):
        self._write_col = is_write_col

    def set_is_read_col(self, is_read_col=False):
        self._read_col = is_read_col

    def run_at_sigma_and_temp(self): # sigma and temperature
        if self._log_level:
            print "######Perturbation flow for the next iterative step at sigma %s, t=%f#######" \
                  %(self._sigma, self._temp)
            print "Calculation progress..."
        self.print_calculation_progress_header()
        if not self._is_converge[self._isigma, self._itemp]:
            for i, grid_point in enumerate(self._grid_points):
                self._collision.calculate_collision(grid_point=grid_point)
                #calculating scattering rate
                self.perturbation_next_s_t_g(self._isigma, self._itemp, i) # sigma, temperature, grid
                self.print_calculation_progress(i)
            self._collision.write_collision_all(log_level=self._log_level)

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

    def ite_init(self):
        self.smrt()
        num_sigma = len(self._sigmas)
        num_temp = len(self._temperatures)
        self._F0 = self._F
        self._F_prev = self._F0.copy()
        self._F = np.zeros_like(self._F0)
        self._R = np.zeros_like(self._F0) # residual
        self._is_converge=np.zeros((num_sigma, num_temp), dtype="bool")

    def renew(self):
        for s, sigma in enumerate(self._sigmas):
            for t, temp in enumerate(self._temperatures):
                if not self._is_converge[s, t]:
                    self._F_prev[s,:,t] =  self._F[s,:,t]

    def check_convergence(self):
        for s, sigma in enumerate(self._sigmas):
            for t, temp in enumerate(self._temperatures):
                dkappa_max = np.abs(self.get_kappa_residual_at_s_t(s, t)).max()
                kappa = self.get_kappa()[s, :, t]
                dkappa_max /= kappa.sum(axis=(0,1)).max()
                print "Relative residual kappa for sigma=%s, T=%.2f K is %10.5e" % (sigma, temp, dkappa_max)
                is_converge=(dkappa_max < self._diff_kappa)
                self._is_converge[s,t]= is_converge
                if is_converge:
                    if self._log_level:
                        print "Calculation for sigma=%s, temp=%f has converged!"%(str(sigma), temp)

    def smrt(self):
        num_sigma = len(self._sigmas)
        num_temp = len(self._temperatures)
        num_grid = len(self._grid_points)
        num_band = self._frequencies.shape[-1]
        self._F= np.zeros((num_sigma, num_grid, num_temp, num_band, 3), dtype="double")
        self._b = np.zeros((num_grid,num_temp,num_band,3), dtype="double")
        self._kappa = np.zeros((num_sigma, num_grid, num_temp, num_band, 6), dtype="double")
        self._rkappa = np.ones((num_sigma, num_temp, 6), dtype="double") # relative kappa residual
        self._gv = np.zeros((len(self._grid_points),
                             num_band,
                             3), dtype='double')
        self._collision_out = np.zeros((num_sigma, num_grid, num_temp, num_band), dtype="double")
        self._gamma = np.zeros((num_sigma, num_grid, num_temp, num_band), dtype="double")

        # self.run_smrt_no_sigma_adaption()
        # if self._is_adaptive_sigma:
        self.run_smrt_sigma_adaption()
        if self._is_read_col:
            self._pp.release_amplitude_all()

    def run_smrt_sigma_adaption(self):
        asigma_step = 0
        while asigma_step <= self._max_asigma_step:
            if self._collision.get_read_collision()  and self._is_adaptive_sigma:
                self._collision.set_write_collision(True)
            if self._collision.get_read_collision() and self._is_adaptive_sigma:
                self._collision.set_write_collision(True)

            if asigma_step > 1:
                if asigma_step > 2:
                    self._gamma_ppp = self._gamma_pp.copy()
                self._gamma_pp = self._gamma_prev.copy()
            self._gamma_prev = self._collision._gamma_all.copy()
            for s in range(len(self._sigmas)):
                if (self._rkappa[s] < self._diff_kappa).all():
                    continue
                self.set_sigma(s)
                for t in range(len(self._temperatures)):
                    if (self._rkappa[s,t] < self._diff_kappa).all():
                        continue
                    self.set_temperature(t)
                    self.set_collision()
                    if self._log_level:
                        if asigma_step == 0:
                            print "######Kappa calculation within SMRT at constant sigma=%s, temp=%.2f #######" %(self._sigma, self._temp)
                        else:
                            print "######Kappa calculation within SMRT with sigma-adaption at temp=%.2f, (%d/%d) #######" %\
                                  (self._temp, asigma_step, self._max_asigma_step)
                    self.print_calculation_progress_header()
                    for g, grid_point in enumerate(self._grid_points):
                        self._collision.set_grid(grid_point)
                        self._collision.set_phonons_triplets()
                        self._collision.set_grid_points_occupation()
                        self._collision.reduce_triplets()
                        if asigma_step > 0:
                            if asigma_step > 2:
                                self._collision.set_asigma(self._gamma_pp[s, :, t], self._gamma_ppp[s, :, t])
                            else:
                                self._collision.set_asigma()
                        self._collision.set_integration_weights()
                        self._collision.run_interaction_at_grid_point(self._collision._g_skip_reduced)
                        self._collision.run()
                        self.assign_perturbation_at_grid_point(s, g, t)
                        self.print_calculation_progress(g)
                    if asigma_step == 0 and (not self._is_read_col):
                        self._pp.write_amplitude_all()
                    if not self._collision.get_is_dispersed():
                        self._collision.write_collision_all(log_level=self._log_level, is_adaptive_sigma=self._is_adaptive_sigma)
                self.set_kappa_at_sigma(s)
            self.print_kappa()
            asigma_step += 1
            if (self._rkappa < self._diff_kappa).all():
                print "The sigma iteration process has converged."
                break
        self._collision.set_is_on_iteration()
        if self._collision.get_write_collision():
            self._collision.set_write_collision(False)
            self._collision.set_read_collision(True)

    # def smrt(self): #single-mode relaxation time approximation
    #     num_sigma = len(self._sigmas)
    #     num_temp = len(self._temperatures)
    #     num_grid = len(self._grid_points)
    #     num_band = self._frequencies.shape[-1]
    #     self._F= np.zeros((num_sigma, num_grid, num_temp, num_band, 3), dtype="double")
    #     self._b = np.zeros((num_grid,num_temp,num_band,3), dtype="double")
    #     self._kappa = np.zeros((num_sigma, num_grid, num_temp, num_band, 6), dtype="double")
    #     self._gv = np.zeros((len(self._grid_points),
    #                          num_band,
    #                          3), dtype='double')
    #     self._collision_out = np.zeros((num_sigma, num_grid, num_temp, num_band), dtype="double")
    #     self._gamma = np.zeros((num_sigma, num_grid, num_temp, num_band), dtype="double")
    #     for s in range(len(self._sigmas)):
    #         self.set_sigma(s)
    #         for t in range(len(self._temperatures)):
    #             self.set_temperature(t)
    #             self.set_collision()
    #             if self._log_level:
    #                 print "######Kappa calculation within SMRT at sigma=%s, t=%f#######"  %(self._sigma, self._temp)
    #             self.print_calculation_progress_header()
    #             for g, grid_point in enumerate(self._grid_points):
    #                 self._collision.calculate_collision(grid_point=grid_point)
    #                 n = self._collision.get_occupation()[grid_point, t]
    #                 is_pass = self._frequencies[g] < self._cutoff_frequency
    #                 collision_out = self._collision.get_collision_out()
    #                 out_reverse=np.where(is_pass,0, 1/collision_out)
    #                 freqs = self._frequencies[g]
    #                 self._set_gv(g)
    #                 nn1 = n * (n + 1)
    #                 fnn1 = freqs * nn1
    #                 self._b[g,t] = fnn1[:,np.newaxis] * self._gv[g] / self._temp ** 2
    #                 self._F[s,g,t]= out_reverse[:,np.newaxis] * self._b[g,t]
    #                 self._collision_out[s,g,t] = self._collision.get_collision_out()
    #                 self._gamma[s,g,t] = self._collision.get_collision_out() * np.where(is_pass, 0, 1 / nn1) / 2.
    #                 self.print_calculation_progress(g)
    #         self.set_kappa_at_sigma(s)
    #     print "Within SMRT, the thermal conductivities are recalculated to be (W/mK)"
    #     self.print_kappa()
    #     self._collision.write_collision_all(log_level=self._log_level)
    #     if self._collision.get_write_collision():
    #         self._collision.set_write_collision(False)
    #         self._collision.set_read_collision(True)

    def print_calculation_progress_header(self):
        self._scale_bar = 0 # used for marking the calculation progress
        if self._log_level:
            print "%9s%%"*10 %tuple(np.arange(1,11)*10)
            sys.stdout.flush()

    def print_calculation_progress(self, i):
        if self._log_level>1:
            print ("===================== Grid point %d (%d/%d) "
                   "=====================" %
                   (self._grid_points[i], i + 1, len(self._grid_points)))
            print "q-point: (%5.2f %5.2f %5.2f)" % tuple(self._qpoints[i])
            sys.stdout.flush()
        else:
            scale = 100./len(self._grid_points)
            num = np.rint(scale)
            self._scale_bar+=(scale-num)
            sys.stdout.write("="*(num+int(self._scale_bar)))
            self._scale_bar-=int(self._scale_bar)
            sys.stdout.flush()
        if i == len(self._grid_points) - 1:
            print

    def set_collision(self):
        self._collision.set_temperature(temperature=self._temp)
        self._collision.set_sigma(self._sigma)
        self._collision.set_grids(self._grid_points)
        if self._collision.get_read_collision() and not self._collision.get_is_dispersed():
            self._collision.read_collision_all(log_level=self._log_level, is_adaptive_sigma=self._is_adaptive_sigma)

    def get_mode_heat_capacities(self):
        cv = []
        for i, g in enumerate(self._grid_points):
            f = self._frequencies[i]
            cv.append(self._get_cv(f))
        return np.array(cv, dtype="double")

    def get_total_rotation(self):
        triplet1 = self._collision._grid_point_triplets[:,1]
        triplet1_pp = self._bz_to_pp_map[triplet1]
        num_rots = len(self._collision._kpg_at_q_index)
        inv_rot = self._kpoint_group_inv_map[self._rot_mappings[triplet1_pp]]
        r1dotr2_inv = self._kpoint_group_sum_map[self._collision._kpg_at_q_index][:,inv_rot]
        a1 = self._kpoint_operations[r1dotr2_inv].sum(axis=0)
        self._rot1_sums = a1 / float(num_rots)

    def perturbation_next_s_t_g(self,isigma, itemp, igrid):
        self.get_total_rotation()
        if self._lang=="C":
            self.perturbation_next_c(isigma, itemp, igrid)
        else:
            self.perturbation_next_py(isigma,itemp, igrid)
        self._R[isigma,igrid,itemp] = self._F[isigma,igrid,itemp] - self._F_prev[isigma,igrid,itemp]

    def perturbation_next_c(self,isigma,itemp, igrid):
        import anharmonic._phono3py as phono3c
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        spg_mapping_index = np.unique(self._mappings, return_inverse=True)[1]
        branch_len = self._collision._collision_in.shape[-2]
        PF_sum = np.zeros((branch_len, 3), dtype="double")
        triplets1 = self._bz_to_pp_map[self._collision._grid_point_triplets[:,1]]
        F_prev = self._F_prev[isigma, spg_mapping_index[triplets1], itemp]

        phono3c.phonon_multiply_dmatrix_gbb_dvector_gb3(PF_sum,
                                                        self._collision._collision_in.copy(),
                                                        F_prev.copy(),
                                                        np.intc(self._collision._triplet_weights).copy(),
                                                        self._rot1_sums,
                                                        rec_lat)
        rout = np.where(self._collision._collision_out > 0, 1 / self._collision._collision_out, 0)
        self._F[isigma,igrid,itemp] = self._F0[isigma,igrid,itemp] - PF_sum * rout[:,np.newaxis]

    def assign_perturbation_at_grid_point(self, isigma, igrid, itemp):
        grid_point = self._grid_points[igrid]
        n = self._collision.get_occupation()[grid_point, itemp]
        is_pass = self._frequencies[igrid] < self._cutoff_frequency
        collision_out = self._collision.get_collision_out()
        out_reverse=np.where(is_pass,0, 1/collision_out)
        freqs = self._frequencies[igrid]
        self._set_gv(igrid)
        nn1 = n * (n + 1)
        fnn1 = freqs * nn1
        self._b[igrid,itemp] = fnn1[:,np.newaxis] * self._gv[igrid] / self._temp ** 2
        self._F[isigma,igrid,itemp]= out_reverse[:,np.newaxis] * self._b[igrid,itemp]
        self._collision_out[isigma,igrid,itemp] = self._collision.get_collision_out()
        self._gamma[isigma,igrid,itemp] = self._collision.get_collision_out() * np.where(is_pass, 0, 1 / nn1) / 2.

    # def perturbation_next_py(self,isigma,itemp, igrid):
    #     triplet1 = self._scarate._grid_point_triplets[:,1]
    #     # num_triplets = len(self._scarate._triplet_weights)
    #     num_rots = len(self._scarate._kpg_at_q_index)
    #     rec_lat = np.linalg.inv(self._primitive.get_cell())
    #     inv_rot = self._kpoint_group_inv_map[self._rot_mappings[triplet1]]
    #     inv_rot_sum = np.array([self._scarate._inv_rot_sum * weight for weight in  self._scarate._triplet_weights]) / float(num_rots)
    #     rot1_sum = np.einsum("lij, ljk -> lik", inv_rot_sum, self._kpoint_operations[inv_rot])
    #     self._rot1_sums = np.einsum("ij, ljk, km -> lim", rec_lat, rot1_sum, np.linalg.inv(rec_lat))
    #
    #     spg_mapping_index = np.unique(self._mappings, return_inverse=True)[1]
    #     if not self._is_converge[isigma, itemp]:
    #         # ncpos=np.where(self._is_converge[isigma,igrid, itemp]==False)#non_convergence positions
    #         # summation=np.zeros((len(zip(*ncpos)), 3), dtype="double")
    #         for j, triplet in enumerate(self._scarate._grid_point_triplets):
    #             g1=triplet[1] # q'
    #             j1 = spg_mapping_index[g1]
    #             # for n,(t, b0) in enumerate(zip(*ncpos)):
    #             F1_prev_sum = np.dot(self._rot1_sums[j], self._F_prev[isigma,j1,itemp].T)
    #             summation[n] += (self._scarate._scatt_in[j,itemp] * F1_prev_sum).sum(axis=-1)
    #
    #         for n,(t,b0) in enumerate(zip(*ncpos)):
    #             self._F[isigma,igrid,t,b0] = self._F0[isigma,igrid,t,b0] - summation[n] *\
    #                                  np.where(self._scarate._scatt_out[t,b0]>0, 1 / self._scarate._scatt_out[t,b0], 0)

    def perturbation_next_py(self,isigma,itemp, igrid):
        if not self._is_converge[isigma,igrid]:
            scatt = self._collision.get_collision_out()
            ncpos=np.where(self._is_converge[isigma,igrid, itemp]==False)[0]#non_convergence positions
            for b0 in ncpos:
                freq = self._frequencies[igrid,b0]
                occu = occupation(freq, self._temperatures[itemp])
                numerator = self._gv[igrid,b0] * freq * occu * (occu + 1)
                denomenator = scatt[b0] * self._temperatures[itemp] ** 2
                self._F[isigma,igrid,itemp,b0] = np.where(scatt[b0]>0, numerator / denomenator, 0)


    def set_kappa(self):
        for i,s in enumerate(self._sigmas):
            self.set_kappa_at_sigma(i)

    def set_equivalent_gamma_at_sigma(self,i):
        gv_norm=np.sum(self._gv**2, axis=-1)
        for j, t in enumerate(self._temperatures):
            FdotvT2 = 2 * np.sum(self._F[i,:,j] * self._gv, axis=-1) * t ** 2
            self._gamma[i,:,j] = self._frequencies * gv_norm * np.where(np.abs(FdotvT2)==0, 0, 1 / FdotvT2)

    def set_kappa_at_sigma(self,s):
        if self._lang == "C":
            self.set_kappa_at_s_c(s)
        else:
            self.set_kappa_at_s_py(s)
        self._kappa[s] /= np.prod(self._mesh)

    def set_write_amplitude(self, write_amplitude):
        self._write_amplitude = write_amplitude

    def set_read_amplitude(self, read_amplitude):
        self._read_amplitude = read_amplitude

    def set_kappa_at_s_c(self, s):
        import anharmonic._phono3py as phono3c
        kappa = np.zeros_like(self._kappa[s])
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        for t, temp in enumerate(self._temperatures):
            gouterm_temp = np.zeros((self._frequencies.shape[0], self._frequencies.shape[1], 6), dtype="double")
            phono3c.phonon_gb33_multiply_dvector_gb3_dvector_gb3(gouterm_temp,
                                                                 self._b[:,t].copy(),
                                                                 self._F[s,:,t].copy(),
                                                                 np.intc(self._irr_index_mapping).copy(),
                                                                 np.intc(self._kpoint_operations[self._rot_mappings]).copy(),
                                                                 rec_lat.copy())
            kappa[:,t] = gouterm_temp * temp ** 2
        self._kappa[s] = kappa * self._kappa_factor

    def get_kappa_residual_at_s_t(self, s, t):
        import anharmonic._phono3py as phono3c
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        dkappa = np.zeros((self._frequencies.shape[0], self._frequencies.shape[1], 6), dtype="double")
        phono3c.phonon_gb33_multiply_dvector_gb3_dvector_gb3(dkappa,
                                                             self._b[:,t].copy(),
                                                             self._R[s,:,t].copy(),
                                                             np.intc(self._irr_index_mapping).copy(),
                                                             np.intc(self._kpoint_operations[self._rot_mappings]).copy(),
                                                             rec_lat.copy())
        dkappa = np.sum(self._grid_weights[:, np.newaxis, np.newaxis] * np.abs(dkappa), axis=(0,1))
        return dkappa * self._temperatures[t] ** 2 * self._kappa_factor / np.prod(self._mesh)

    def set_kappa_at_s_py(self, s):
        cv = self._br._cv
        for t,temp in enumerate(self._temperatures):
            for i, grid in enumerate(self._grid_points):
                gvbysr_tensor = self.get_gv_by_sr_at_stars(s,i,t) # get the product of gv and scattering rate
                n = occupation(self._frequencies[i], self._temperatures[t])
                # Sum all vxv at k*
                gv_sr_sum = np.zeros((6, self._frequencies.shape[1]), dtype='double')
                for j, vxv in enumerate(
                    ([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])):
                    gv_sr_sum[j] = gvbysr_tensor[:, :, vxv[0], vxv[1]].sum(axis=0)
                # Kappa
                for j in range(self._frequencies.shape[1]):
                    if self._frequencies[i,j] < self._cutfr:
                        continue
                    self._kappa[s,i,t,j,:] = cv[i,t,j] * gv_sr_sum[:, j] * temp ** 2 / self._frequencies[i,j]  * self._kappa_factor

    def get_gv_by_sr_at_stars(self, s,i,t):
        deg_sets = degenerate_sets(self._frequencies[i])
        grid_point = self._grid_points[i]
        orig_address = self._grid_address[grid_point]
        rotations=get_rotations_for_star(orig_address, self._mesh, self._kpoint_operations, no_sym=self._no_kappa_stars)
        # self._get_group_veclocities_at_star(i, gv)
        gv_by_F_tensor = []
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        rotations_cartesian = [similarity_transformation(rec_lat, r)
                               for r in rotations]
        for rot_c in rotations_cartesian:
            gvs_rot = np.dot(rot_c, self._gv[i].T).T
            F_rot = np.dot(rot_c, self._F[s,i,t].T).T

            # Take average of group veclocities of degenerate phonon modes
            # and then calculate gv x gv to preserve symmetry
            gvs = np.zeros_like(gvs_rot)
            Fs = np.zeros_like(F_rot)
            for deg in deg_sets:
                gv_ave = gvs_rot[deg].sum(axis=0) / len(deg)
                F_ave = F_rot[deg].sum(axis=0)/len(deg)
                for j in deg:
                    gvs[j] = gv_ave
                    Fs[j]=F_ave
            gv_by_F_tensor.append([np.outer(gvs[b], Fs[b]) for b in range(len(gvs))])

        return np.array(gv_by_F_tensor)

    def get_gv_by_sr0_at_stars(self, s,i,t):
        deg_sets = degenerate_sets(self._frequencies[i])
        grid_point = self._grid_points[i]
        orig_address = self._grid_address[grid_point]
        rotations=get_rotations_for_star(orig_address, self._mesh, self._kpoint_operations)
        # self._get_group_veclocities_at_star(i, gv)
        gv_by_F_tensor = []
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        rotations_cartesian = [similarity_transformation(rec_lat, r)
                               for r in rotations]
        for rot_c in rotations_cartesian:
            gvs_rot = np.dot(rot_c, self._gv[i].T).T
            F_rot = np.dot(rot_c, self._F0[s,i,t].T).T

            # Take average of group veclocities of degenerate phonon modes
            # and then calculate gv x gv to preserve symmetry
            gvs = np.zeros_like(gvs_rot)
            Fs = np.zeros_like(F_rot)
            for deg in deg_sets:
                gv_ave = gvs_rot[deg].sum(axis=0) / len(deg)
                F_ave = F_rot[deg].sum(axis=0)/len(deg)
                for j in deg:
                    gvs[j] = gv_ave
                    Fs[j]=F_ave
            gv_by_F_tensor.append([np.outer(gvs[b], Fs[b]) for b in range(len(gvs))])

        return np.array(gv_by_F_tensor)
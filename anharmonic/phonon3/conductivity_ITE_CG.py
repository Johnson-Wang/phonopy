import numpy as np
import sys
from phonopy.units import THz, kb_J, Angstrom, THzToEv, EV, total_time
from anharmonic.phonon3.collision import Collision
from anharmonic.phonon3.conductivity import Conductivity
unit_to_WmK = kb_J / Angstrom ** 3 * ((THz * Angstrom) * THzToEv * EV / kb_J) ** 2  / THz/ (2 * np.pi) # 2pi comes from the definition of tau
np.seterr(divide="ignore")
class conductivity_ITE_CG(Conductivity):
    def __init__(self,
                 interaction,
                 symmetry=None,
                 sigmas=[0.2],
                 grid_points=None,
                 temperatures=None,
                 max_ite= None,
                 adaptive_sigma_step = 1,
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
        self._F = None
        self._F0 = None
        self._F_prev = None
        self._b = None
        self._r = None
        self._r_prev = None
        self._z = None
        self._z_prev = None
        self.ite_init()
        self._ite_num = max_ite
        self._ite_step = 0
        if filename is not None:
            self._filename = filename
        else:
            self._filename = "ite_cg"

    def __iter__(self):
        return self

    def next(self):
        if (self._is_converge ==True).all():
            print "All calculations have converged"
            raise StopIteration
        elif self._ite_step == self._ite_num: # maximum iteration step reached
            non_converge=np.where(self._is_converge==False)
            sigma_nc=np.array(self._sigmas)[non_converge[0]] # sigma value for the non-converged cases
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
                if self._ite_step == 0:
                    self.calculate_residual0_at_sigma_and_temp() # initialize the residual and the searching path
                total_time.reset()
                self.run_at_sigma_and_temp()
                total_time.output()
        self.check_convergence()

        self.renew()
        self._ite_step += 1
        return self

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
        total_time.reset()
        self.run_smrt_sigma_adaption()
        total_time.output()
        if self._is_read_col:
            self._pp.release_amplitude_all()


    def run_smrt_sigma_adaption(self):
        asigma_step = 0
        while asigma_step <= self._max_asigma_step:
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
                    if asigma_step > 1:
                        print "Relative kappa difference %.2e" %self._rkappa[s,t].max()
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
                                # self._collision.set_asigma()
                            else:
                                self._collision.set_asigma()
                        self._collision.set_integration_weights()
                        self._collision.run_interaction_at_grid_point(self._collision.get_interaction_skip())
                        # self._collision.run_interaction_at_grid_point()

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


    def calculate_residual0_at_sigma_and_temp(self):
        t = self._itemp
        s = self._isigma
        if not self._is_converge[s,t]:
            if self._log_level:
                print "######Iteration preparation at the first step sigma=%s, t=%.2f#######"%(self._sigma, self._temp)
                print "Calculating the residual from SMRT..."
            self.print_calculation_progress_header()
            for i, grid_point in enumerate(self._grid_points):
                self._collision.calculate_collision(grid_point)
                #calculating scattering rate
                self.get_total_rotation()
                import anharmonic._phono3py as phono3c
                rec_lat = np.linalg.inv(self._primitive.get_cell())
                spg_mapping_index = np.unique(self._mappings, return_inverse=True)[1]
                branch_len = self._collision._collision_in.shape[-2]
                AF_sum = np.zeros((branch_len, 3), dtype="double")
                triplets1 = self._bz_to_pp_map[self._collision._grid_point_triplets[:,1]]
                F_prev = self._F_prev[s, spg_mapping_index[triplets1], t]
                phono3c.phonon_multiply_dmatrix_gbb_dvector_gb3(AF_sum,
                                                                self._collision._collision_in.copy(),
                                                                F_prev.copy(),
                                                                np.intc(self._collision._triplet_weights).copy(),
                                                                self._rot1_sums,
                                                                rec_lat)
                self._r_prev[s, i, t] = - AF_sum[:]
                if self._is_precondition:
                    #Here the preconditioning follows the algorithm in wiki/Conjugate_gradient_method
                    out = self._collision_out[s, i, t]
                    out_reverse = np.where(self._frequencies[i]>self._cutoff_frequency, 1 / out, 0)
                    self._z_prev[s, i, t] = self._r_prev[s, i, t] * out_reverse[..., np.newaxis]
                else:
                    self._z_prev[s, i, t] = self._r_prev[s, i, t]
                self._p_prev[s, i, t] = self._z_prev[s, i, t]
                self.print_calculation_progress(i)

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

    def calculate_gh_at_sigma_and_temp(self):
        import anharmonic._phono3py as phono3c
        if self._is_precondition:
            out = self._collision_out[self._isigma, :, self._itemp]
            out_reverse = np.where(self._frequencies>self._cutoff_frequency, 1 / out, 0)
            self._z[self._isigma, :, self._itemp] = self._r[self._isigma, :, self._itemp] * out_reverse[..., np.newaxis]
        else:
            self._z[self._isigma, :, self._itemp] = self._r[self._isigma, :, self._itemp]
        zr0 = np.zeros(3, dtype="double")
        phono3c.phonon_3_multiply_dvector_gb3_dvector_gb3(zr0,
                                                        self._z_prev[self._isigma,:,self._itemp].copy(),
                                                        self._r_prev[self._isigma,:,self._itemp].copy(),
                                                        np.intc(self._irr_index_mapping).copy(),
                                                        np.intc(self._kpoint_operations[self._rot_mappings]),
                                                        np.double(np.linalg.inv(self._primitive.get_cell())).copy())
        #Flexible preconditioned CG: r(i+1)-r(i) instead of r(i+1)
        r = self._r[self._isigma,:,self._itemp] - self._r_prev[self._isigma,:,self._itemp]
        zr1 = np.zeros(3, dtype="double")
        phono3c.phonon_3_multiply_dvector_gb3_dvector_gb3(zr1,
                                                        self._z[self._isigma,:,self._itemp].copy(),
                                                        r.copy(),
                                                        np.intc(self._irr_index_mapping).copy(),
                                                        np.intc(self._kpoint_operations[self._rot_mappings]),
                                                        np.double(np.linalg.inv(self._primitive.get_cell())).copy())
        zr1_over_zr0 = np.where(np.abs(zr0>0), zr1/zr0, 0)
        self._p[self._isigma,:,self._itemp] = self._z[self._isigma,:,self._itemp] +\
                                              zr1_over_zr0 * self._p_prev[self._isigma,:,self._itemp]


    def run_at_sigma_and_temp(self): # sigma and temperature
        "Run each single iteration, all terminology follows the description in wiki/Conjugate_gradient_method"
        t = self._itemp
        s = self._isigma
        if not self._is_converge[s, self._itemp]:
            if self._log_level:
                if self._sigma:
                    print "######Perturbation flow for the next iterative step at sigma=%s, t=%f#######"\
                          %(self._sigma, self._temp)
                else:
                    print "######Perturbation flow for the next iterative step with tetrahedron method#######"
                self.print_calculation_progress_header()
            for i, grid_point in enumerate(self._grid_points):
                self._collision.calculate_collision(grid_point)
                #calculating scattering rate
                self.calculate_At_s_t_g(s, t, i) # sigma, temperature, grid
                self.print_calculation_progress(i)
            # self._collision.write_collision_all(log_level=self._log_level)
        import anharmonic._phono3py as phono3c
        gz = np.zeros(3, dtype="double")
        phono3c.phonon_3_multiply_dvector_gb3_dvector_gb3(gz,
                                                        self._r_prev[self._isigma,:,self._itemp].copy(),
                                                        self._z_prev[self._isigma,:,self._itemp].copy(),
                                                        np.intc(self._irr_index_mapping).copy(),
                                                        np.intc(self._kpoint_operations[self._rot_mappings]),
                                                        np.double(np.linalg.inv(self._primitive.get_cell())).copy())
        ht = np.zeros(3, dtype="double")
        phono3c.phonon_3_multiply_dvector_gb3_dvector_gb3(ht,
                                                        self._p_prev[self._isigma,:,self._itemp].copy(),
                                                        self._t_prev[self._isigma,:,self._itemp].copy(),
                                                        np.intc(self._irr_index_mapping).copy(),
                                                        np.intc(self._kpoint_operations[self._rot_mappings]),
                                                        np.double(np.linalg.inv(self._primitive.get_cell())).copy())

        gz_over_ht = np.where(np.abs(ht)>0, gz / ht, 0)
        self._F[self._isigma,:,self._itemp] = self._F_prev[self._isigma,:,self._itemp] + gz_over_ht * self._p_prev[self._isigma,:,self._itemp]
        self._r[self._isigma,:,self._itemp] = self._r_prev[self._isigma,:,self._itemp] - gz_over_ht * self._t_prev[self._isigma,:,self._itemp]
        self.calculate_gh_at_sigma_and_temp()


    def ite_init(self):
        self.smrt()
        num_sigma = len(self._sigmas)
        num_temp = len(self._temperatures)
        self._F0 = self._F
        self._F_prev = self._F0.copy()
        self._F = np.zeros_like(self._F0)
        self._is_converge=np.zeros((num_sigma, num_temp), dtype="bool")
        self._r = np.zeros_like(self._F0) # residual
        self._r_prev = np.zeros_like(self._F0) #previous residual
        self._t_prev = np.zeros_like(self._F0) # Ap
        self._p = np.zeros_like(self._F0)  # searching path
        self._p_prev = np.zeros_like(self._F0) # previous searching path
        self._z = np.zeros_like(self._F0)
        self._z_prev = np.zeros_like(self._F0)

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

    def renew(self):
        for s, sigma in enumerate(self._sigmas):
            for t, temp in enumerate(self._temperatures):
                if not self._is_converge[s, t]:
                    self._F_prev[s,:,t] =  self._F[s,:,t]
                    self._p_prev[s,:,t] =  self._p[s,:,t]
                    self._r_prev[s,:,t] =  self._r[s,:,t]
                    self._z_prev[s,:,t] = self._z[s,:,t]

    def get_total_rotation(self):
        triplet1 = self._collision._grid_point_triplets[:,1]
        triplet1_pp = self._bz_to_pp_map[triplet1]
        num_rots = len(self._collision._kpg_at_q_index) # kpoint group at q
        inv_rot = self._kpoint_group_inv_map[self._rot_mappings[triplet1_pp]]
        r1dotr2_inv = self._kpoint_group_sum_map[self._collision._kpg_at_q_index][:,inv_rot]
        a1 = self._kpoint_operations[r1dotr2_inv].sum(axis=0)
        self._rot1_sums = a1 / float(num_rots)

    def calculate_At_s_t_g(self,isigma, itemp, igrid):
        self.get_total_rotation()
        import anharmonic._phono3py as phono3c
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        spg_mapping_index = np.unique(self._mappings, return_inverse=True)[1]
        branch_len = self._collision._collision_in.shape[-2]
        t_prev = np.zeros((branch_len, 3), dtype="double")
        triplets1 = self._bz_to_pp_map[self._collision._grid_point_triplets[:,1]]
        p_prev = self._p_prev[isigma, spg_mapping_index[triplets1], itemp]
        scatt = self._collision._collision_in
        out = self._p_prev[isigma, igrid, itemp] * self._collision._collision_out[:,np.newaxis]
        phono3c.phonon_multiply_dmatrix_gbb_dvector_gb3(t_prev,
                                                        np.double(scatt).copy(),
                                                        np.double(p_prev).copy(),
                                                        np.intc(self._collision._triplet_weights).copy(),
                                                        np.double(self._rot1_sums).copy(),
                                                        np.double(rec_lat).copy())
        self._t_prev[isigma, igrid, itemp] = t_prev + out


    def set_kappa(self):
        for i,s in enumerate(self._sigmas):
            self.set_kappa_at_sigma(i)

    def set_equivalent_gamma_at_sigma(self,i):
        gv_norm=np.sum(self._gv**2, axis=-1)
        for j, t in enumerate(self._temperatures):
            FdotvT2 = 2 * np.sum(self._F[i,:,j] * self._gv, axis=-1) * t ** 2
            self._gamma[i,:,j] = self._frequencies * gv_norm * np.where(np.abs(FdotvT2)==0, 0, 1 / FdotvT2)

    def set_kappa_at_sigma(self,s):
        self.set_kappa_at_s_c(s)

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
        kappa *= self._kappa_factor / np.prod(self._mesh)
        kappa_max = kappa.sum(axis=(0,2)).max(axis=-1)
        rkappa = np.sum(np.abs(kappa - self._kappa[s]), axis=(0, 2)) # over qpoints and nbands
        for i in range(6):
            self._rkappa[s, :, i] = rkappa[:, i] /  kappa_max
        self._kappa[s] = kappa



    def get_kappa_residual_at_s_t(self, s, t):
        import anharmonic._phono3py as phono3c
        out = self._collision_out[s, :, t]
        out_reverse = np.where(self._frequencies>self._cutoff_frequency, 1 / out, 0)
        r = self._r[s, :, t] * out_reverse[..., np.newaxis]
        #Normalize the residual to the unit of F
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        dkappa = np.zeros((self._frequencies.shape[0], self._frequencies.shape[1], 6), dtype="double")
        phono3c.phonon_gb33_multiply_dvector_gb3_dvector_gb3(dkappa,
                                                             self._b[:,t].copy(),
                                                             r.copy(),
                                                             np.intc(self._irr_index_mapping).copy(),
                                                             np.intc(self._kpoint_operations[self._rot_mappings]).copy(),
                                                             rec_lat.copy())
        # dkappa = np.sum(self._grid_weights[:, np.newaxis, np.newaxis] * np.abs(dkappa), axis=(0,1))
        dkappa = np.sum(np.abs(dkappa), axis=(0,1))
        return dkappa * self._temperatures[t] ** 2 * self._kappa_factor / np.prod(self._mesh)
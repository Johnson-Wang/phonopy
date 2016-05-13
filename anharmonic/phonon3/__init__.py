import sys
import numpy as np
from anharmonic.phonon3.triplets import get_grid_address, get_ir_grid_points
from anharmonic.phonon3.imag_self_energy import ImagSelfEnergy, MatrixContribution
from anharmonic.phonon3.frequency_shift import FrequencyShift
from anharmonic.phonon3.interaction import Interaction
from anharmonic.phonon3.conductivity_RTA import conductivity_RTA
from anharmonic.phonon3.conductivity_ITE import conductivity_ITE
from anharmonic.phonon3.conductivity_ITE_CG import conductivity_ITE_CG
from anharmonic.phonon3.jointDOS import JointDos
from anharmonic.phonon3.gruneisen import Gruneisen
from anharmonic.file_IO import write_kappa_to_hdf5, write_iso_scattering_to_hdf5, write_joint_dos
from anharmonic.other.isotope import CollisionIso
from anharmonic.file_IO import read_gamma_from_hdf5,read_kappa_from_hdf5, write_damping_functions, write_linewidth, \
    write_frequency_shift,write_linewidth_band_csv, write_kappa_to_tecplot_BZ, write_matrix_contribution
from anharmonic.phonon3.decay_channel import DecayChannel
from anharmonic.phonon3.conductivity import Conductivity
from phonopy.units import VaspToTHz
from phonopy.structure.symmetry import Symmetry
from collision import Collision

class Progress_monitor:
    def __init__(self, num_task):
        self.num_task = num_task
        self.last_step=0
        self.current_step=0
        self.preprint()
    def preprint(self):
        print "Calculation progress..."
        print "%10s%%"*10 %tuple(np.arange(1,11)*10)
    def progress_print(self,i): # i starts from 0
        self.current_step = float(i+1) / self.num_task * 110.
        num_of_symbols_to_print = np.rint(self.current_step) - self.last_step
        self.last_step += num_of_symbols_to_print
        sys.stdout.write("="*num_of_symbols_to_print)
        sys.stdout.flush()
        if i+1 == self.num_task:
            print

class Phono3py:
    def __init__(self,
                 interaction,
                 mass_variances=None,
                 length=None,
                 adaptive_sigma_step = 0,
                 frequency_factor_to_THz=None,
                 is_nosym=False,
                 symprec=1e-5,
                 log_level=0,
                 is_thm = False, # is tetrahedron method
                 write_tecplot=False):
        self._interaction = interaction
        self._primitive = interaction.get_primitive()
        self._supercell = interaction.get_supercell()
        self._mesh = interaction.get_mesh_numbers()
        self._band_indices = interaction.get_band_indices()
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_nosym = is_nosym
        self._is_symmetry = not is_nosym
        self._symprec = symprec
        self._log_level = log_level
        self._asigma_step=adaptive_sigma_step
        self._step=0
        self._kappa = None
        self._gamma = None
        self._br = None
        self._is_thm = is_thm
        self._write_tecplot=write_tecplot
        self._mass_variances = mass_variances
        self._length=length
        self._symmetry = None
        self._primitive_symmetry = None
        self._search_symmetry()
        self._search_primitive_symmetry()

    def get_symmetry(self):
        """return symmetry of supercell"""
        return self._symmetry

    def get_primitive_symmetry(self):
        return self._primitive_symmetry

    def _search_symmetry(self):
        self._symmetry = Symmetry(self._supercell,
                                  self._symprec,
                                  self._is_symmetry)

    def _search_primitive_symmetry(self):
        self._primitive_symmetry = Symmetry(self._primitive,
                                            self._symprec,
                                            self._is_symmetry)
        if (len(self._symmetry.get_pointgroup_operations()) !=
            len(self._primitive_symmetry.get_pointgroup_operations())):
            print ("Warning: point group symmetries of supercell and primitive"
                   "cell are different.")

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             nac_q_direction=None,
                             frequency_scale_factor=None):
        self._interaction.set_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor)
        self._interaction.set_nac_q_direction(nac_q_direction=nac_q_direction)
                           
    def get_imag_self_energy(self,
                             grid_points,
                             frequency_step=1.0,
                             sigmas=[None],
                             temperatures=[0.0],
                             filename=None):
        ise = ImagSelfEnergy(self._interaction, is_thm = self._is_thm)
        for gp in grid_points:
            ise.set_grid_point(gp)
            ise.run_interaction()
            for sigma in sigmas:
                ise.set_sigma(sigma)
                for t in temperatures:
                    ise.set_temperature(t)
                    max_freq = (np.amax(self._interaction.get_phonons()[0]) * 2
                                + sigma * 4)
                    fpoints = np.arange(0, max_freq + frequency_step / 2,
                                        frequency_step)
                    ise.set_fpoints(fpoints)
                    ise.run()
                    gamma = ise.get_imag_self_energy()

                    for i, bi in enumerate(self._band_indices):
                        pos = 0
                        for j in range(i):
                            pos += len(self._band_indices[j])

                        write_damping_functions(
                            gp,
                            bi,
                            self._mesh,
                            fpoints,
                            gamma[:, pos:(pos + len(bi))].sum(axis=1) / len(bi),
                            sigma=sigma,
                            temperature=t,
                            filename=filename)

    def get_matrix_contribution(self,
                                grid_points,
                                frequency_step=1.0,
                                sigmas=[0.1],
                                is_adaptive_sigma=False,
                                filename=None):
        mc = Collision(self._interaction,
                            sigmas=sigmas,
                            is_adaptive_sigma=is_adaptive_sigma,
                            frequencies = self._frequencies,
                            degeneracy=self._degeneracy,
                            write=write_scr,
                            read=read_scr,
                            cutoff_frequency=self._cutfr,
                            cutoff_lifetime=self._cutlt)
        is_sum = False
        if grid_points == None:
            is_sum = True
            if self._is_nosym: # All grid points
                grid_points = np.arange(np.product(self._mesh))
            else: # Automatic sampling
                (grid_points,
                 grid_weights,
                 grid_address) = get_ir_grid_points(
                    self._mesh,
                    self._primitive)
        matrix_contributions = []
        for gp in grid_points:
            print "# grid %d" %gp
            mc.set_grid_point(gp)
            mc.run_interaction()
            matrix_contributions_temp = []
            for sigma in sigmas:
                mc.set_sigma(sigma)
                max_freq = (np.amax(self._interaction.get_phonons()[0]) * 2
                            + sigma * 4)
                fpoints = np.arange(0, max_freq + frequency_step / 2,
                                    frequency_step)
                mc.set_fpoints(fpoints)
                mc.run()
                matrix_contribution = mc.get_imag_self_energy()
                matrix_contributions_temp.append(matrix_contribution)
                if not is_sum:
                    write_matrix_contribution(
                                gp,
                                self._mesh,
                                fpoints,
                                matrix_contribution,
                                sigma=sigma,
                                filename=filename)
            matrix_contributions.append(matrix_contributions_temp)
        matrix_contributions = np.array(matrix_contributions)

        if is_sum:
            for i, sigma in enumerate(sigmas):
                write_matrix_contribution(
                                    None,
                                    self._mesh,
                                    fpoints,
                                    np.sum(matrix_contributions[:,i], axis=0),
                                    sigma=sigma,
                                    filename=filename)

    def get_decay_channels(self,
                           grid_points,
                           sets_of_band_indices,
                           sigmas=[0.2],
                           read_amplitude=False,
                           temperature=None,
                           filename=None):

        if grid_points==None:
            print "Grid points are not specified."
            return False

        if sets_of_band_indices==None:
            print "Band indices are not specified."
            return False
        self._read_amplitude = read_amplitude
        decay=DecayChannel(self._interaction)
        for gp in grid_points:
            decay.set_grid_point(gp)
            if self._log_level:
                weights = self._interaction.get_triplets_at_q()[1]
                print "------ Decay channel ------"
                print "Number of ir-triplets:",
                print "%d / %d" % (len(weights), weights.sum())
            decay.run_interaction(read_amplitude=self._read_amplitude)
            for sigma in sigmas:
                decay.set_sigma(sigma)
                for i, t in enumerate(temperature):
                    decay.set_temperature(t)
                    decay.get_decay_channels(filename = filename)



    def get_linewidth(self,
                      grid_points,
                      sigmas=[0.1],
                      temperatures=None,
                      read_amplitude=False,
                      is_nu=False,
                      band_paths=None,
                      filename=None):
        ise = ImagSelfEnergy(self._interaction, is_nu=is_nu)
        if temperatures is None:
            temperatures = np.arange(0, 1000.0 + 10.0 / 2.0, 10.0)
        self._temperatures=temperatures
        self._read_amplitude=read_amplitude
        if grid_points is not None:
            self.get_linewidth_at_grid_points(ise,
                                              sigmas,
                                              temperatures,
                                              grid_points,
                                              is_nu,
                                              filename)
        elif band_paths is not None:
            self.get_linewidth_at_paths(ise,
                                        sigmas,
                                        temperatures,
                                        band_paths,
                                        is_nu,
                                        filename)

    def get_linewidth_at_paths(self,
                               ise,
                               sigmas,
                               temperatures,
                               band_paths,
                               is_nu,
                               filename):

        if self._is_nosym:
            grid_address, grid_mapping =get_grid_address([x + (x % 2 == 0) for x in self._mesh], is_return_map=True)
        else:
            grids, weights, grid_address, grid_mapping = get_ir_grid_points([x + (x % 2 == 0) for x in self._mesh],
                                                                            self._primitive,
                                                                            is_return_map=True)
        qpoints_address=grid_address / np.array(self._mesh, dtype=float)
        print "Paths in reciprocal reduced coordinates:"
        if is_nu:
            path_fun=self.path_nu
        else:
            path_fun=self.path
        for sigma in sigmas:
            self._sigma = sigma
            lws_paths=[]
            distances=[]
            frequencies=[]
            new_paths = []
            for path in band_paths:
                print "[%5.2f %5.2f %5.2f] --> [%5.2f %5.2f %5.2f]" % (tuple(path[0]) + tuple(path[-1]))
                (new_path,
                 dists,
                freqs,
                lws)=path_fun(path,
                              qpoints_address,
                              grid_mapping,
                              ise,
                              temperatures)
                new_paths.append(new_path)
                distances.append(dists)
                frequencies.append(freqs)
                lws_paths.append(lws)
            self._distances=distances
            self._freqs_on_path=frequencies
            self._lws_on_path=lws_paths
            write_linewidth_band_csv(new_paths,
                                     distances=self._distances,
                                     lws=self._lws_on_path,
                                     band_indices=self._band_indices,
                                     temperatures=temperatures,
                                     frequencies=self._freqs_on_path,
                                     is_nu=is_nu,
                                     filename="lw_band-sigma%.2f.csv"%sigma)
            self.plot_lw()

    def plot_lw(self, symbols=None):
        import matplotlib.pyplot as plt
        if symbols:
            from matplotlib import rc
            rc('text', usetex=True)
        distances=np.array(sum(np.array(self._distances).tolist(), []))
        special_point=[0.0] + [d[-1] for d in self._distances]
        lws=np.concatenate(tuple(self._lws_on_path), axis=0)
        for t in np.arange(lws.shape[1]):
            for b in np.arange(lws.shape[2]):
                plt.figure()
                plt.plot(distances, lws[:,t,b])
                plt.ylabel('Linewidth')
                plt.xlabel('Wave vector')
                if symbols and len(symbols)==len(special_point):
                    plt.xticks(special_point, symbols)
                else:
                    plt.xticks(special_point, [''] * len(special_point))
                plt.xlim(0, special_point[-1])
                plt.axhline(y=0, linestyle=':', linewidth=0.5, color='b')
                if len(lws.shape)== 4:
                    plt.legend(["Total","Normal process", "Umklapp process"])
                plt.savefig("linewidth-t%d-b%s.pdf"%(self._temperatures[t],
                                                   "".join(map(str,np.array(self._band_indices[b])+1))))
                plt.show()


    def path_nu(self,
                path,
                qpoints_address,
                grids_mapping,
                ise,
                temperatures):
        lw_on_path=np.zeros((len(path),len(temperatures), len(self._band_indices)), dtype="double")
        lw_N_on_path=np.zeros_like(lw_on_path)
        lw_U_on_path=np.zeros_like(lw_on_path)
        freqs_on_path=np.zeros((len(path), len(self._band_indices)), dtype="double")
        new_path = []
        for p, qpoint in enumerate(path):
            # dist_vector=np.abs(qpoints_address-qpoint)
            # distance= np.array([np.linalg.norm(x) for x in  dist_vector])
            # grid=np.argmin(distance)
            # print "grid1: ", grid
            # calculating the grid position from qpoint
            mesh=np.array(self._mesh)
            grid_point= np.rint(qpoint *mesh).astype(int)
            mesh2=mesh+(mesh%2==0)
            grid_address = np.where(grid_point>=0, grid_point,mesh2+grid_point)
            grid=np.dot(grid_address, np.r_[1,np.cumprod(mesh2[:-1])])
            ise.set_grid_point(grids_mapping[grid])
            if self._log_level:
                weights = self._interaction.get_triplets_at_q()[1]
                print "------ Linewidth ------(%d/%d)"%(p, len(path))
                print "calculated at qpoint", qpoint
                print "but represented by qpoint", qpoints_address[grid],  "due to the finite mesh points"
                print "grid:", grid
                print "Number of ir-triplets:",
                print "%d / %d" % (len(weights), weights.sum())
            new_path.append(qpoints_address[grid])
            ise.run_interaction(self._read_amplitude)
            ise.set_sigma(self._sigma)
            freqs_on_path[p]=self._interaction._frequencies[grids_mapping[grid],self._band_indices]
            gamma = np.zeros((len(temperatures),
                              len(self._band_indices)),
                             dtype='double')
            gamma_N=np.zeros_like(gamma)
            gamma_U=np.zeros_like(gamma)
            for i, t in enumerate(temperatures):
                ise.set_temperature(t)
                ise.run_at_sigma_and_temp()
                gamma[i] = ise.get_imag_self_energy()
                gamma_N[i]=ise.get_imag_self_energy_N()
                gamma_U[i]=ise.get_imag_self_energy_U()

            for i, bi in enumerate(self._band_indices):
                pos = 0
                for j in range(i):
                    pos += len(self._band_indices[j])
                lw_on_path[p,:,i]=gamma[:, pos:(pos+len(bi))].sum(axis=1) * 2 / len(bi)
                lw_N_on_path[p,:,i]=gamma_N[:, pos:(pos+len(bi))].sum(axis=1) * 2 / len(bi)
                lw_U_on_path[p,:,i]=gamma_U[:, pos:(pos+len(bi))].sum(axis=1) * 2 / len(bi)
        lws=np.concatenate((lw_on_path[..., np.newaxis],
                            lw_N_on_path[..., np.newaxis],
                            lw_U_on_path[..., np.newaxis]),
                           axis=-1)
        new_path = np.array(new_path)
        distances=map(lambda x: np.linalg.norm(np.dot(x-new_path[0],
                                                      np.linalg.inv(self._primitive.get_cell().T))),new_path)
        return (new_path, distances, freqs_on_path, lws)


    def path(self,
             path,
             qpoints_address,
             grids_mapping,
             ise,
             temperatures):
        lw_on_path=np.zeros((len(path),len(temperatures), len(self._band_indices)), dtype="double")
        freqs_on_path=np.zeros((len(path), len(self._band_indices)), dtype="double")
        new_path = []
        for p, qpoint in enumerate(path):
            dist_vector=np.abs(qpoints_address-qpoint)
            distance= np.array([np.linalg.norm(x) for x in  dist_vector])
            grid=np.argmin(distance)
            ise.set_grid_point(grids_mapping[grid])
            if self._log_level:
                weights = self._interaction.get_triplets_at_q()[1]
                print "------ Linewidth ------(%d/%d)"%(p, len(path))
                print "calculated at qpoint", qpoint
                print "but represented by qpoint", qpoints_address[grid],  "due to the finite mesh"
                print "Number of ir-triplets:",
                print "%d / %d" % (len(weights), weights.sum())
            new_path.append(qpoints_address[grid])
            ise.run_interaction(self._read_amplitude)
            ise.set_sigma(self._sigma)
            freqs_on_path[p]=self._interaction._frequencies[grids_mapping[grid]][self._band_indices]
            gamma = np.zeros((len(temperatures),
                              len(self._band_indices)),
                             dtype='double')
            for i, t in enumerate(temperatures):
                ise.set_temperature(t)
                ise.run_at_sigma_and_temp()
                gamma[i] = ise.get_imag_self_energy()

            for i, bi in enumerate(self._band_indices):
                pos = 0
                for j in range(i):
                    pos += len(self._band_indices[j])
                lw_on_path[p,:,i]=gamma[:, pos:(pos+len(bi))].sum(axis=1) * 2 / len(bi)
        new_path = np.array(new_path)
        distances=map(lambda x: np.linalg.norm(np.dot(x-new_path[0],
                                                      np.linalg.inv(self._primitive.get_cell().T))),new_path)
        distances=np.array(distances, dtype="double")
        return (new_path, distances, freqs_on_path, lw_on_path)


    def get_linewidth_at_grid_points(self,
                                     ise,
                                     sigmas,
                                     temperatures,
                                     grid_points,
                                     is_nu,
                                     filename):
        for gp in grid_points:
            ise.set_grid_point(gp)
            if self._log_level:
                weights = self._interaction.get_triplets_at_q()[1]
                print "------ Linewidth ------"
                print "Number of ir-triplets:",
                print "%d / %d" % (len(weights), weights.sum())
            ise.run_interaction(self._read_amplitude)
            for sigma in sigmas:
                ise.set_sigma(sigma)
                gamma = np.zeros((len(temperatures),
                                  len(self._band_indices)),
                                 dtype='double')
                if is_nu:
                    gamma_N=np.zeros_like(gamma)
                    gamma_U=np.zeros_like(gamma)
                for i, t in enumerate(temperatures):
                    ise.set_temperature(t)
                    ise.run_at_sigma_and_temp()
                    gamma[i] = ise.get_imag_self_energy()
                    if is_nu:
                        gamma_N[i]=ise.get_imag_self_energy_N()
                        gamma_U[i]=ise.get_imag_self_energy_U()

                for i, bi in enumerate(self._band_indices):
                    pos = 0
                    for j in range(i):
                        pos += len(self._band_indices[j])

                    if not is_nu:
                        write_linewidth(gp,
                                        bi,
                                        temperatures,
                                        gamma[:, pos:(pos+len(bi))],
                                        self._mesh,
                                        sigma=sigma,
                                        filename=filename)
                    else:
                        write_linewidth(gp,
                                        bi,
                                        temperatures,
                                        gamma[:, pos:(pos+len(bi))],
                                        self._mesh,
                                        sigma=sigma,
                                        filename=filename,
                                        gamma_N=gamma_N[:, pos:(pos+len(bi))],
                                        gamma_U=gamma_U[:, pos:(pos+len(bi))])

    def get_frequency_shift(self,
                            grid_points,
                            epsilon=0.1,
                            temperatures=np.linspace(0,1000,endpoint=True, num=101),
                            filename=None):
        fst = FrequencyShift(self._interaction)
        for gp in grid_points:
            fst.set_grid_point(gp)
            if self._log_level:
                weights = self._interaction.get_triplets_at_q()[1]
                print "------ Frequency shift -o- ------"
                print "Number of ir-triplets:",
                print "%d / %d" % (len(weights), weights.sum())
            fst.run_interaction()
            fst.set_epsilon(epsilon)
            delta = np.zeros((len(temperatures),
                              len(self._band_indices)),
                             dtype='double')
            for i, t in enumerate(temperatures):
                fst.set_temperature(t)
                fst.run()
                delta[i] = fst.get_frequency_shift()

            for i, bi in enumerate(self._band_indices):
                pos = 0
                for j in range(i):
                    pos += len(self._band_indices[j])

                write_frequency_shift(gp,
                                      bi,
                                      temperatures,
                                      delta[:, pos:(pos+len(bi))],
                                      self._mesh,
                                      epsilon=epsilon,
                                      filename=filename)

    def get_thermal_conductivity(self,
                                 sigmas=None,
                                 temperatures=None,
                                 grid_points=None,
                                 mesh_divisors=None,
                                 coarse_mesh_shifts=None,
                                 cutoff_lifetime=1e-4,  # in second
                                 diff_kappa = 1e-5,  # relative
                                 is_nu=False,
                                 no_kappa_stars=False,
                                 gv_delta_q=1e-4,  # for group velocity
                                 write_gamma=False,
                                 read_gamma=False,
                                 kappa_write_step=None,
                                 filename=None):

        br = conductivity_RTA(self._interaction,
                              symmetry=self._primitive_symmetry,
                              sigmas=sigmas,
                              asigma_step= self._asigma_step,
                              temperatures=temperatures,
                              mesh_divisors=mesh_divisors,
                              coarse_mesh_shifts=coarse_mesh_shifts,
                              grid_points=grid_points,
                              cutoff_lifetime=cutoff_lifetime,
                              diff_kappa= diff_kappa,
                              is_nu=is_nu,
                              no_kappa_stars=no_kappa_stars,
                              gv_delta_q=gv_delta_q,
                              log_level=self._log_level,
                              write_tecplot = self._write_tecplot,
                              kappa_write_step=kappa_write_step,
                              is_thm=self._is_thm,
                              filename=filename)

        if read_gamma:
            for sigma in sigmas:
                self.read_gamma_at_sigma(sigma)
        br.calculate_kappa(write_gamma=write_gamma)
        mode_kappa = br.get_kappa()
        gamma = br.get_gamma()
        gamma_N=br._gamma_N
        gamma_U=br._gamma_U
        if self._log_level:
            br.print_kappa()
        if grid_points is None:
            temperatures = br.get_temperatures()
            for i, sigma in enumerate(sigmas):
                kappa = mode_kappa[i]
                write_kappa_to_hdf5(gamma[i],
                                    temperatures,
                                    br.get_mesh_numbers(),
                                    frequency=br.get_frequencies(),
                                    group_velocity=br.get_group_velocities(),
                                    heat_capacity=br.get_mode_heat_capacities(),
                                    kappa=kappa,
                                    qpoint=br.get_qpoints(),
                                    weight=br.get_grid_weights(),
                                    mesh_divisors=br.get_mesh_divisors(),
                                    sigma=sigma,
                                    filename=filename,
                                    gnu=(gamma_N[i],gamma_U[i]))
                if self._write_tecplot:
                    for j,temp in enumerate(temperatures):
                        write_kappa_to_tecplot_BZ(np.where(gamma[i,:,j]>1e-8, gamma[i,:,j],0),
                                               temp,
                                               br.get_mesh_numbers(),
                                               bz_q_address=br._bz_grid_address / br.get_mesh_numbers().astype(float),
                                               tetrahedrdons=br._unique_vertices,
                                               bz_to_pp_mapping=br._bz_to_pp_map,
                                               rec_lattice=np.linalg.inv(br._primitive.get_cell()),
                                               spg_indices_mapping=br._irr_index_mapping,
                                               spg_rotation_mapping=br._rot_mappings,
                                               frequency=br.get_frequencies(),
                                               group_velocity=br.get_group_velocities(),
                                               heat_capacity=br.get_mode_heat_capacities()[:,j],
                                               kappa=kappa[:,j],
                                               weight=br.get_grid_weights(),
                                               sigma=sigma,
                                               filename="bz")

        self._kappa = mode_kappa
        self._gamma = gamma
        self._br=br

    def read_gamma_at_sigma(self, sigma):
        br = self._br
        gamma = []
        try:
            properties = read_kappa_from_hdf5(
                br.get_mesh_numbers(),
                mesh_divisors=br.get_mesh_divisors(),
                sigma=sigma,
                filename=br._filename)
            if properties is None:
                raise ValueError
            tbr =  br.get_temperatures()
            ts_pos0, ts_pos1 = np.where(properties['temperature'] == tbr.reshape(-1,1))
            br.set_temperatures(tbr[ts_pos0])
            gamma_at_sigma=properties['gamma'][:,ts_pos1]
            gamma.append(gamma_at_sigma)
            br.broadcast_collision_out(np.double(gamma))
        except ValueError:
            properties = []
            for point in br.get_grid_points():
                property = read_kappa_from_hdf5(
                    br.get_mesh_numbers(),
                    mesh_divisors=br.get_mesh_divisors(),
                    grid_point=point,
                    sigma=sigma,
                    filename=br._filename)
                properties.append(property)
            tbr =  br.get_temperatures()
            ts_pos0,ts_pos1 = np.where(properties[0]['temperature'] == tbr.reshape(-1,1))
            br.set_temperatures(tbr[ts_pos0])
            gamma_at_sigma = np.array([p['gamma'][ts_pos1] for p in properties], dtype="double")
            gamma.append(gamma_at_sigma)
            br.broadcast_collision_out(np.double(gamma))

    def get_kappa_ite(self,
                      sigmas=[0.2],
                      temperatures=None,
                      grid_points=None,
                      max_ite = None,
                      no_kappa_stars=False,
                      diff_kappa = 1e-5, # relative difference
                      write_gamma=False,
                      read_gamma=False,
                      read_col=False,
                      write_col=False,
                      filename="ite"):
        bis=conductivity_ITE(self._interaction, #Iterative Boltzmann solutions
                             symmetry=self._primitive_symmetry,
                             sigmas=sigmas,
                             grid_points = grid_points,
                             temperatures=temperatures,
                             max_ite = max_ite,
                             adaptive_sigma_step = self._asigma_step,
                             no_kappa_stars=no_kappa_stars,
                             diff_kappa= diff_kappa,
                             mass_variances=self._mass_variances,
                             length=self._length,
                             log_level=self._log_level,
                             read_gamma = read_gamma,
                             write_gamma = write_gamma,
                             read_col = read_col,
                             write_col=write_col,
                             filename=filename)
        try:
            for bi in bis:
                bi.set_kappa()
                print "After %d iteration(s), the thermal conductivities are recalculated to be (W/mK)"%bi._ite_step
                bi.print_kappa()
        except KeyboardInterrupt:
            print
            print "A keyboard Interruption is captured. The iterations are terminated!"
            print "The kappa is retrieved from the last iterations."
            bis._F = bis._F_prev
        print "Final thermal conductivity (W/mK)"
        for s, sigma in enumerate(sigmas):
            bis.set_equivalent_gamma_at_sigma(s)
            bis.set_kappa_at_sigma(s)
            write_kappa_to_hdf5(bis._gamma[s],
                                bis._temperatures,
                                bis._mesh,
                                frequency=bis._frequencies,
                                group_velocity=bis._gv,
                                heat_capacity=bis.get_mode_heat_capacities(),
                                kappa=bis._kappa[s],
                                qpoint=bis._qpoints,
                                weight=bis._grid_weights,
                                sigma=sigma,
                                filename=bis._filename)
        self._gamma=bis._gamma
        self._kappa=bis._kappa
        bis.print_kappa()
        # bis.print_kappa_rta()

    def get_kappa_ite_cg(self,
                          sigmas=[0.2],
                          temperatures=None,
                          grid_points=None,
                          max_ite = None,
                          no_kappa_stars=False,
                          diff_kappa = 1e-5, # relative value
                          write_gamma=False,
                          read_gamma=False,
                          read_col=False,
                          write_col=False,
                          filename="ite_cg"):
        bis=conductivity_ITE_CG(self._interaction, #Iterative Boltzmann solutions
                            symmetry=self._primitive_symmetry,
                            sigmas=sigmas,
                            grid_points = grid_points,
                            temperatures=temperatures,
                            max_ite = max_ite,
                            adaptive_sigma_step = self._asigma_step,
                            no_kappa_stars=no_kappa_stars,
                            diff_kappa= diff_kappa,
                            mass_variances=self._mass_variances,
                            length=self._length,
                            log_level=self._log_level,
                            read_gamma = read_gamma,
                            write_gamma = write_gamma,
                            read_col = read_col,
                            write_col=write_col,
                            is_thm=self._is_thm,
                            filename=filename)
        try:
            for bi in bis:
                bi.set_kappa()
                print "After %d iteration(s), the thermal conductivities are recalculated to be (W/mK)"%bi._ite_step
                bi.print_kappa()
        except KeyboardInterrupt:
            print
            print "A keyboard Interruption is captured. The iterations are terminated!"
            print "The kappa is retrieved from the last iterations."
            bis._F = bis._F_prev
        print "Final thermal conductivity (W/mK)"
        for s, sigma in enumerate(sigmas):
            bis.set_equivalent_gamma_at_sigma(s)
            bis.set_kappa_at_sigma(s)
            write_kappa_to_hdf5(bis._gamma[s],
                                bis._temperatures,
                                bis._mesh,
                                frequency=bis._frequencies,
                                group_velocity=bis._gv,
                                heat_capacity=bis.get_mode_heat_capacities(),
                                kappa=bis._kappa[s],
                                qpoint=bis._qpoints,
                                weight=bis._grid_weights,
                                sigma=sigma,
                                filename=bis._filename)
        self._gamma=bis._gamma
        self._kappa=bis._kappa
        bis.print_kappa()

class Phono3pyIsotope:
    def __init__(self,
                 mesh,
                 primitive,
                 mass_variances, # length of list is num_atom.
                 band_indices=None,
                 sigmas=[],
                 frequency_factor_to_THz=VaspToTHz,
                 symprec=1e-5,
                 cutoff_frequency=None,
                 log_level=0,
                 lapack_zheev_uplo='L',
                 temperatures=np.array([300], dtype="double")):
        self._sigmas = sigmas
        self._log_level = log_level
        self._iso = CollisionIso(mesh,
                                 primitive,
                                mass_variances,
                                band_indices=band_indices,
                                frequency_factor_to_THz=frequency_factor_to_THz,
                                symprec=symprec,
                                temperatures=temperatures,
                                cutoff_frequency=cutoff_frequency,
                                lapack_zheev_uplo=lapack_zheev_uplo)
        self._temps = temperatures

    def run(self, grid_points):
        if grid_points is None:
            (grid_points,grid_weights,grid_address)=get_ir_grid_points(self._iso._mesh,self._primitive)
        else:
            grid_weights = None
        iso_gamma=np.zeros((len(self._sigmas), len(grid_points), len(self._iso._temps), len(self._iso._band_indices)), dtype="double")
        process = Progress_monitor(len(grid_points))
        for i,gp in enumerate(grid_points):
            process.progress_print(i)
            self._iso.set_grid_point(gp)
            if self._iso._phonon_done is None:
                self._iso._allocate_phonon()
            if self._log_level:
                print "------ Isotope scattering ------"
                print "Grid point: %d" % gp

            for j, sigma in enumerate(self._sigmas):
                self._iso.set_sigma(sigma)
                for k, temp in enumerate(self._temps):
                    self._iso.set_temperature(temp)
                    self._iso.run()
                    iso_gamma[j,i,k]=self._iso.get_gamma()

        for j, sigma in enumerate(self._sigmas):
            if sigma is not None:
                print "Mean isotope scattering rate at sigma=%f:"%sigma
                filename = "iso_gamma-m%d%d%d-s%.2f.hdf5" %(tuple(self._iso._mesh)+(sigma,))
            else:
                print "Mean isotope scattering rate using Tetrahedra method"
                filename = "iso_gamma-m%d%d%d.hdf5" %(tuple(self._iso._mesh))

            if grid_weights is not None:
                print "%20s %20s" %("Temperature(K)", "Gamma (THz)")
                for k, temp in enumerate(self._iso._temps):
                    iso_ave = np.sum(np.dot(grid_weights, iso_gamma[j,:,k])) / grid_weights.sum()
                    print "%20.2f %20.7e" %(temp, iso_ave)
                write_iso_scattering_to_hdf5(iso_gamma[j],
                                             mesh=self._iso._mesh,
                                             temperatures=self._iso._temps,
                                             sigma=sigma,
                                             filename=filename)
            else: # Grid-point mode
                for k, temp in enumerate(self._iso._temps):
                    print "T=%.2f" %temp
                    for i,gp in enumerate(grid_points):
                        print "  Grid-point: %d" %gp
                        print  "    %-20s %-20s" %("Frequency (THz)", "Gamma (THz)")
                        for l in np.arange(self._iso._frequencies.shape[-1]):
                            print "    %-20.4f %-20.4e" %(self._iso._frequencies[gp, l], iso_gamma[j, i, k, l])


                
    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             frequency_scale_factor=None,
                             decimals=None):
        self._primitive = primitive
        self._iso.set_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals)

    def set_sigma(self, sigma):
        self._iso.set_sigma(sigma)

class JointDOS:
    def __init__(self,
                 supercell,
                 primitive,
                 mesh,
                 fc2,
                 nac_params=None,
                 nac_q_direction=None,
                 sigmas=[],
                 cutoff_frequency=1e-4,
                 frequency_step=None,
                 num_frequency_points=None,
                 temperatures=None,
                 frequency_factor_to_THz=VaspToTHz,
                 frequency_scale_factor=None,
                 is_nosym=False,
                 symprec=1e-5,
                 output_filename=None,
                 log_level=0):
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = mesh
        self._fc2 = fc2
        self._nac_params = nac_params
        self._nac_q_direction = nac_q_direction
        self._sigmas = sigmas
        self._cutoff_frequency = cutoff_frequency
        self._frequency_step = frequency_step
        self._num_frequency_points = num_frequency_points
        self._temperatures = temperatures
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._frequency_scale_factor = frequency_scale_factor
        self._is_nosym = is_nosym
        self._symprec = symprec
        self._filename = output_filename
        self._log_level = log_level

        self._jdos = JointDos(
            self._mesh,
            self._primitive,
            self._supercell,
            self._fc2,
            nac_params=self._nac_params,
            nac_q_direction=self._nac_q_direction,
            cutoff_frequency=self._cutoff_frequency,
            frequency_step=self._frequency_step,
            num_frequency_points=self._num_frequency_points,
            temperatures=self._temperatures,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            frequency_scale_factor=self._frequency_scale_factor,
            is_nosym=self._is_nosym,
            symprec=self._symprec,
            filename=output_filename,
            log_level=self._log_level)

    def run(self, grid_points):
        for gp in grid_points:
            self._jdos.set_grid_point(gp)

            if self._log_level:
                weights = self._jdos.get_triplets_at_q()[1]
                print "--------------------------------- Joint DOS ---------------------------------"
                print "Grid point: %d" % gp
                print "Number of ir-triplets:",
                print "%d / %d" % (len(weights), weights.sum())
                adrs = self._jdos.get_grid_address()[gp]
                q = adrs.astype('double') / self._mesh
                print "q-point:", q
                print "Phonon frequency:"
                frequencies = self._jdos.get_phonons()[0]
                print frequencies[gp]

            if self._sigmas:
                for sigma in self._sigmas:
                    if sigma is None:
                        print "Tetrahedron method"
                    else:
                        print "Sigma:", sigma
                    self._jdos.set_sigma(sigma)
                    self._jdos.run()
                    self._write(gp, sigma=sigma)
            else:
                print "sigma or tetrahedron method has to be set."

    def _write(self, gp, sigma=None):
        write_joint_dos(gp,
                        self._mesh,
                        self._jdos.get_frequency_points(),
                        self._jdos.get_joint_dos(),
                        sigma=sigma,
                        temperatures=self._temperatures,
                        filename=self._filename,
                        is_nosym=self._is_nosym)


def get_gruneisen_parameters(fc2,
                             fc3,
                             supercell,
                             primitive,
                             supercell_extra=None,
                             nac_params=None,
                             nac_q_direction=None,
                             ion_clamped=False,
                             factor=None,
                             symprec=1e-5):
    return Gruneisen(fc2,
                     fc3,
                     supercell,
                     primitive,
                     supercell_extra=supercell_extra,
                     nac_params=nac_params,
                     nac_q_direction=nac_q_direction,
                     ion_clamped=ion_clamped,
                     factor=factor,
                     symprec=symprec)


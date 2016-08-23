__author__ = 'xinjiang'
import sys
import numpy as np
from phonopy.phonon.mesh import get_qpoints
from anharmonic.phonon3.triplets import get_grid_address, get_ir_grid_points,reduce_grid_points
from anharmonic.phonon3.interaction import get_dynamical_matrix, set_phonon_c
from anharmonic.phonon3.conductivity_RTA import conductivity_RTA
from anharmonic.phononmd.md import Mdphonon
from anharmonic.file_IO import write_kappa_to_hdf5
from anharmonic.file_IO import read_gamma_from_hdf5,read_kappa_from_hdf5, write_damping_functions, write_linewidth, \
    write_frequency_shift,write_linewidth_band_csv, write_kappa_to_tecplot_BZ
from phonopy.units import VaspToTHz

class Phonompy:
    def __init__(self,
                 supercell,
                 primitive,
                 band_indices=None,
                 frequency_factor_to_THz=None,
                 is_tsym=True,
                 is_nosym=False,
                 symprec=1e-5,
                 log_level=0,
                 temperature=None,
                 lapack_zheev_uplo='L'):

        self._supercell = supercell
        self._primitive = primitive
        if band_indices is None:
            self._band_indices = [np.arange(primitive.get_number_of_atoms() * 3)]
        else:
            self._band_indices = band_indices
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_nosym = is_nosym
        self._is_tsym = is_tsym
        self._symprec = symprec
        self._temperature = temperature
        self._log_level = log_level
        self._step=0
        self._kappa = None
        self._gamma = None
        self._br = None
        self._dm = None
        self._nac_q_direction = None
        self._qpoints = None
        self._band_path = None
        self._lapack_zheev_uplo = lapack_zheev_uplo

        self._band_indices_flatten = np.intc(
            [x for bi in self._band_indices for x in bi])
        # self._mdphonon = Mdphonon(primitive = self._primitive,
        #                           equipos = self._equipos,
        #                           dim=dimension,
        #                           corcut = corcut,
        #                           time_step = time_step,
        #                           nstep = nstep,
        #                           ncormax = ncormax)

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             nac_q_direction=None,
                             frequency_scale_factor=None):
        self._dm = get_dynamical_matrix(fc2,
                                        supercell,
                                        primitive,
                                        nac_params=nac_params,
                                        frequency_scale_factor=frequency_scale_factor,
                                        symprec=self._symprec)
        if nac_q_direction is not None:
            self._nac_q_direction = np.double(nac_q_direction)


    def get_linewidth(self, filename=None):
        temperatures = self._temperature
        self.set_phonons()
        self.lw = np.zeros((len(self._qpoints), len(self._band_indices_flatten)), dtype="double")
        for qpoint in self._qpoints:
            lw = self.get_linewidth_at_q(qpoint, temperatures)

        # write_linewidth_band_csv(band_paths,
        #                          distances=self._distances,
        #                          lws=self._lws_on_path,
        #                          band_indices=self._band_indices,
        #                          temperatures=temperatures,
        #                          frequencies=self._freqs_on_path,
        #                          is_nu=is_nu,
        #                          filename="lw_band-sigma%.2f.csv"%sigma)
        # self.plot_lw()

    def set_phonons(self):
        num_freqs = len(self._primitive.get_masses()) * 3
        self._frequencies = np.zeros((len(self._qpoints), num_freqs), dtype="double")
        self._eigenvectors = np.zeros((len(self._qpoints), num_freqs, num_freqs), dtype='complex128')
        for i,q in enumerate(self._qpoints):
            self._frequencies[i], self._eigenvectors[i] = self.get_phonons(i, is_out_eigenvector=True)

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

    def get_linewidth_at_q(self, qpoint, temperature):

        freq, eigv = self.get_phonons(qpoint)
        # freqs_on_path[p]=self._interaction._frequencies[grid][self._band_indices_flatten]
        # gamma = np.zeros((len(temperatures),
        #                   len(self._band_indices_flatten)),
        #                  dtype='double')
        # for i, t in enumerate(temperatures):
        #     ise.set_temperature(t)
        #     ise.run()
        #     gamma[i] = ise.get_imag_self_energy()
        #
        # for i, bi in enumerate(self._band_indices):
        #     pos = 0
        #     for j in range(i):
        #         pos += len(self._band_indices[j])
        #     lw_on_path[p,:,i]=gamma[:, pos:(pos+len(bi))].sum(axis=1) * 2 / len(bi)
        # return (distances, freqs_on_path, lw_on_path)

    def get_phonons(self, qpoint_index, is_out_eigenvector=False):

        if  is_out_eigenvector:
            return self._frequencies[qpoint_index], self._eigenvectors[qpoint_index]
        else:
            return self._frequencies[qpoint_index]



    def get_thermal_conductivity(self,
                                 sigmas=[0.1],
                                 t_max=1500,
                                 t_min=0,
                                 t_step=10,
                                 grid_points=None,
                                 mesh_divisors=None,
                                 coarse_mesh_shifts=None,
                                 cutoff_lifetime=1e-4, # in second
                                 diff_gamma = 1e-5, # in THz
                                 is_nu=False,
                                 no_kappa_stars=False,
                                 gv_delta_q=1e-4, # for group velocity
                                 write_gamma=False,
                                 read_gamma=False,
                                 write_amplitude=False,
                                 read_amplitude=False,
                                 filename=None):
        br = conductivity_RTA(self._interaction,
                              sigmas=sigmas,
                              asigma_step= self._adaptive_sigma_step,
                              t_max=t_max,
                              t_min=t_min,
                              t_step=t_step,
                              mesh_divisors=mesh_divisors,
                              coarse_mesh_shifts=coarse_mesh_shifts,
                              cutoff_lifetime=cutoff_lifetime,
                              diff_kappa= diff_gamma,
                              nu=is_nu,
                              no_kappa_stars=no_kappa_stars,
                              gv_delta_q=gv_delta_q,
                              log_level=self._log_level,
                              write_tecplot = self._write_tecplot,
                              filename=filename)
        br.set_grid_points(grid_points)

        if read_gamma:
            gamma = []
            for sigma in sigmas:
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
                    br.set_gamma(np.double(gamma))
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
                    br.set_gamma(np.double(gamma))
        br.calculate_kappa(write_amplitude=write_amplitude,
                           read_amplitude=read_amplitude,
                           write_gamma=write_gamma)
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

    def set_grid_points(self, grid_points, mesh, is_gamma_center = True, mesh_shift=[False, False, False]):
        if grid_points is not None: # Specify grid points
            q_nosym, w_nosym = get_qpoints(mesh,
                                  self._primitive,
                                  grid_shift=mesh_shift,
                                  is_time_reversal=False,
                                  is_gamma_center=is_gamma_center,
                                  symprec=self._symprec,
                                  is_symmetry= False)
            self._qpoints = q_nosym[grid_points]
        else: # Automatic sampling
            self._qpoints, self._weights = get_qpoints(mesh,
                                                       self._primitive,
                                                       grid_shift=mesh_shift,
                                                       is_time_reversal=False,
                                                       is_gamma_center=is_gamma_center,
                                                       symprec=self._symprec,
                                                       is_symmetry= (not self._is_nosym))

    def set_qpoints(self, qpoints):
        self._qpoints=qpoints

    def set_band_path(self, band_paths):
        print "Paths in reciprocal reduced coordinates:"
        for path in band_paths:
            print "[%5.2f %5.2f %5.2f] --> [%5.2f %5.2f %5.2f]" % (tuple(path[0]) + tuple(path[-1]))
        self._band_path = band_paths
        self._qpoints = sum(band_paths, [])

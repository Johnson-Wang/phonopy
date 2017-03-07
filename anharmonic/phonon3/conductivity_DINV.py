import numpy as np
import sys
from phonopy.units import THz, kb_J, Angstrom, THzToEv, EV, total_time
from anharmonic.phonon3.collision import Collision
from anharmonic.phonon3.conductivity import Conductivity
from anharmonic.phonon3.triplets import get_grid_points_by_rotations
unit_to_WmK = kb_J / Angstrom ** 3 * ((THz * Angstrom) * THzToEv * EV / kb_J) ** 2  / THz/ (2 * np.pi) # 2pi comes from the definition of tau
np.seterr(divide="ignore")
class conductivity_DINV(Conductivity):
    def __init__(self,
                 interaction,
                 symmetry=None,
                 sigmas=[0.2],
                 grid_points=None,
                 temperatures=None,
                 adaptive_sigma_step = 1,
                 no_kappa_stars=False,
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
        self._allocate_values()
        if filename is not None:
            self._filename = filename
        else:
            self._filename = "dinv"

    def run(self):
        for i in range(len(self._sigmas)):
            self.set_sigma(i)
            for j in range(len(self._temperatures)):
                total_time.reset()
                self.set_temperature(j)
                self.set_collision()
                self.run_at_sigma_and_temp()
                total_time.output()


    @total_time.timeit
    def set_collision(self):
        self._collision.set_temperature(temperature=self._temp)
        self._collision.set_sigma(self._sigma)
        self._collision.set_grids(self._grid_points)
        if self._collision.get_read_collision() and not self._collision.get_is_dispersed():
            self._collision.read_collision_all(log_level=self._log_level, is_adaptive_sigma=self._is_adaptive_sigma)

    def _allocate_values(self):
        self.set_rot_grid_points()
        nqpoint, nband = self._frequencies.shape
        num_mesh_points = np.prod(self._mesh)
        self._collision_total = np.zeros((len(self._sigmas),
                                          len(self._temperatures),
                                          num_mesh_points,
                                          nband,
                                          num_mesh_points,
                                          nband), dtype='double')
        self._gv = np.zeros((len(self._grid_points),nband,3), dtype='double')
        self._F= np.zeros((len(self._sigmas), num_mesh_points, len(self._temperatures), nband, 3), dtype="double")
        self._b = np.zeros((num_mesh_points, len(self._temperatures), nband, 3), dtype="double")
        self._kappa = np.zeros((len(self._sigmas), num_mesh_points, len(self._temperatures), nband, 6), dtype="double")

    def get_mode_heat_capacities(self):
        cv = []
        for i, g in enumerate(self._grid_points):
            f = self._frequencies[i]
            cv.append(self._get_cv(f))
        return np.array(cv, dtype="double")

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

    @total_time.set_main
    @total_time.timeit
    def run_at_sigma_and_temp(self): # sigma and temperature
        "Run each single iteration, all terminology follows the description in wiki/Conjugate_gradient_method"
        if self._log_level:
            if self._sigma:
                print "######Perturbation flow for the next iterative step at sigma=%s, t=%f#######"\
                      %(self._sigma, self._temp)
            else:
                print "######Perturbation flow for the next iterative step with tetrahedron method#######"
            self.print_calculation_progress_header()
        nqpoint, nband = self._frequencies.shape
        num_mesh_points = np.prod(self._mesh)
        for i, ir_gp in enumerate(self._ir_grid_points):

            collision_total = np.zeros((num_mesh_points, nband, nband), dtype='double')
            self._collision.calculate_collision(ir_gp)
            col = self._collision._collision_in
            _, gp2tp = np.unique(self._pp._second_mappings[i], return_inverse=True)
            for j, index in enumerate(gp2tp):
                collision_total[j] = col[index]
                if ir_gp == j:
                    collision_total[j] += np.diag(self._collision._collision_out)
            multi = (self._rot_grid_points[:, ir_gp] == ir_gp).sum()
            self._set_gv(i)
            n = self._collision.get_occupation()[ir_gp, self._itemp]
            freqs = self._frequencies[i]
            nn1 = n * (n + 1)
            fnn1 = freqs * nn1
            b = fnn1[:,np.newaxis] * self._gv[i] / self._temp ** 2
            for j, r in enumerate(self._rotations_cartesian):
                gp_r = self._rot_grid_points[j, ir_gp]
                self._collision_total[self._isigma, self._itemp, gp_r, :, self._rot_grid_points[j]] += \
                    collision_total / multi
                self._b[gp_r, self._itemp] += np.dot(b, r.T) / multi

                # for k in range(num_mesh_points):
                #     colmat_elem = collision_total[k]
                #     colmat_elem = colmat_elem.copy() / multi
                #     gp_c = self._rot_grid_points[j, k]
                #     self._collision_total[self._isigma, self._itemp, gp_r, :, gp_c, :] += colmat_elem
            self.print_calculation_progress(i)



    def set_kappa(self):
        for i,s in enumerate(self._sigmas):
            self.set_kappa_at_sigma(i)

    def set_equivalent_gamma_at_sigma(self,i):
        gv_norm=np.sum(self._gv**2, axis=-1)
        for j, t in enumerate(self._temperatures):
            FdotvT2 = 2 * np.sum(self._F[i,:,j] * self._gv, axis=-1) * t ** 2
            self._gamma[i,:,j] = self._frequencies * gv_norm * np.where(np.abs(FdotvT2)>0, 1 / FdotvT2, 0)

    def set_kappa_at_sigma(self,s):
        self.set_kappa_at_s_c(s)

    def set_kappa_at_s_c(self, s):
        kappa = self._kappa[s]
        num_mesh_point = np.prod(self._mesh)
        num_band = self._frequencies.shape[-1]
        for t, temp in enumerate(self._temperatures):
            self._set_inv_reducible_collision_matrix(s, t)
            inv_collision = self._collision_total[s,t].reshape(num_mesh_point, num_band, num_mesh_point*num_band)
            # collision = self._collision_total[s, t].swapaxes(0,1).swapaxes(2,3)
            # np.savetxt("invcollision.dat", collision.reshape(-1, num_band*num_mesh_point))
            self._F[s, :, t] = np.dot(inv_collision, self._b[:, t].reshape(-1,3))
            for i in range(3):
                for j in range(i, 3):
                    direct = i if i==j else (j-i) + 2 * (i + 1)
                    bf = self._b[:, t,:, i] * self._F[s, :, t, :, j]
                    if i != j:
                        bf2 = self._b[:, t, :, j] * self._F[s, :, t, :, i]
                        bf = (bf + bf2) / 2
                    kappa[:, t,:, direct] = bf * temp ** 2 * self._kappa_factor / np.prod(self._mesh)


    def _set_inv_reducible_collision_matrix(self, i_sigma, i_temp):
        num_mesh_points = np.prod(self._mesh)
        num_band = self._primitive.get_number_of_atoms() * 3
        col_mat = self._collision_total[i_sigma, i_temp].reshape(
            num_mesh_points * num_band, num_mesh_points * num_band)
        w, col_mat[:] = np.linalg.eigh(col_mat)
        v = col_mat
        e = np.zeros(len(w), dtype='double')
        for l, val in enumerate(w):
            if val > 1e-6:
                e[l] = 1 / np.sqrt(val)
        v[:] = e * v
        v[:] = np.dot(v, v.T) # inv_col
        self._collision_eigenvalues = w
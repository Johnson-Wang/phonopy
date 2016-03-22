import os
import sys
import numpy as np
def read_xyz_file(filename,num_atom, format = "d"):
    if not os.path.exists(filename):
        print "Error! the file %s does not exist" %filename
        sys.exit(1)
    if format == "d":
        line_extra = 2    # number of lines which contain extra information
        cv_pos = [1, 2, 3, 5, 6, 7] # coordinates and velocity position in a line
    elif format == "l":
        line_extra = 9
        cv_pos = [0,1,2, 3, 4, 5]
    with open(filename) as f:
        num_lines = sum(1 for lines in f)
    lines_per_step = num_atom + line_extra
    num_steps = num_lines / lines_per_step
    xyz = np.zeros((num_steps, num_atom, 6), dtype="double")
    with open(filename) as f:
        for i, line in enumerate(f):
            if i % lines_per_step  < line_extra:
                pass
            else:
                step = i / lines_per_step
                atom = i % lines_per_step - line_extra
                xyz[step, atom] = np.array(line.strip().split())[cv_pos].astype(np.double)
    return xyz


class Mdphonon():
    def __init__(self,
                 primitive,
                 equipos,
                 dim,
                 corcut,
                 time_step,
                 nstep,
                 ncormax,
                 frequency=None,
                 eigv = None,
                 format="d"):
        self._primitive = primitive
        self._equipos = equipos
        self._natom_s = len(equipos.get_masses())
        self._dim = dim
        self._corcut = corcut,
        self._dtime = time_step
        self._nstep = nstep
        self._ncormax = ncormax
        self._freq = frequency
        self._eigv = eigv
        self._format = format
        self._pos = None
        self._vel = None

    def set_pos_vel(self, filename):
        xyz = read_xyz_file(filename, num_atom= self._natom_s, format = self._format)
        len_t = len(xyz)
        if self._nstep > len_t:
            print "Error the sample size is larger than the total number of steps!"
            sys.exit(1)
        self._num_cor = len_t / self._nstep
        xyz = xyz[:self._num_cor*self._nstep].reshape(self._num_cor, self._nstep, self._natom_s, 6)
        self._pos = xyz[:,:,0:3]
        self._vel = xyz[:,:,3:5]


    def set_freq_eigv(self, freq = None, eigv = None):
        self._freq = freq
        self._eigv = eigv





if __name__== "__main__":
    xyz = read_xyz_file("geo_end.xyz", num_atom = 64, format="d")

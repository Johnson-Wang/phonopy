# Copyright (C) 2011 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from phonopy.structure.symmetry import get_ir_reciprocal_mesh
from phonopy.units import VaspToTHz
from phonopy.phonon.irreps import IrReps
from phonopy.structure.grid_points import GridPoints
from phonopy.structure.symmetry import Symmetry

def get_qpoints(mesh_numbers,
                cell,
                grid_shift=None,
                is_gamma_center=True,
                is_time_reversal=True,
                symprec=1e-5,
                is_symmetry=True):
    mesh = np.array(mesh_numbers)
    if grid_shift == None:
        shift = np.zeros(3, dtype='double')
    else:
        shift = np.array(grid_shift)

    diffby2 = np.abs(shift * 2 - np.rint(shift * 2))
    if (diffby2 < symprec).all() and is_symmetry: # No shift or half shift case
        diff = np.abs(shift - np.rint(shift))
        if is_gamma_center:
            return _get_qpoint_symmetry(mesh,
                                        (diff > symprec),
                                        cell,
                                        is_time_reversal,
                                        symprec)
        else: # Monkhorst-pack
            return _get_qpoint_symmetry(mesh,
                                        np.logical_xor((diff > symprec),
                                                       (mesh % 2 == 0)),
                                        cell,
                                        is_time_reversal,
                                        symprec)
    else:
        return _get_qpoint_no_symmetry(mesh, shift,is_gamma_center)

def estimate_band_connection(prev_eigvecs, eigvecs, prev_band_order, degenerate=None, threshold=0.99):
    if degenerate == None:
        degenerate=[[i] for i in range(len(prev_band_order))]

    metric = np.abs(np.dot(prev_eigvecs.conjugate().T, eigvecs))
    connection_order = []
    for bi in degenerate:
        overlaps = metric[:, bi]
        overlaps = np.sqrt(np.sum(overlaps ** 2, axis=1))
        max_val = np.sort(overlaps)[-len(bi):]
        if (max_val < threshold).any():
            return None
        max_index = np.argsort(overlaps)[-len(bi):].tolist()
        connection_order += max_index
    band_order = np.arange(len(connection_order))
    connection_order = np.array(connection_order, dtype=np.int)
    # order_raw = prev_band_order[connection_order]
    for bi in degenerate:
        x = prev_band_order[bi]
        band_order[bi] = np.sort(connection_order[x])
        # band_order[bi] = np.sort(order_raw[bi])
    return band_order

def _get_qpoint_symmetry(mesh,
                         is_shift,
                         cell,
                         is_time_reversal,
                         symprec):
    mapping, grid = get_ir_reciprocal_mesh(mesh,
                                           cell,
                                           is_shift * 1,
                                           is_time_reversal,
                                           symprec)
    ir_list = np.unique(mapping)
    weights = np.zeros(ir_list.shape[0], dtype='intc')
    qpoints = np.zeros((ir_list.shape[0], 3), dtype='double')

    for i, g in enumerate(ir_list):
        weights[i] = np.sum(mapping == g)
        qpoints[i] = (grid[g] + is_shift * 0.5) / mesh
        qpoints[i] -= (qpoints[i] > 0.5) * 1

    return qpoints, weights

def get_degenerate_sets(freqs, degeneracy_tolerance=1e-5):
    degenerates = []
    indices_done = []
    for i, f1 in enumerate(freqs):
        if i in indices_done:
            continue
        deg_set = []
        for j, f2 in enumerate(freqs):
            if abs(f2 - f1) < degeneracy_tolerance:
                deg_set.append(j)
                indices_done.append(j)
        degenerates.append(deg_set)
    return degenerates

def _get_qpoint_no_symmetry(mesh, shift,is_gamma_center=True):
    qpoints = []
    qshift = shift / mesh
    for i in (0, 1, 2):
        if mesh[i] % 2 == 0 and not is_gamma_center:
            qshift[i] += 0.5 / mesh[i]
            
    for i in range(mesh[2]):
        for j in range(mesh[1]):
            for k in range(mesh[0]):
                q = np.array([float(k) / mesh[0],
                              float(j) / mesh[1],
                              float(i) / mesh[2]]) + qshift
                qpoints.append(np.array([q[0] - (q[0] > 0.5),
                                         q[1] - (q[1] > 0.5),
                                         q[2] - (q[2] > 0.5)]))

    qpoints = np.array(qpoints, dtype='double')
    weights = np.ones(qpoints.shape[0], dtype='intc')

    return qpoints, weights


class Mesh:
    def __init__(self,
                 dynamical_matrix,
                 cell,
                 mesh,
                 shift=None,
                 is_time_reversal=False,
                 is_mesh_symmetry=True,
                 is_eigenvectors=False,
                 is_band_connection=False,
                 is_gamma_center=False,
                 group_velocity=None,
                 factor=VaspToTHz,
                 symprec=1e-5):
        self._mesh = np.array(mesh)
        self._is_band_connection=is_band_connection
        self._band_order = None
        self._is_eigenvectors = is_eigenvectors
        self._factor = factor
        self._cell = cell
        self._dynamical_matrix = dynamical_matrix
        # self._qpoints, self._weights = get_qpoints(self._mesh,
        #                                            self._cell,
        #                                            shift,
        #                                            is_gamma_center,
        #                                            is_time_reversal,
        #                                            symprec,
        #                                            is_symmetry)
        rotations  = Symmetry(self._cell).get_pointgroup_operations()
        self._gp = GridPoints(self._mesh,
                              np.linalg.inv(self._cell.get_cell()),
                              q_mesh_shift=shift,
                              is_gamma_center=is_gamma_center,
                              is_time_reversal=is_time_reversal,
                              rotations=rotations,
                              is_mesh_symmetry=is_mesh_symmetry)
        self._qpoints = self._gp.get_ir_qpoints()
        self._weights = self._gp.get_ir_grid_weights()

        self._eigenvalues = None
        self._eigenvectors = None
        self._set_eigenvalues()
        self._set_band_connection()

        self._group_velocities = None
        if group_velocity is not None:
            self._set_group_velocities(group_velocity)

    def get_dynamical_matrix(self):
        return self._dynamical_matrix

    def get_mesh_numbers(self):
        return self._mesh

    def get_qpoints(self):
        return self._qpoints

    def get_weights(self):
        return self._weights

    def get_grid_address(self):
        return self._gp.get_grid_address()

    def get_ir_grid_points(self):
        return self._gp.get_ir_grid_points()

    def get_grid_mapping_table(self):
        return self._gp.get_grid_mapping_table()

    def get_eigenvalues(self):
        return self._eigenvalues

    def get_frequencies(self):
        return self._frequencies

    def get_group_velocities(self):
        return self._group_velocities

    def get_eigenvectors(self):
        """
        Eigenvectors is a numpy array of three dimension.
        The first index runs through q-points.
        In the second and third indices, eigenvectors obtained
        using numpy.linalg.eigh are stored.
        
        The third index corresponds to the eigenvalue's index.
        The second index is for atoms [x1, y1, z1, x2, y2, z2, ...].
        """
        return self._eigenvectors

    def write_hdf5(self):
        import h5py
        f=h5py.File('mesh.hdf5', 'w')
        eigenvalues = self._eigenvalues
        natom = self._cell.get_number_of_atoms()
        f.create_dataset('mesh', data=self._mesh)
        f.create_dataset('nqpoint',data=self._qpoints.shape[0])
        f.create_dataset('q-position',data=self._qpoints)
        f.create_dataset('natom',data=natom)
        f.create_dataset('weight', data=self._weights)
        f.create_dataset('frequency',data=self._factor*
                                          np.sign(eigenvalues)*
                                          np.sqrt(np.abs(eigenvalues)))
        if self._group_velocities is not None:
            f.create_dataset('group_velocity', data=self._group_velocities)
        if self._is_eigenvectors:
            f.create_dataset('eigenvector_r',data=self._eigenvectors.real)
            f.create_dataset('eigenvector_i',data=self._eigenvectors.imag)
        if self._is_band_connection:
            f.create_dataset('band_order', data=self._band_order)

    def write_yaml(self):
        f = open('mesh.yaml', 'w')
        eigenvalues = self._eigenvalues
        natom = self._cell.get_number_of_atoms()
        f.write("mesh: [ %5d, %5d, %5d ]\n" % tuple(self._mesh))
        f.write("nqpoint: %-7d\n" % self._qpoints.shape[0])
        f.write("natom:   %-7d\n" % natom)
        f.write("phonon:\n")

        for i, q in enumerate(self._qpoints):
            f.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % tuple(q))
            f.write("  weight: %-5d\n" % self._weights[i])
            f.write("  band:\n")

            for j, eig in enumerate(eigenvalues[i]):
                f.write("  - # %d\n" % (j+1))
                if eig < 0:
                    freq = -np.sqrt(-eig)
                else:
                    freq = np.sqrt(eig)
                f.write("    frequency:  %15.10f\n" % (freq * self._factor))

                if self._group_velocities is not None:
                    f.write("    group_velocity: ")
                    f.write("[ %13.7f, %13.7f, %13.7f ]\n" %
                            tuple(self._group_velocities[i, j]))

                if self._band_order is not None:
                    f.write("    band_order: %-5d\n"%self._band_order[i, j])

                if self._is_eigenvectors:
                    f.write("    eigenvector:\n")
                    for k in range(natom):
                        f.write("    - # atom %d\n" % (k+1))
                        for l in (0,1,2):
                            f.write("      - [ %17.14f, %17.14f ]\n" %
                                    (self._eigenvectors[i,k*3+l,j].real,
                                     self._eigenvectors[i,k*3+l,j].imag))
            f.write("\n")

    def _set_eigenvalues(self):
        eigs = []
        vecs = []
        for q in self._qpoints:
            self._dynamical_matrix.set_dynamical_matrix(q)
            dm = self._dynamical_matrix.get_dynamical_matrix()

            if self._is_eigenvectors:
                val, vec = np.linalg.eigh(dm)
                eigs.append(val.real)
                vecs.append(vec)
            else:
                eigs.append(np.linalg.eigvalsh(dm).real)

        self._eigenvalues = np.array(eigs)
        if self._is_eigenvectors:
            self._eigenvectors = np.array(vecs)

        self._set_frequencies()

    def _set_band_connection(self):
        nband = self._cell.get_number_of_atoms() * 3
        nqpoint = len(self._qpoints)
        qpoints = np.dot(np.linalg.inv(self._cell.get_cell()), self._qpoints.T).T
        band_order = np.zeros((nqpoint, nband), dtype="intc")

        is_degenerate = np.zeros(len(self._eigenvalues), dtype="bool")

        degenerate = []
        for i, freq in  enumerate(self._frequencies):
            deg = get_degenerate_sets(freq, 1e-5)
            degenerate.append(deg)
            if len(deg) != nband:
                is_degenerate[i] = True

        nodegq = qpoints[np.where(is_degenerate == False)]
        indicesn = np.where(is_degenerate == False)[0]
        done = np.zeros(len(indicesn), dtype="bool")
        indicesd = np.where(is_degenerate)[0]
        qpts_dist = np.sum(nodegq ** 2, axis=-1)
        start_q = np.argmin(qpts_dist)
        i=start_q
        band_order[indicesn[start_q]] = np.arange(nband)
        while True:
            done[i]=True
            is_break = True
            for j in np.argsort(np.sum((nodegq-nodegq[i])**2, axis=-1)):
                if not done[j]:
                    is_break = False
                    break
            if is_break:
                break
            for i in np.argsort(np.sum((nodegq-nodegq[j])**2, axis=-1)):
                if done[i]:
                    break
            bi=indicesn[i]
            bj = indicesn[j]
            bo = self._estimate_band_connection(self._qpoints[bi], self._qpoints[bj], band_order[bi])
            band_order[bj] = bo
            i=j
        for id in indicesd:
            j = np.argmin(np.sum((nodegq-qpoints[id])**2, axis=-1))
            ij = indicesn[j]
            if (self._qpoints[id] == np.array([0.48, 0.48, 0])).all():
                self._estimate_band_connection(np.array([0.2, 0.2, 0]), np.array([0.48, 0.48, 0]), np.array([0,1,2,3,4,5]))
            bo = self._estimate_band_connection(self._qpoints[ij],
                                          self._qpoints[id],
                                          band_order[ij])
            band_order[id] = bo
        self._band_order = band_order


    def _estimate_band_connection(self, p, n, band_order_pre):
        self._dynamical_matrix.set_dynamical_matrix(p)
        e1, ev1 = np.linalg.eigh(self._dynamical_matrix.get_dynamical_matrix())
        fre1 = np.array(np.sqrt(abs(e1)) * np.sign(e1)) * self._factor
        self._dynamical_matrix.set_dynamical_matrix(n)
        e2, ev2 = np.linalg.eigh(self._dynamical_matrix.get_dynamical_matrix())
        fre2 = np.array(np.sqrt(abs(e2)) * np.sign(e2)) * self._factor
        deg1 = get_degenerate_sets(fre1, 1e-5)
        deg2 = get_degenerate_sets(fre2, 1e-5)
        assert len(deg1) >= len(deg2)
        mid_point = p + (n-p) * (0.5 +np.random.random() / 10)
        self._dynamical_matrix.set_dynamical_matrix(mid_point)
        evv, evm = np.linalg.eigh(self._dynamical_matrix.get_dynamical_matrix())
        frem = np.array(np.sqrt(abs(evv)) * np.sign(evv)) * self._factor
        degeneratem = get_degenerate_sets(frem, 1e-7)
        assert len(degeneratem) == len(deg1)
        b1 = estimate_band_connection(ev1, evm, band_order_pre, degeneratem, 0.9)
        if b1 is None:
            b1 = self._estimate_band_connection(p, mid_point, band_order_pre)
        b2 = estimate_band_connection(evm, ev2, b1, deg2, 0.9)
        if b2 is not None:
            return b2
        else:
            return self._estimate_band_connection(mid_point, n, b1)


    def _set_frequencies(self):
        ## This expression works only python >= 2.5
        #  frequencies = []
        # for eigs in self._eigenvalues:
        #     frequencies.append(
        #         [np.sqrt(x) if x > 0 else -np.sqrt(-x) for x in eigs])
        
        self._frequencies = np.array(np.sqrt(abs(self._eigenvalues)) *
                                     np.sign(self._eigenvalues)) * self._factor

    def _set_group_velocities(self, group_velocity):
        group_velocity.set_q_points(self._qpoints)
        self._group_velocities = group_velocity.get_group_velocity()

        

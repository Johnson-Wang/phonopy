# Copyright (C) 2013 Atsushi Togo
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
import phonopy.structure.spglib as spg

parallelepiped_vertices = np.array([[0, 0, 0],
                                    [1, 0, 0],
                                    [0, 1, 0],
                                    [1, 1, 0],
                                    [0, 0, 1],
                                    [1, 0, 1],
                                    [0, 1, 1],
                                    [1, 1, 1]], dtype='intc', order='C')
parallelogram_vertices = np.array([[0,0],[0,1],[1,0],[1,1]], dtype="intc", order="C")
segment_vertices = np.array([0, 1], dtype="intc", order="C")

class TriagonalMethod:
    def __init__(self,
                 primitive_vectors,
                 mesh=[1, 1,1]):
        self._mesh = np.array(mesh)
        assert np.count_nonzero(self._mesh>1) == 2
        self._primitive_vectors = np.array(
            primitive_vectors, dtype='double', order='C') / mesh # column vectors
        self._vertices = None
        self._relative_grid_addresses = None
        self._central_indices = None
        self._tetrahedra_omegas = None
        self._sort_indices = None
        self._omegas = None
        self._create_triagonal() # cleave the parallelepiped lattices into 6 individual tetrahedrons
        self._set_relative_grid_addresses()

        self._integration_weight = None

    def get_triagonal(self):
        return self._relative_grid_addresses

    def _create_triagonal(self):
        # 2--------------3
        # |              |
        # |              |
        # |              |
        # |              |
        # |              |
        # |              |
        # 0--------------1
        #
        # i: vec        neighbours
        # 0: O          1, 2
        # 1: a          0, 3
        # 2: b          0, 3
        # 3: a + b      1, 2
        a, b = self._primitive_vectors.T[np.where(self._mesh>1)]
        diag_vecs = np.array([ a + b,  # 0-3
                               a - b]) # 1-2
        shortest_index = np.argmin(np.sum(diag_vecs ** 2, axis=1))
        # vertices = [np.zeros(3), a, b, a + b, c, c + a, c + b, c + a + b]
        if shortest_index == 0:
            triag = np.sort([[0, 3, x] for x in (1,2)])
        elif shortest_index == 1:
            triag = np.sort([[1, 2, x] for x in (0,3)])
        else:
            assert False
        self._vertices = triag

    def _set_relative_grid_addresses(self):
        relative_grid_addresses = np.zeros((6, 3, 2), dtype='intc')
        central_indices = np.zeros(6, dtype='intc') # 6 corresponds to all the vertices in the 2 triagonals
        pos = 0
        for i in range(4):
            ppd_shifted = (parallelogram_vertices -
                           parallelogram_vertices[i]) # relative position of the parallelogram to the grid i
            for triag in self._vertices:
                if i in triag:
                    central_indices[pos] = np.where(triag==i)[0][0]
                    relative_grid_addresses[pos, :, :] = ppd_shifted[triag]
                    pos += 1
        self._relative_grid_addresses = np.zeros((6,3,3), dtype="intc")
        j=0
        for i in range(3):
            if self._mesh[i] == 1:
                self._relative_grid_addresses[:,:,i] = 0
            else:
                self._relative_grid_addresses[:,:,i] = relative_grid_addresses[:,:,j]
                j+=1
        self._central_indices = central_indices

class SegmentsMethod:
    def __init__(self,
                 primitive_vectors,
                 mesh):
        self._mesh = np.array(mesh)
        assert np.count_nonzero(self._mesh > 1) == 1
        self._primitive_vectors = np.array(
            primitive_vectors, dtype='double', order='C') / mesh # column vectors
        self._vertices = None
        self._relative_grid_addresses = None
        self._central_indices = None
        self._segments_omegas = None
        self._omegas = None
        self._create_segments()
        self._set_relative_grid_addresses()
        self._integration_weight = None

    def get_tetrahedra(self):
        return self._relative_grid_addresses

    def _run_c(self, omegas, value='I'):
        self._integration_weight = spg.get_tetrahedra_integration_weight(
            omegas,
            self._segments_omegas,
            function=value)

    def run(self, omegas, value='I'):
        try:
            # raise ImportError
            import phonopy._phonopy as phonoc
            self._run_c(omegas, value=value)
        except ImportError:
            self._run_py(omegas, value=value)

    def get_unique_tetrahedra_vertices(self):
        unique_vertices = []
        for adrs in self._relative_grid_addresses.reshape(-1, 3):
            found = False
            for uadrs in unique_vertices:
                if (uadrs == adrs).all():
                    found = True
                    break
            if not found:
                unique_vertices.append(adrs)
        return np.array(unique_vertices, dtype='intc', order='C')

    def set_tetrahedra_omegas(self, segments_omegas):
        """
        tetrahedra_omegas: (24, 4) omegas at self._relative_grid_addresses
        """
        self._segments_omegas = segments_omegas

    def get_integration_weight(self):
        return self._integration_weight

    def _run_py(self, omegas, value='I'):
        if isinstance(omegas, float) or isinstance(omegas, int):
            iw = self._get_integration_weight_py(omegas, value=value)
        else:
            iw = np.zeros(len(omegas), dtype='double')
            for i, omega in enumerate(omegas):
                iw[i] = self._get_integration_weight_py(omega, value=value)
        self._integration_weight = iw

    def _get_integration_weight_py(self, omega, value='I'):
        if value == 'I':
            IJ = self._I
            gn = self._g
        else:
            IJ = self._J
            gn = self._n

        self._sort_indices = np.argsort(self._segments_omegas, axis=1)
        sum_value = 0.0
        self._omega = omega
        for omegas, indices, ci in zip(self._segments_omegas,
                                       self._sort_indices,
                                       self._central_indices):
            self._vertices_omegas = omegas[indices]

            v = self._vertices_omegas
            if (omega < v[0]):
                sum_value += IJ(0, np.where(indices==ci)[0][0]) * gn(0)
            elif (v[0] < omega and omega < v[1]):
                sum_value += IJ(1, np.where(indices==ci)[0][0]) * gn(1)
            elif (v[1] < omega):
                sum_value += IJ(2, np.where(indices==ci)[0][0]) * gn(2)

        return sum_value

    def _create_segments(self):
        # 0--------------1
        #
        self._vertices = [0, 1]

    def _set_relative_grid_addresses(self):
        relative_grid_addresses = np.array([[0, 1], [0, -1]], dtype="intc")
        self._relative_grid_addresses = np.zeros((2,2,3), dtype="intc")
        for i in range(3):
            if not self._mesh[i] == 1:
                self._relative_grid_addresses[:,:,i] = relative_grid_addresses
                break
        self._central_indices = np.array([0,0], dtype="intc")

    def _f(self, n, m):
        return ((self._omega - self._vertices_omegas[m]) /
                (self._vertices_omegas[n] - self._vertices_omegas[m]))

    def _J(self, i, ci):
        if i == 0:
            return self._J_0()
        elif i == 1:
            if ci == 0:
                return self._J_10()
            elif ci == 1:
                return self._J_11()
            else:
                assert False
        elif i == 2:
            return self._J_2()
        else:
            assert False

    def _I(self, i, ci):
        if i == 0:
            return self._I_0()
        elif i == 1:
            if ci == 0:
                return self._I_10()
            elif ci == 1:
                return self._I_11()
            else:
                assert False
        elif i == 2:
            return self._I_2()
        else:
            assert False

    def _n(self, i):
        if i == 0:
            return self._n_0()
        elif i == 1:
            return self._n_1()
        elif i == 2:
            return self._n_2()
        else:
            assert False

    def _g(self, i):
        if i == 0:
            return self._g_0()
        elif i == 1:
            return self._g_1()
        elif i == 2:
            return self._g_2()
        else:
            assert False

    def _n_0(self):
        """omega < omega1"""
        return 0.0

    def _n_1(self):
        """omega1 < omega < omega2"""
        return self._f(1, 0)

    def _n_2(self):
        """omega2 < omega"""
        return 1.0

    def _g_0(self):
        """omega < omega1"""
        return 0.0

    def _g_1(self):
        """omega1 < omega < omega2"""
        return self._n_1() / (self._omega - self._vertices_omegas[0])

    def _g_2(self):
        """omega2 < omega"""
        return 0.0

    def _J_0(self):
        return 0.0

    def _J_10(self):
        return (1.0 + self._f(0, 1)) / 2

    def _J_11(self):
        return self._f(1, 0) / 2

    def _J_2(self):
        return 0.5

    def _I_0(self):
        return 0.0

    def _I_10(self):
        return self._f(0, 1)

    def _I_11(self):
        return self._f(1, 0)

    def _I_2(self):
        return 0.0

class TetrahedronMethod:
    def __init__(self,
                 primitive_vectors,
                 mesh=[1, 1, 1]):
        self._primitive_vectors = np.array(
            primitive_vectors, dtype='double', order='C') / mesh # column vectors
        self._vertices = None
        self._relative_grid_addresses = None
        self._central_indices = None
        self._tetrahedra_omegas = None
        self._sort_indices = None
        self._omegas = None
        self._create_tetrahedra() # cleave the parallelepiped lattices into 6 individual tetrahedrons
        self._set_relative_grid_addresses()
        self._integration_weight = None

    def run(self, omegas, value='I'):
        try:
            import phonopy._phonopy as phonoc
            self._run_c(omegas, value=value)
        except ImportError:
            self._run_py(omegas, value=value)

    def get_tetrahedra(self):
        """
        Returns relative grid addresses at vertices of tetrahedra
        """
        return self._relative_grid_addresses

    def get_unique_tetrahedra_vertices(self):
        unique_vertices = []
        for adrs in self._relative_grid_addresses.reshape(-1, 3):
            found = False
            for uadrs in unique_vertices:
                if (uadrs == adrs).all():
                    found = True
                    break
            if not found:
                unique_vertices.append(adrs)
        return np.array(unique_vertices, dtype='intc', order='C')
    
    def set_tetrahedra_omegas(self, tetrahedra_omegas):
        """
        tetrahedra_omegas: (24, 4) omegas at self._relative_grid_addresses
        """
        self._tetrahedra_omegas = tetrahedra_omegas

    def get_integration_weight(self):
        return self._integration_weight

    def _run_c(self, omegas, value='I'):
        self._integration_weight = spg.get_tetrahedra_integration_weight(
            omegas,
            self._tetrahedra_omegas,
            function=value)

    def _run_py(self, omegas, value='I'):
        if isinstance(omegas, float) or isinstance(omegas, int):
            iw = self._get_integration_weight_py(omegas, value=value)
        else:
            iw = np.zeros(len(omegas), dtype='double')
            for i, omega in enumerate(omegas):
                iw[i] = self._get_integration_weight_py(omega, value=value)
        self._integration_weight = iw

    def _get_integration_weight_py(self, omega, value='I'):
        if value == 'I':
            IJ = self._I
            gn = self._g
        else:
            IJ = self._J
            gn = self._n

        self._sort_indices = np.argsort(self._tetrahedra_omegas, axis=1)
        sum_value = 0.0
        self._omega = omega
        for omegas, indices, ci in zip(self._tetrahedra_omegas,
                                       self._sort_indices,
                                       self._central_indices):
            self._vertices_omegas = omegas[indices]
            v = self._vertices_omegas
            if (omega < v[0]):
                sum_value += IJ(0, np.where(indices==ci)[0][0]) * gn(0)
            elif (v[0] < omega and omega < v[1]):
                sum_value += IJ(1, np.where(indices==ci)[0][0]) * gn(1)
            elif (v[1] < omega and omega < v[2]):
                sum_value += IJ(2, np.where(indices==ci)[0][0]) * gn(2)
            elif (v[2] < omega and omega < v[3]):
                sum_value += IJ(3, np.where(indices==ci)[0][0]) * gn(3)
            elif (v[3] < omega):
                sum_value += IJ(4, np.where(indices==ci)[0][0]) * gn(4)

        return sum_value / 6 # a parallelpipe consists of 6 tetrahedra

    def _create_tetrahedra(self):
        #
        #     6-------7
        #    /|      /|
        #   / |     / |
        #  4-------5  |
        #  |  2----|--3
        #  | /     | /
        #  |/      |/
        #  0-------1
        #
        # i: vec        neighbours
        # 0: O          1, 2, 4    
        # 1: a          0, 3, 5
        # 2: b          0, 3, 6
        # 3: a + b      1, 2, 7
        # 4: c          0, 5, 6
        # 5: c + a      1, 4, 7
        # 6: c + b      2, 4, 7
        # 7: c + a + b  3, 5, 6
        a, b, c = self._primitive_vectors.T
        diag_vecs = np.array([ a + b + c,  # 0-7
                              -a + b + c,  # 1-6
                               a - b + c,  # 2-5
                               a + b - c]) # 3-4
        shortest_index = np.argmin(np.sum(diag_vecs ** 2, axis=1))
        # vertices = [np.zeros(3), a, b, a + b, c, c + a, c + b, c + a + b]
        if shortest_index == 0:
            pairs = ((1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6))
            tetras = np.sort([[0, 7] + list(x) for x in pairs])
        elif shortest_index == 1:
            pairs = ((0, 2), (0, 4), (2, 3), (3, 7), (4, 5), (5, 7))
            tetras = np.sort([[1, 6] + list(x) for x in pairs])
        elif shortest_index == 2:
            pairs = ((0, 1), (0, 4), (1, 3), (3, 7), (4, 6), (6, 7))
            tetras = np.sort([[2, 5] + list(x) for x in pairs])
        elif shortest_index == 3:
            pairs = ((0, 1), (0, 2), (1, 5), (2, 6), (5, 7), (6, 7))
            tetras = np.sort([[3, 4] + list(x) for x in pairs])
        else:
            assert False

        self._vertices = tetras

    def _set_relative_grid_addresses(self):
        try:
            rga = spg.get_tetrahedra_relative_grid_address(
                self._primitive_vectors)
            self._relative_grid_addresses = rga

        except ImportError:
            relative_grid_addresses = np.zeros((24, 4, 3), dtype='intc')
            central_indices = np.zeros(24, dtype='intc') # 24 corresponds to all the vertices in the 6 tetrahedrons
            pos = 0
            for i in range(8): # 8 corners in a parallelpipe
                ppd_shifted = (parallelepiped_vertices -
                               parallelepiped_vertices[i])
                for tetra in self._vertices:
                    if i in tetra:
                        central_indices[pos] = np.where(tetra==i)[0][0]
                        relative_grid_addresses[pos, :, :] = ppd_shifted[tetra]
                        pos += 1
            # To make sure it coincides with the results calculated from C code
            for i in range(24):
                centrals = relative_grid_addresses[i, central_indices[i]].copy()
                relative_grid_addresses[i, central_indices[i]] = relative_grid_addresses[i, 0].copy()
                relative_grid_addresses[i, 0] = centrals
            central_indices = np.zeros(24, dtype='intc')
            self._central_indices = central_indices
            self._relative_grid_addresses = relative_grid_addresses

    def _f(self, n, m):
        return ((self._omega - self._vertices_omegas[m]) /
                (self._vertices_omegas[n] - self._vertices_omegas[m]))

    def _J(self, i, ci):
        if i == 0:
            return self._J_0()
        elif i == 1:
            if ci == 0:
                return self._J_10()
            elif ci == 1:
                return self._J_11()
            elif ci == 2:
                return self._J_12()
            elif ci == 3:
                return self._J_13()
            else:
                assert False
        elif i == 2:
            if ci == 0:
                return self._J_20()
            elif ci == 1:
                return self._J_21()
            elif ci == 2:
                return self._J_22()
            elif ci == 3:
                return self._J_23()
            else:
                assert False
        elif i == 3:
            if ci == 0:
                return self._J_30()
            elif ci == 1:
                return self._J_31()
            elif ci == 2:
                return self._J_32()
            elif ci == 3:
                return self._J_33()
            else:
                assert False
        elif i == 4:
            return self._J_4()
        else:
            assert False

    def _I(self, i, ci):
        if i == 0:
            return self._I_0()
        elif i == 1:
            if ci == 0:
                return self._I_10()
            elif ci == 1:
                return self._I_11()
            elif ci == 2:
                return self._I_12()
            elif ci == 3:
                return self._I_13()
            else:
                assert False
        elif i == 2:
            if ci == 0:
                return self._I_20()
            elif ci == 1:
                return self._I_21()
            elif ci == 2:
                return self._I_22()
            elif ci == 3:
                return self._I_23()
            else:
                assert False
        elif i == 3:
            if ci == 0:
                return self._I_30()
            elif ci == 1:
                return self._I_31()
            elif ci == 2:
                return self._I_32()
            elif ci == 3:
                return self._I_33()
            else:
                assert False
        elif i == 4:
            return self._I_4()
        else:
            assert False

    def _n(self, i):
        if i == 0:
            return self._n_0()
        elif i == 1:
            return self._n_1()
        elif i == 2:
            return self._n_2()
        elif i == 3:
            return self._n_3()
        elif i == 4:
            return self._n_4()
        else:
            assert False

    def _g(self, i):
        if i == 0:
            return self._g_0()
        elif i == 1:
            return self._g_1()
        elif i == 2:
            return self._g_2()
        elif i == 3:
            return self._g_3()
        elif i == 4:
            return self._g_4()
        else:
            assert False
    
    def _n_0(self):
        """omega < omega1"""
        return 0.0

    def _n_1(self):
        """omega1 < omega < omega2"""
        return self._f(1, 0) * self._f(2, 0) * self._f(3, 0)

    def _n_2(self):
        """omega2 < omega < omega3"""
        return (self._f(3, 1) * self._f(2, 1) +
                self._f(3, 0) * self._f(1, 3) * self._f(2, 1) +
                self._f(3, 0) * self._f(2, 0) * self._f(1, 2))
                
    def _n_3(self):
        """omega2 < omega < omega3"""
        return (1.0 - self._f(0, 3) * self._f(1, 3) * self._f(2, 3))

    def _n_4(self):
        """omega4 < omega"""
        return 1.0

    def _g_0(self):
        """omega < omega1"""
        return 0.0

    def _g_1(self):
        """omega1 < omega < omega2"""
        # return 3 * self._n_1() / (self._omega - self._vertices_omegas[0])
        return (3 * self._f(1, 0) * self._f(2, 0) /
                (self._vertices_omegas[3] - self._vertices_omegas[0]))

    def _g_2(self):
        """omega2 < omega < omega3"""
        return 3 / (self._vertices_omegas[3] - self._vertices_omegas[0]) * (
            self._f(1, 2) * self._f(2, 0) +
            self._f(2, 1) * self._f(1, 3))

    def _g_3(self):
        """omega3 < omega < omega4"""
        # return 3 * (1.0 - self._n_3()) / (self._vertices_omegas[3] - self._omega)
        return (3 * self._f(1, 3) * self._f(2, 3) /
                (self._vertices_omegas[3] - self._vertices_omegas[0]))

    def _g_4(self):
        """omega4 < omega"""
        return 0.0

    def _J_0(self):
        return 0.0
    
    def _J_10(self):
        return (1.0 + self._f(0, 1) + self._f(0, 2) + self._f(0, 3)) / 4

    def _J_11(self):
        return self._f(1, 0) / 4

    def _J_12(self):
        return self._f(2, 0) / 4

    def _J_13(self):
        return self._f(3, 0) / 4

    def _J_20(self):
        return (self._f(3, 1) * self._f(2, 1) +
                self._f(3, 0) * self._f(1, 3) * self._f(2, 1) *
                (1.0 + self._f(0, 3)) +
                self._f(3, 0) * self._f(2, 0) * self._f(1, 2) *
                (1.0 + self._f(0, 3) + self._f(0, 2))) / 4 / self._n_2()

    def _J_21(self):
        return (self._f(3, 1) * self._f(2, 1) *
                (1.0 + self._f(1, 3) + self._f(1, 2)) +
                self._f(3, 0) * self._f(1, 3) * self._f(2, 1) *
                (self._f(1, 3) + self._f(1, 2)) +
                self._f(3, 0) * self._f(2, 0) * self._f(1, 2) *
                self._f(1, 2)) / 4 / self._n_2()

    def _J_22(self):
        return (self._f(3, 1) * self._f(2, 1) *
                self._f(2, 1) +
                self._f(3, 0) * self._f(1, 3) * self._f(2, 1) *
                self._f(2, 1) +
                self._f(3, 0) * self._f(2, 0) * self._f(1, 2) *
                (self._f(2, 1) + self._f(2, 0))) / 4 / self._n_2()

    def _J_23(self):
        return (self._f(3, 1) * self._f(2, 1) *
                self._f(3, 1) +
                self._f(3, 0) * self._f(1, 3) * self._f(2, 1) *
                (self._f(3, 1) + self._f(3, 0)) +
                self._f(3, 0) * self._f(2, 0) * self._f(1, 2) *
                self._f(3, 0)) / 4 / self._n_2()

    def _J_30(self):
        return ((1.0 - self._f(0, 3) ** 2 * self._f(1, 3) * self._f(2, 3)) /
                4 / self._n_3())

    def _J_31(self):
        return ((1.0 - self._f(0, 3) * self._f(1, 3) ** 2 * self._f(2, 3)) /
                4 / self._n_3())

    def _J_32(self):
        return ((1.0 + self._f(0, 3) * self._f(1, 3) * self._f(2, 3) ** 2) /
                4 / self._n_3())

    def _J_33(self):
        return ((1.0 - self._f(0, 3) * self._f(1, 3) * self._f(2, 3) *
                 (1.0 + self._f(3, 0) + self._f(3, 1) + self._f(3, 2))) /
                4 / self._n_3())

    def _J_4(self):
        return 0.25

    def _I_0(self):
        return 0.0
    
    def _I_10(self):
        return (self._f(0, 1) + self._f(0, 2) + self._f(0, 3)) / 3

    def _I_11(self):
        return self._f(1, 0) / 3

    def _I_12(self):
        return self._f(2, 0) / 3

    def _I_13(self):
        return self._f(3, 0) / 3

    def _I_20(self):
        return (self._f(0, 3) +
                self._f(0, 2) * self._f(2, 0) * self._f(1, 2) /
                (self._f(1, 2) * self._f(2, 0) + self._f(2, 1) * self._f(1, 3))
                ) / 3

    def _I_21(self):
        return (self._f(1, 2) +
                self._f(1, 3) ** 2 * self._f(2, 1) /
                (self._f(1, 2) * self._f(2, 0) + self._f(2, 1) * self._f(1, 3))
                ) / 3

    def _I_22(self):
        return (self._f(2, 1) +
                self._f(2, 0) ** 2 * self._f(1, 2) /
                (self._f(1, 2) * self._f(2, 0) + self._f(2, 1) * self._f(1, 3))
                ) / 3
                
    def _I_23(self):
        return (self._f(3, 0) +
                self._f(3, 1) * self._f(1, 3) * self._f(2, 1) /
                (self._f(1, 2) * self._f(2, 0) + self._f(2, 1) * self._f(1, 3))
                ) / 3

    def _I_30(self):
        return self._f(0, 3) / 3

    def _I_31(self):
        return self._f(1, 3) / 3

    def _I_32(self):
        return self._f(2, 3) / 3

    def _I_33(self):
        return (self._f(3, 0) + self._f(3, 1) + self._f(3, 2)) / 3

    def _I_4(self):
        return 0.0


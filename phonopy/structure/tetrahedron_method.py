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
# 8 corners
parallelepiped_vertices = np.array([[0, 0, 0],
                                    [1, 0, 0],
                                    [0, 1, 0],
                                    [1, 1, 0],
                                    [0, 0, 1],
                                    [1, 0, 1],
                                    [0, 1, 1],
                                    [1, 1, 1]], dtype='intc', order='C')

parallelepiped_vertices_extended = np.array([v + parallelepiped_vertices for v in parallelepiped_vertices],
                                            dtype="intc", order="C") # shape: [8, 8, 3]

class TetrahedronMethod:
    def __init__(self,
                 primitive_vectors,
                 mesh=[1, 1, 1],
                 grid_order = None,
                 is_linear=True): # False: 3rd correction; True: Linear
        self._primitive_vectors = np.array(
            primitive_vectors, dtype='double', order='C') / mesh # column vectors
        self._mesh = mesh
        if grid_order is not None:
            self._grid_order = grid_order
        else:
            self._grid_order = np.r_[1, np.cumprod(mesh[:-1])]
        self._vertices = None
        self._is_linear = is_linear
        self._relative_grid_addresses = None
        self._central_indices = None
        self._tetrahedra_omegas = None
        self._sort_indices = None
        self._omegas = None
        self._create_tetrahedra() # cleave the parallelepiped lattices into 6 individual tetrahedrons
        self._set_relative_grid_addresses()
        self._set_weight_correction()
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

    def get_relative_tetra_indices(self):
        return self._relative_tetra_indices

    def get_unique_tetrahedra_vertices(self):
        unique_vertices = []
        for adrs in self._relative_grid_addresses.reshape(-1, 3):
            unique_vertices_array = np.array(unique_vertices).reshape(-1,3)
            if len(np.where(np.alltrue(unique_vertices_array == adrs, axis=1))[0]) == 0:
                unique_vertices.append(adrs)
        return np.array(unique_vertices, dtype='intc', order='C')

    def _set_weight_correction(self):
        weight_correction = np.zeros((20, 4), dtype="double")
        if self._is_linear:
            weight_correction[:4] = np.eye(4)
            self._weight_correction = weight_correction

        else:
            weight_correction[:4] = np.array([[1440,    0,   30,    0],
                                              [   0, 1440,    0,   30],
                                              [  30,    0, 1440,    0],
                                              [   0,   30,    0, 1440]])

            weight_correction[4:8] = np.array([[ -38,  -28,   17,    7],
                                                [   7,  -38,  -28,   17],
                                                [  17,    7,  -38,  -28],
                                                [ -28,   17,    7,  -38]])

            weight_correction[8:12] =np.array([[-56,    9,  -46,    9],
                                               [  9,  -56,    9,  -46],
                                               [-46,    9,  -56,    9],
                                               [  9,  -46,    9,  -56]])

            weight_correction[12:16] = np.array([[ -38,    7,   17,  -28],
                                               [ -28,  -38,    7,   17],
                                               [  17,  -28,  -38,    7],
                                               [   7,   17,  -28,  -38]])

            weight_correction[16:20] = np.array([[ -18,  -18,   12,  -18],
                                                [ -18,  -18,  -18,   12],
                                                [  12,  -18,  -18,  -18],
                                                [ -18,   12,  -18,  -18]])

            self._weight_correction = weight_correction /  1260.

    def get_weight_correction(self):
        return self._weight_correction

    def get_center_indices(self):
        return self._central_indices

    def set_tetrahedra_omegas(self, tetrahedra_omegas):
        """
        tetrahedra_omegas: (120, 4) omegas at self._relative_grid_addresses
        """
        self._tetrahedra_omegas = np.dot(tetrahedra_omegas, self._weight_correction)


    def get_integration_weight(self):
        return self._integration_weight

    def _run_c(self, omegas, value='I'):
        integration_weight = spg.get_tetrahedra_integration_weight(
            omegas,
            self._tetrahedra_omegas,
            function=value,
            is_linear=self._is_linear)
        self._integration_weight = integration_weight


    def _run_py(self, omegas, value='I'):
        if isinstance(omegas, float) or isinstance(omegas, int):
            iw = self._get_integration_weight_py(omegas, value=value)
        else:
            iw = np.zeros((len(omegas), 120, 4), dtype="double")
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
        num_tetrahedra = self._tetrahedra_omegas.shape[0]
        sum_value =  np.zeros((num_tetrahedra, 4), dtype="double")
        self._omega = omega
        for i, (omegas, indices) in enumerate(zip(self._tetrahedra_omegas,
                                       self._sort_indices)):
            if self._is_linear and i > 24:
                continue
            self._vertices_omegas = omegas[indices]
            v = self._vertices_omegas
            for j in range(4):
                if (omega < v[0]):
                    sum_value[i,j] += IJ(0, np.where(indices==j)[0][0]) * gn(0)
                elif (v[0] < omega and omega < v[1]):
                    sum_value[i,j] += IJ(1, np.where(indices==j)[0][0]) * gn(1)
                elif (v[1] < omega and omega < v[2]):
                    sum_value[i,j] += IJ(2, np.where(indices==j)[0][0]) * gn(2)
                elif (v[2] < omega and omega < v[3]):
                    sum_value[i,j] += IJ(3, np.where(indices==j)[0][0]) * gn(3)
                elif (v[3] < omega):
                    sum_value[i,j] += IJ(4, np.where(indices==j)[0][0]) * gn(4)
        return sum_value / 6 # a parallelogram consists of 6 tetrahedra

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
        self._main_diagonal_index = shortest_index
        # vertices = [np.zeros(3), a, b, a + b, c, c + a, c + b, c + a + b]
        if shortest_index == 0:
            pairs = ((1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6))
            tetras = np.array([[0] + list(x) + [7] for x in pairs])
        elif shortest_index == 1:
            pairs = ((0, 2), (0, 4), (3, 2), (3, 7), (5, 4), (5, 7))
            tetras = np.array([[1] + list(x) + [6] for x in pairs])
        elif shortest_index == 2:
            pairs = ((0, 1), (0, 4), (3, 1), (3, 7), (6, 4), (6, 7))
            tetras = np.array([[2] + list(x) + [5] for x in pairs])
        elif shortest_index == 3:
            pairs = ((1, 0), (2, 0), (1, 5), (2, 6), (7, 5), (7, 6))
            tetras = np.array([[3] + list(x) + [4] for x in pairs])
        else:
            assert False

        self._vertices = tetras

    def get_main_diagonal_index(self):
        return self._main_diagonal_index

    def _set_relative_grid_addresses(self):
        try:
            raise ImportError
            rga = spg.get_tetrahedra_relative_grid_address(
                self._primitive_vectors)
            self._relative_grid_addresses = rga

        except ImportError:
            relative_grid_addresses = np.zeros((384, 20, 3), dtype='intc')

            central_indices = np.zeros(384, dtype='intc') # 8 x 8 x 6 All possible tetrahedra within the next nearest qpoint
            relative_tetra_indices = np.zeros((384, 4), dtype="intc")
            pos = 0
            for i in range(8): # Octants in a 3 dimensional space
                ppd_shifted = (parallelepiped_vertices_extended -
                               parallelepiped_vertices[i] * 2)
                for j in range(8): # 8 parallelograms in each octant
                    ppd_shifted2 = ppd_shifted[j]
                    ppd_origin = ppd_shifted2[0]

                    for k, tetra in enumerate(self._vertices):
                        relative_grid_addresses[pos, :4] = ppd_shifted2[tetra]
                        relative_tetra_indices[pos] = np.r_[ppd_origin, k]
                        pos += 1


            relative_grid_addresses[:, 4] = 2 * relative_grid_addresses[:,0] - relative_grid_addresses[:,1]
            relative_grid_addresses[:, 5] = 2 * relative_grid_addresses[:,1] - relative_grid_addresses[:,2]
            relative_grid_addresses[:, 6] = 2 * relative_grid_addresses[:,2] - relative_grid_addresses[:,3]
            relative_grid_addresses[:, 7] = 2 * relative_grid_addresses[:,3] - relative_grid_addresses[:,0]

            relative_grid_addresses[:, 8] = 2 * relative_grid_addresses[:,0] - relative_grid_addresses[:,2]
            relative_grid_addresses[:, 9] = 2 * relative_grid_addresses[:,1] - relative_grid_addresses[:,3]
            relative_grid_addresses[:,10] = 2 * relative_grid_addresses[:,2] - relative_grid_addresses[:,0]
            relative_grid_addresses[:,11] = 2 * relative_grid_addresses[:,3] - relative_grid_addresses[:,1]

            relative_grid_addresses[:,12] = 2 * relative_grid_addresses[:,0] - relative_grid_addresses[:,3]
            relative_grid_addresses[:,13] = 2 * relative_grid_addresses[:,1] - relative_grid_addresses[:,0]
            relative_grid_addresses[:,14] = 2 * relative_grid_addresses[:,2] - relative_grid_addresses[:,1]
            relative_grid_addresses[:,15] = 2 * relative_grid_addresses[:,3] - relative_grid_addresses[:,2]
            relative_grid_addresses[:,16] =  relative_grid_addresses[:,3] - relative_grid_addresses[:,0] + relative_grid_addresses[:,1]
            relative_grid_addresses[:,17] =  relative_grid_addresses[:,0] - relative_grid_addresses[:,1] + relative_grid_addresses[:,2]
            relative_grid_addresses[:,18] =  relative_grid_addresses[:,1] - relative_grid_addresses[:,2] + relative_grid_addresses[:,3]
            relative_grid_addresses[:,19] =  relative_grid_addresses[:,2] - relative_grid_addresses[:,3] + relative_grid_addresses[:,0]
            for i in range(384):
                pos = np.where(np.alltrue(relative_grid_addresses[i]==0, axis=1))[0]
                if len(pos) == 0:
                    central_indices[i] = -1
                elif len(pos) == 1:
                    central_indices[i] = pos[0]
                else:
                    raise IndexError
            center_included = np.where(central_indices!=-1)
            sequence = np.argsort(central_indices[center_included])
            self._central_indices = central_indices[center_included][sequence].copy()
            self._relative_grid_addresses = relative_grid_addresses[center_included][sequence].copy()
            self._relative_tetra_indices = relative_tetra_indices[center_included][sequence].copy()

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

if __name__ == "__main__":
    thm = TetrahedronMethod(np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float), mesh=[1, 3, 8])
    print "Main diagonal index: %d" %thm.get_main_diagonal_index()
    print "Number of tetrahedra within the second neighbor: %d" % len(thm.get_tetrahedra())
    print "Number of closely-adjacent tetrahedra: %d" %np.sum(thm.get_center_indices() < 4)
    print "Indices of center vertices:"
    print thm.get_center_indices()




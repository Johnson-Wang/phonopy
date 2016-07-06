import numpy as np
from phonopy.harmonic.dynamical_matrix import get_smallest_vectors

class RealToReciprocal:
    def __init__(self,
                 fc3,
                 supercell,
                 primitive,
                 mesh,
                 symprec=1e-5,
                 atom_triplet_cut=None):
        self._fc3 = fc3
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = mesh
        self._symprec = symprec
        self._atc=atom_triplet_cut # atom_triplet cut (dtype=bool)
        num_satom = supercell.get_number_of_atoms()
        self._p2s_map = np.intc(primitive.get_primitive_to_supercell_map())
        self._s2p_map = np.intc(primitive.get_supercell_to_primitive_map())
        # Reduce supercell atom index to primitive index
        (self._smallest_vectors,
         self._multiplicity) = get_smallest_vectors(supercell,
                                                    primitive,
                                                    symprec)
        self._fc3_reciprocal = None

    def run(self, triplet, is_sym_fc3_q=False):
        num_patom = self._primitive.get_number_of_atoms()
        if is_sym_fc3_q:
            index_exchage = np.array([[0,1,2],[1,2,0],[2,0,1],[0,2,1],[1,0,2],[2,1,0]])
            fc3_reciprocal = np.zeros(
                (num_patom, num_patom, num_patom, 3, 3, 3), dtype='complex128')
            for e, index in enumerate(index_exchage):
                self._triplet = triplet[index]
                self._fc3_reciprocal = np.zeros(
                    (num_patom, num_patom, num_patom, 3, 3, 3), dtype='complex128')
                self._real_to_reciprocal()
                for patoms in np.ndindex((num_patom, num_patom, num_patom)):
                    i,j,k = np.array(patoms)
                    ii, ji, ki = np.array(patoms)[index]
                    for cart in np.ndindex((3,3,3)):
                        l, m, n = np.array(cart)
                        li, mi, ni = np.array(cart)[index]
                        fc3_reciprocal[i,j,k,l,m,n] += self._fc3_reciprocal[ii, ji, ki, li, mi, ni] / 6
            self._fc3_reciprocal[:] = fc3_reciprocal

        else:
            self._triplet = triplet
            self._fc3_reciprocal = np.zeros(
                (num_patom, num_patom, num_patom, 3, 3, 3), dtype='complex128')
            self._real_to_reciprocal()

    def get_fc3_reciprocal(self):
        return self._fc3_reciprocal

    def _real_to_reciprocal(self):
        num_patom = self._primitive.get_number_of_atoms()
        sum_triplets = np.where(
            np.all(self._triplet != 0, axis=0), self._triplet.sum(axis=0), 0)

        sum_q = sum_triplets.astype('double') / self._mesh
        for i in range(num_patom):
            for j in range(num_patom):
                for k in range(num_patom):
                    self._fc3_reciprocal[
                        i, j, k] = self._real_to_reciprocal_elements((i, j, k))

            prephase = self._get_prephase(sum_q, i)
            self._fc3_reciprocal[i] *= prephase
                
    def _real_to_reciprocal_elements(self, patom_indices):
        num_satom = self._supercell.get_number_of_atoms()
        pi = patom_indices
        i = self._p2s_map[pi[0]]
        fc3_reciprocal = np.zeros((3, 3, 3), dtype='complex128')
        for j in range(num_satom):
            if self._s2p_map[j] != self._p2s_map[pi[1]]:
                continue
            for k in range(num_satom):
                if self._s2p_map[k] != self._p2s_map[pi[2]]:
                    continue
                if self._atc[i,j,k]:
                    continue
                phase = self._get_phase((j, k), pi[0])
                fc3_reciprocal += self._fc3[i, j, k] * phase
        return fc3_reciprocal

    def _get_prephase(self, sum_q, patom_index):
        r = self._primitive.get_scaled_positions()[patom_index]
        return np.exp(2j * np.pi * np.dot(sum_q, r))

    def _get_phase(self, satom_indices, patom0_index):
        si = satom_indices
        p0 = patom0_index
        phase = 1+0j
        for i in (0, 1):
            vs = self._smallest_vectors[si[i], p0,
                                        :self._multiplicity[si[i], p0]]
            phase *= (np.exp(2j * np.pi * np.dot(
                        vs, self._triplet[i + 1].astype('double') /
                        self._mesh)).sum() / self._multiplicity[si[i], p0])
        return phase

    # def _get_phase2(self, satom_indices, patom0_index):
    #     si = satom_indices
    #     p0 = patom0_index
    #     phase = 0
    #     vs1 = self._smallest_vectors[si[0], p0,
    #                                 :self._multiplicity[si[0], p0]]
    #     vs2 = self._smallest_vectors[si[1], p0,
    #                                 :self._multiplicity[si[1], p0]]
    #     if len(vs1) > 1 and len(vs2) > 1:
    #         pass
    #     vs12 = (vs1[:, np.newaxis] - vs2[np.newaxis])
    #     cell = self._primitive.get_cell()
    #     dist = np.sqrt(np.sum(np.dot(vs12, cell) ** 2, axis=-1))
    #     where = np.where(dist < np.min(dist) + 1e-5)
    #     for (i, j) in zip(*where):
    #         phase += (np.exp(2j * np.pi * (np.dot(
    #             vs1[i], self._triplet[1].astype('double') /self._mesh) + np.dot(
    #             vs2[j], self._triplet[2].astype('double') / self._mesh))))
    #     phase /= len(zip(*where))
    #     return phase

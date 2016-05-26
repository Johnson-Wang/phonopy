import numpy as np

class ReciprocalToNormal:
    def __init__(self,
                 primitive,
                 frequencies,
                 eigenvectors,
                 cutoff_frequency=0,
                 cutoff_hfrequency=10000.0,
                 cutoff_delta = None):
        self._primitive = primitive
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._cutoff_frequency = cutoff_frequency
        self._cutoff_hfrequency = cutoff_hfrequency
        self._cutoff_delta = cutoff_delta
        self._masses = self._primitive.get_masses()

        self._fc3_normal = None
        self._fc3_reciprocal = None

    def run(self, fc3_reciprocal, grid_triplet, g_skip=None):
        num_band = self._primitive.get_number_of_atoms() * 3
        self._fc3_reciprocal = fc3_reciprocal
        self._fc3_normal = np.zeros((num_band,) * 3, dtype='double')
        self._reciprocal_to_normal(grid_triplet, g_skip=g_skip)

    def get_reciprocal_to_normal(self):
        return self._fc3_normal

    def _reciprocal_to_normal(self, grid_triplet, g_skip=None):
        e1, e2, e3 = self._eigenvectors[grid_triplet]
        f1, f2, f3 = self._frequencies[grid_triplet]
        num_band = len(f1)
        cutoff = self._cutoff_frequency
        sum_sequence = (np.ones((3,3))-2*np.eye(3))
        for (i, j, k) in list(np.ndindex((num_band,) * 3)):
            if g_skip[i,j,k]:
                continue
            if f1[i] > cutoff  and f1[i] < self._cutoff_hfrequency \
                and f2[j] > cutoff and f3[k] > cutoff:
                f=self._frequencies[grid_triplet]
                # if (np.abs(np.sum( f* sum_sequence, axis=1)) > self._cutoff_delta).all():
                #     continue
                fc3_elem = self._sum_in_atoms((i, j, k), (e1, e2, e3))
                fff = f1[i] * f2[j] * f3[k]
                self._fc3_normal[i, j, k] = np.abs(fc3_elem) ** 2 / fff

    def _sum_in_atoms(self, band_indices, eigvecs):
        num_atom = self._primitive.get_number_of_atoms()
        (e1, e2, e3) = eigvecs
        (b1, b2, b3) = band_indices

        sum_fc3 = 0j
        for (i, j, k) in list(np.ndindex((num_atom,) * 3)):
            sum_fc3_cart = 0
            for (l, m, n) in list(np.ndindex((3, 3, 3))):
                sum_fc3_cart += (e1[i * 3 + l, b1] *
                                 e2[j * 3 + m, b2] *
                                 e3[k * 3 + n, b3] *
                                 self._fc3_reciprocal[i, j, k, l, m, n])
            mass_sqrt = np.sqrt(np.prod(self._masses[[i, j, k]]))
            sum_fc3 += sum_fc3_cart / mass_sqrt

        return sum_fc3

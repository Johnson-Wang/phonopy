import numpy as np
import sys
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors

def print_error():
    print """  ___ _ __ _ __ ___  _ __
 / _ \ '__| '__/ _ \| '__|
|  __/ |  | | | (_) | |
 \___|_|  |_|  \___/|_|
"""
    sys.exit(1)

def print_error_message(message):
    print message
    print_error()

class Cutoff():
    def __init__(self, species, cut_radius, cut_pair, cut_triplets):
        self._cell = None
        self._pair_distances = None
        n = len(species)
        if cut_radius is not None:
            if len(cut_radius) == 1:
                self._cut_radius = [cut_radius[0] for i in range(n)]
            elif len(cut_radius) == n:
                self._cut_radius = cut_radius
            else:
                print_error_message("Cutoff radius number %d not equal the number of species %d!" %(len(cut_radius), n))
        else:
            self._cut_radius = None

        # cutoff pair
        cp = np.ones((n, n), dtype="float") * 10000
        if cut_pair is not None:
            if len(cut_pair) == (n +1) * n / 2:
                for i in range(n):
                    for j in range(i,n):
                        cp[i,j] = cp[j,i] = cut_pair.pop(0)
                self._cut_pair = cp
            else:
                print_error_message("Cutoff pairs %d not equal to the number needed %d!" %(len(cut_pair), (n +1) * n / 2))
        elif self._cut_radius is not None:
            for i, j in np.ndindex((n,n)):
                cp[i,j] = self._cut_radius[i]+ self._cut_radius[j]
            self._cut_pair = cp
        else:
            self._cut_pair = None

        # cutoff triplet
        ct = np.ones((n,n,n), dtype="float") * 10000
        if cut_triplets is not None:
            if len(cut_triplets) == n * (n + 1) * (n + 2) / 6:
                for i in range(n):
                    for j in range(i,n):
                        for k in range(j,n):
                            ct[i,j,k] = ct[i,k,j] = ct[j,i,k] = ct[j,k,i] =\
                                ct[k,i,j] = ct[k,j,i] = cut_triplets.pop(0)
                self._cut_triplet = ct
            else:
                print_error_message("Cutoff triplets %d not equal to the number needed %d!"\
                                    %(len(cut_triplets), n * (n + 1) * (n + 2) / 6))
        elif self._cut_pair is not None:
            for i,j,k in np.ndindex((n,n,n)):
                ct[i,j,k] = min(self._cut_pair[i,j], self._cut_pair[i,k], self._cut_pair[j,k])
            self._cut_triplet = ct
        else:
            self._cut_triplet = None

    def set_cell(self, cell, symprec = 1e-5):
        self._cell = cell
        self._symprec = symprec
        self._pair_distances = None

    def get_cutoff_pair(self):
        return self._cut_pair

    def get_cutoff_radius(self):
        return self._cut_radius

    def get_cutoff_triplet(self):
        return self._cut_triplet

    def expand_pair(self):
        unique_atoms, index_unique = np.unique(self._cell.get_atomic_numbers(), return_index=True)
        unique_atoms = unique_atoms[np.argsort(index_unique)] # in order to keep the specie sequence unchanged
        if self.get_cutoff_pair() is not None:
            cutpair_expand = np.zeros((self._cell.get_number_of_atoms(), self._cell.get_number_of_atoms()), dtype="double")
            for i in range(self._cell.get_number_of_atoms()):
                index_specie_i = np.where(unique_atoms == self._cell.get_atomic_numbers()[i])[0]
                for j in range(i, self._cell.get_number_of_atoms()):
                    index_specie_j = np.where(unique_atoms == self._cell.get_atomic_numbers()[j])[0]
                    cutpair_expand[i,j] = cutpair_expand[j,i] = self._cut_pair[index_specie_i, index_specie_j]
        else:
            cutpair_expand = None
        return cutpair_expand

    def expand_triplet(self):
        natom = self._cell.get_number_of_atoms()
        unique_atoms, index_unique = np.unique(self._cell.get_atomic_numbers(), return_index=True)
        unique_atoms = unique_atoms[np.argsort(index_unique)] # in order to keep the specie sequence unchanged
        if self.get_cutoff_triplet() is not None:
            cut_triplet_expand = np.zeros((natom, natom, natom), dtype="double")
            for i in range(natom):
                index_specie_i = np.where(unique_atoms == self._cell.get_atomic_numbers()[i])[0]
                for j in range(i, natom):
                    index_specie_j = np.where(unique_atoms == self._cell.get_atomic_numbers()[j])[0]
                    for k in range(j, natom):
                        index_specie_k = np.where(unique_atoms == self._cell.get_atomic_numbers()[k])[0]
                        cut_temp  = self._cut_triplet[index_specie_i, index_specie_j, index_specie_k]
                        cut_triplet_expand[i,j,k] = cut_temp
                        cut_triplet_expand[j,i,k] = cut_temp
                        cut_triplet_expand[i,k,j] = cut_temp
                        cut_triplet_expand[j,k,i] = cut_temp
                        cut_triplet_expand[k,i,j] = cut_temp
                        cut_triplet_expand[k,j,i] = cut_temp
                        # cut_triplet_expand[i,j, k] =  self._cut_triplet[index_specie_i, index_specie_j, index_specie_k]
        else:
            cut_triplet_expand = None
        return cut_triplet_expand

    def set_pair_distances(self):
        num_atom = self._cell.get_number_of_atoms()
        lattice = self._cell.get_cell()

        try:
            from scipy.spatial.distance import cdist
            positions = self._cell.get_scaled_positions()
            positions_cart = self._cell.get_positions()
            distances = []
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        positions_trans = positions + np.array([i,j,k])
                        positions_trans_cart = np.dot(positions_trans, lattice)
                        distances.append(cdist(positions_cart, positions_trans_cart))
            min_distances = np.min(np.array(distances), axis=0)
        except ImportError:
            min_distances = np.zeros((num_atom, num_atom), dtype='double')
            for i in range(num_atom): # run in cell
                for j in range(num_atom): # run in primitive
                    min_distances[i, j] = np.linalg.norm(np.dot(
                            get_equivalent_smallest_vectors(
                                i, j, self._cell, lattice, self._symprec)[0], lattice))
        self._pair_distances = min_distances

    def get_pair_inclusion(self):
        num_atom = self._cell.get_number_of_atoms()
        cut_pair = self.expand_pair()
        include_pair= np.ones((num_atom, num_atom), dtype=bool)
        if self._pair_distances == None:
            self.set_pair_distances()
        for i, j in np.ndindex(num_atom, num_atom):
            if cut_pair is not None:
                max_dist = max(self._pair_distances[i,j], self._pair_distances[j,i])
                if max_dist > cut_pair[i,j]:
                    include_pair[i,j] = False
        return include_pair

    def get_triplet_inclusion(self):
        num_atom = self._cell.get_number_of_atoms()
        cut_triplet = self.expand_triplet()
        include_triplet= np.ones((num_atom, num_atom, num_atom), dtype=bool)
        if self._pair_distances == None:
            self.set_pair_distances()
        if cut_triplet is not None:
            for i, j, k in np.ndindex(num_atom, num_atom, num_atom):
                max_dist = max(self._pair_distances[i,j], self._pair_distances[j,k],self._pair_distances[i,k])
                if max_dist > cut_triplet[i, j, k]:
                    include_triplet[i,j, k] = False
        return include_triplet
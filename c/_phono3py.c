#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <lapacke.h>
#include "lapack_wrapper.h"
#include "phonoc_array.h"
#include "phonoc_utils.h"
#include "phonoc_math.h"
#include "phonon3_h/fc3.h"
#include "phonon3_h/interaction.h"
#include "phonon3_h/imag_self_energy.h"
#include "phonon3_h/kappa.h"
#include "phonon4_h/real_to_reciprocal.h"
#include "phonon4_h/frequency_shift.h"
#include "other_h/isotope.h"
#include "spglib_h/tetrahedron_method.h"
static PyObject * py_get_jointDOS(PyObject *self, PyObject *args);
static PyObject * py_get_decay_channel(PyObject *self, PyObject *args);
static PyObject * py_get_interaction(PyObject *self, PyObject *args);
static PyObject * py_get_imag_self_energy(PyObject *self, PyObject *args);
static PyObject * py_get_imag_self_energy_at_bands(PyObject *self,
						   PyObject *args);
static PyObject * py_get_isotope_strength(PyObject *self, PyObject *args);
static PyObject * py_get_thm_isotope_strength(PyObject *self, PyObject *args);
static PyObject * py_set_phonon_triplets(PyObject *self, PyObject *args);
static PyObject * py_get_phonon(PyObject *self, PyObject *args);
static PyObject * py_distribute_fc3(PyObject *self, PyObject *args);
static PyObject * py_phonopy_zheev(PyObject *self, PyObject *args);
static PyObject * py_set_phonons_at_gridpoints(PyObject *self, PyObject *args);
static PyObject * py_get_thermal_conductivity_at_grid(PyObject *self, PyObject *args);
static PyObject * py_get_thermal_conductivity(PyObject *self, PyObject *args);
static PyObject * py_collision(PyObject *self, PyObject *args);
static PyObject * py_collision_all_permute(PyObject *self, PyObject *args);
static PyObject * py_collision_from_reduced(PyObject *self, PyObject *args);
static PyObject * py_collision_degeneracy(PyObject *self, PyObject *args);
static PyObject * py_collision_degeneracy_grid(PyObject *self, PyObject *args);
static PyObject * py_get_neighboring_gird_points(PyObject *self, PyObject *args);
static PyObject *py_interaction_from_reduced(PyObject *self, PyObject *args);
static PyObject * py_perturbation_next(PyObject *self, PyObject *args);
static PyObject * py_phonon_multiply_dmatrix_gbb_dvector_gb3(PyObject *self, PyObject *args);
static PyObject * py_phonon_3_multiply_dvector_gb3_dvector_gb3(PyObject *self, PyObject *args);
static PyObject * py_phonon_gb33_multiply_dvector_gb3_dvector_gb3(PyObject *self, PyObject *args);
static PyObject * py_get_thm_imag_self_energy(PyObject *self, PyObject *args);
static PyObject * py_set_integration_weights(PyObject *self, PyObject *args);
static PyObject *
py_set_triplets_integration_weights_with_sigma(PyObject *self, PyObject *args);
static PyObject *
py_set_triplets_integration_weights_with_asigma(PyObject *self, PyObject *args);
static PyObject *
py_set_triplets_integration_weights(PyObject *self, PyObject *args);
static PyObject * py_get_uniq_neighboring_tetrahedra(PyObject *self, PyObject *args);
//static PyObject *
//py_set_triplets_integration_weights_sym(PyObject *self, PyObject *args);

static void get_triplet_tetrahedra_vertices
  (int vertices[2][120][20],
   SPGCONST int relative_grid_address[2][120][20][3],
   const int mesh[3],
   const int triplet[3],
   SPGCONST int grid_address[][3],
   const int bz_map[],
   const int is_linear);
static void get_vector_modulo(int v[3], const int m[3]);
static int get_grid_point_double_mesh(const int address_double[3],
				      const int mesh[3]);
static int get_grid_point_single_mesh(const int address[3],
				      const int mesh[3]);
static void get_corrected_frequencies(double freq_tetra[120][4],
                                      const double freq_vertices[120][20],
                                      const double weight_correction[20][4],
                                      const int is_linear);
static double get_corrected_integration_weights(const double tetra_weights[120][4],
                                                const int center_indices[120],
                                                const double weight_correction[20][4],
                                                const int is_linear);
static void get_neighboring_grid_points(int neighboring_grid_points[],
				     const int grid_point,
				     SPGCONST int relative_grid_address[][3],
				     const int num_relative_grid_address,
				     const int mesh[3],
				     SPGCONST int grid_address[][3], 
				     const int bz_map[]);
static PyMethodDef functions[] = {
  {"joint_dos", py_get_jointDOS, METH_VARARGS, "Calculate joint density of states"},
  {"decay_channel", py_get_decay_channel, METH_VARARGS, "Calculate decay of phonons"},
  {"interaction", py_get_interaction, METH_VARARGS, "Interaction of triplets"},
  {"imag_self_energy", py_get_imag_self_energy, METH_VARARGS, "Imaginary part of self energy"},
  {"imag_self_energy_at_bands", py_get_imag_self_energy_at_bands, METH_VARARGS, "Imaginary part of self energy at phonon frequencies of bands"},
  {"phonon_triplets", py_set_phonon_triplets, METH_VARARGS, "Set phonon triplets"},
  {"phonon", py_get_phonon, METH_VARARGS, "Get phonon"},
  {"phonons_at_gridpoints", py_set_phonons_at_gridpoints, METH_VARARGS, "Set phonons at grid points"},
  {"distribute_fc3", py_distribute_fc3, METH_VARARGS, "Distribute least fc3 to full fc3"},
  {"zheev", py_phonopy_zheev, METH_VARARGS, "Lapack zheev wrapper"},
  {"isotope_strength", py_get_isotope_strength, METH_VARARGS, "Isotope scattering strength"},
  {"thm_isotope_strength", py_get_thm_isotope_strength, METH_VARARGS, "Isotope scattering strength for tetrahedron_method"},
  {"collision", py_collision, METH_VARARGS, "Scattering rate calculation for the iterative method"},
  {"collision_all_permute", py_collision_all_permute, METH_VARARGS, "Scattering rate calculation for all phonons in a triplet" },
  {"collision_from_reduced", py_collision_from_reduced, METH_VARARGS, "Scattering rate from reduced triplets" },
  {"collision_degeneracy", py_collision_degeneracy, METH_VARARGS, "Scattering rate symmetrization considering the degeneracy" },
  {"collision_degeneracy_grid", py_collision_degeneracy_grid, METH_VARARGS, "Scattering rate symmetrization at a grid considering the degeneracy" },
  {"interaction_from_reduced", py_interaction_from_reduced, METH_VARARGS, "interaction strength from reduced triplets" },
  {"perturbation_next", py_perturbation_next, METH_VARARGS, "Calculate the next perturbation flow"},
  {"thermal_conductivity_at_grid",py_get_thermal_conductivity_at_grid, METH_VARARGS, "thermal conductivity calculation at a grid point" },
  {"thermal_conductivity",py_get_thermal_conductivity, METH_VARARGS, "thermal conductivity calculation at all grid points" },
  {"phonon_multiply_dmatrix_gbb_dvector_gb3",py_phonon_multiply_dmatrix_gbb_dvector_gb3, METH_VARARGS,
    "phonon multiplicity between a double matrix(grid, band, band) and another vector (grid, band, 3)"},
  {"triplets_integration_weights", py_set_triplets_integration_weights, METH_VARARGS,
   "Integration weights of tetrahedron method for triplets"},
//  {"triplets_integration_weights_sym", py_set_triplets_integration_weights_sym, METH_VARARGS,
//   "Integration weights of tetrahedron method for triplets with interchange symmetry"},
  {"triplets_integration_weights_with_sigma",py_set_triplets_integration_weights_with_sigma, METH_VARARGS,
   "Integration weights of smearing method for triplets"},
   {"triplets_integration_weights_with_asigma",py_set_triplets_integration_weights_with_asigma, METH_VARARGS,
   "Integration weights of smearing method (with sigma self-adaption) for triplets"},
  {"neighboring_grid_points", py_get_neighboring_gird_points, METH_VARARGS,
   "Neighboring grid points by relative grid addresses"},
  {"uniq_neighboring_tetrahedra", py_get_uniq_neighboring_tetrahedra, METH_VARARGS,
   "Unique neighboring tetrahedra and mapping"},
  {"integration_weights", py_set_integration_weights, METH_VARARGS,
   "Integration weights of tetrahedron method"},
  {"thm_imag_self_energy", py_get_thm_imag_self_energy, METH_VARARGS,
   "Imaginary part of self energy at phonon frequencies of bands for tetrahedron method"},
  {"phonon_3_multiply_dvector_gb3_dvector_gb3",py_phonon_3_multiply_dvector_gb3_dvector_gb3, METH_VARARGS,
    "phonon multiplicity between a two vectors (grid, band, 3)"},
  {"phonon_gb33_multiply_dvector_gb3_dvector_gb3",py_phonon_gb33_multiply_dvector_gb3_dvector_gb3, METH_VARARGS,
    "phonon outer product between a two vectors (grid, band, 3)"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_phono3py(void)
{
  Py_InitModule3("_phono3py", functions, "C-extension for phono3py\n\n...\n");
  return;
}

static PyObject * py_set_phonon_triplets(PyObject *self, PyObject *args)
{
  PyArrayObject* frequencies;
  PyArrayObject* eigenvectors;
  PyArrayObject* degeneracies;
  PyArrayObject* phonon_done_py;
  PyArrayObject* grid_point_triplets;
  PyArrayObject* grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* shortest_vectors_fc2;
  PyArrayObject* multiplicity_fc2;
  PyArrayObject* fc2_py;
  PyArrayObject* atomic_masses_fc2;
  PyArrayObject* p2s_map_fc2;
  PyArrayObject* s2p_map_fc2;
  PyArrayObject* reciprocal_lattice;
  PyArrayObject* born_effective_charge;
  PyArrayObject* q_direction;
  PyArrayObject* dielectric_constant;
  double nac_factor, unit_conversion_factor;
  char uplo;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOdOOOOdc",
			&frequencies,
			&eigenvectors,
			&degeneracies,
			&phonon_done_py,
			&grid_point_triplets,
			&grid_address_py,
			&mesh_py,
			&fc2_py,
			&shortest_vectors_fc2,
			&multiplicity_fc2,
			&atomic_masses_fc2,
			&p2s_map_fc2,
			&s2p_map_fc2,
			&unit_conversion_factor,
			&born_effective_charge,
			&dielectric_constant,
			&reciprocal_lattice,
			&q_direction,
			&nac_factor,
			&uplo)) {
    return NULL;
  }

  double* born;
  double* dielectric;
  double *q_dir;
  Darray* freqs = convert_to_darray(frequencies);
  Iarray* degs = convert_to_iarray(degeneracies);
  /* npy_cdouble and lapack_complex_double may not be compatible. */
  /* So eigenvectors should not be used in Python side */
  Carray* eigvecs = convert_to_carray(eigenvectors);
  char* phonon_done = (char*)phonon_done_py->data;
  Iarray* triplets = convert_to_iarray(grid_point_triplets);
  const int* grid_address = (int*)grid_address_py->data;
  const int* mesh = (int*)mesh_py->data;
  Darray* fc2 = convert_to_darray(fc2_py);
  Darray* svecs_fc2 = convert_to_darray(shortest_vectors_fc2);
  Iarray* multi_fc2 = convert_to_iarray(multiplicity_fc2);
  const double* masses_fc2 = (double*)atomic_masses_fc2->data;
  const int* p2s_fc2 = (int*)p2s_map_fc2->data;
  const int* s2p_fc2 = (int*)s2p_map_fc2->data;
  const double* rec_lat = (double*)reciprocal_lattice->data;
  if ((PyObject*)born_effective_charge == Py_None) {
    born = NULL;
  } else {
    born = (double*)born_effective_charge->data;
  }
  if ((PyObject*)dielectric_constant == Py_None) {
    dielectric = NULL;
  } else {
    dielectric = (double*)dielectric_constant->data;
  }
  if ((PyObject*)q_direction == Py_None) {
    q_dir = NULL;
  } else {
    q_dir = (double*)q_direction->data;
  }

  set_phonon_triplets(freqs,
		      eigvecs,
		      degs,
		      phonon_done,
		      triplets,
		      grid_address,
		      mesh,
		      fc2,
		      svecs_fc2,
		      multi_fc2,
		      masses_fc2,
		      p2s_fc2,
		      s2p_fc2,
		      unit_conversion_factor,
		      born,
		      dielectric,
		      rec_lat,
		      q_dir,
		      nac_factor,
		      uplo);

  free(freqs);
  free(eigvecs);
  free(degs);
  free(triplets);
  free(fc2);
  free(svecs_fc2);
  free(multi_fc2);
  
  Py_RETURN_NONE;
}


static PyObject * py_set_phonons_at_gridpoints(PyObject *self, PyObject *args)
{
  PyArrayObject* frequencies;
  PyArrayObject* eigenvectors;
  PyArrayObject* degeneracies;
  PyArrayObject* phonon_done_py;
  PyArrayObject* grid_points_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* shortest_vectors_fc2;
  PyArrayObject* multiplicity_fc2;
  PyArrayObject* fc2_py;
  PyArrayObject* atomic_masses_fc2;
  PyArrayObject* p2s_map_fc2;
  PyArrayObject* s2p_map_fc2;
  PyArrayObject* reciprocal_lattice;
  PyArrayObject* born_effective_charge;
  PyArrayObject* q_direction;
  PyArrayObject* dielectric_constant;
  double nac_factor, unit_conversion_factor;
  char uplo;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOdOOOOdc",
			&frequencies,
			&eigenvectors,
			&degeneracies,
			&phonon_done_py,
			&grid_points_py,
			&grid_address_py,
			&mesh_py,
			&fc2_py,
			&shortest_vectors_fc2,
			&multiplicity_fc2,
			&atomic_masses_fc2,
			&p2s_map_fc2,
			&s2p_map_fc2,
			&unit_conversion_factor,
			&born_effective_charge,
			&dielectric_constant,
			&reciprocal_lattice,
			&q_direction,
			&nac_factor,
			&uplo)) {
    return NULL;
  }

  double* born;
  double* dielectric;
  double *q_dir;
  Darray* freqs = convert_to_darray(frequencies);
  Iarray* degs = convert_to_iarray(degeneracies);
  /* npy_cdouble and lapack_complex_double may not be compatible. */
  /* So eigenvectors should not be used in Python side */
  Carray* eigvecs = convert_to_carray(eigenvectors);
  char* phonon_done = (char*)phonon_done_py->data;
  Iarray* grid_points = convert_to_iarray(grid_points_py);
  const int* grid_address = (int*)grid_address_py->data;
  const int* mesh = (int*)mesh_py->data;
  Darray* fc2 = convert_to_darray(fc2_py);
  Darray* svecs_fc2 = convert_to_darray(shortest_vectors_fc2);
  Iarray* multi_fc2 = convert_to_iarray(multiplicity_fc2);
  const double* masses_fc2 = (double*)atomic_masses_fc2->data;
  const int* p2s_fc2 = (int*)p2s_map_fc2->data;
  const int* s2p_fc2 = (int*)s2p_map_fc2->data;
  const double* rec_lat = (double*)reciprocal_lattice->data;
  if ((PyObject*)born_effective_charge == Py_None) {
    born = NULL;
  } else {
    born = (double*)born_effective_charge->data;
  }
  if ((PyObject*)dielectric_constant == Py_None) {
    dielectric = NULL;
  } else {
    dielectric = (double*)dielectric_constant->data;
  }
  if ((PyObject*)q_direction == Py_None) {
    q_dir = NULL;
  } else {
    q_dir = (double*)q_direction->data;
  }

  set_phonons_at_gridpoints(freqs,
			    eigvecs,
			    degs,
			    phonon_done,
			    grid_points,
			    grid_address,
			    mesh,
			    fc2,
			    svecs_fc2,
			    multi_fc2,
			    masses_fc2,
			    p2s_fc2,
			    s2p_fc2,
			    unit_conversion_factor,
			    born,
			    dielectric,
			    rec_lat,
			    q_dir,
			    nac_factor,
			    uplo);

  free(freqs);
  free(eigvecs);
  free(degs);
  free(grid_points);
  free(fc2);
  free(svecs_fc2);
  free(multi_fc2);
  
  Py_RETURN_NONE;
}


static PyObject * py_get_phonon(PyObject *self, PyObject *args)
{
  PyArrayObject* frequencies_py;
  PyArrayObject* eigenvectors_py;
  PyArrayObject* q_py;
  PyArrayObject* shortest_vectors_py;
  PyArrayObject* multiplicity_py;
  PyArrayObject* fc2_py;
  PyArrayObject* atomic_masses_py;
  PyArrayObject* p2s_map_py;
  PyArrayObject* s2p_map_py;
  PyArrayObject* reciprocal_lattice_py;
  PyArrayObject* born_effective_charge_py;
  PyArrayObject* q_direction_py;
  PyArrayObject* dielectric_constant_py;
  double nac_factor, unit_conversion_factor;
  char uplo;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOdOOOOdc",
			&frequencies_py,
			&eigenvectors_py,
			&q_py,
			&fc2_py,
			&shortest_vectors_py,
			&multiplicity_py,
			&atomic_masses_py,
			&p2s_map_py,
			&s2p_map_py,
			&unit_conversion_factor,
			&born_effective_charge_py,
			&dielectric_constant_py,
			&reciprocal_lattice_py,
			&q_direction_py,
			&nac_factor,
			&uplo)) {
    return NULL;
  }

  double* born;
  double* dielectric;
  double *q_dir;
  double* freqs = (double*)frequencies_py->data;
  /* npy_cdouble and lapack_complex_double may not be compatible. */
  /* So eigenvectors should not be used in Python side */
  lapack_complex_double* eigvecs =
    (lapack_complex_double*)eigenvectors_py->data;
  const double* q = (double*) q_py->data;
  Darray* fc2 = convert_to_darray(fc2_py);
  Darray* svecs = convert_to_darray(shortest_vectors_py);
  Iarray* multi = convert_to_iarray(multiplicity_py);
  const double* masses = (double*)atomic_masses_py->data;
  const int* p2s = (int*)p2s_map_py->data;
  const int* s2p = (int*)s2p_map_py->data;
  const double* rec_lat = (double*)reciprocal_lattice_py->data;

  if ((PyObject*)born_effective_charge_py == Py_None) {
    born = NULL;
  } else {
    born = (double*)born_effective_charge_py->data;
  }
  if ((PyObject*)dielectric_constant_py == Py_None) {
    dielectric = NULL;
  } else {
    dielectric = (double*)dielectric_constant_py->data;
  }
  if ((PyObject*)q_direction_py == Py_None) {
    q_dir = NULL;
  } else {
    q_dir = (double*)q_direction_py->data;
  }

  get_phonons(eigvecs,
	      freqs,
	      q,
	      fc2,
	      masses,
	      p2s,
	      s2p,
	      multi,
	      svecs,
	      born,
	      dielectric,
	      rec_lat,
	      q_dir,
	      nac_factor,
	      unit_conversion_factor,
	      uplo);

  free(fc2);
  free(svecs);
  free(multi);
  
  Py_RETURN_NONE;
}


static PyObject * py_get_interaction(PyObject *self, PyObject *args)
{
  PyArrayObject* fc3_normal_squared_py;
  PyArrayObject* frequencies;
  PyArrayObject* eigenvectors;
  PyArrayObject* grid_point_triplets;
  PyArrayObject* grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* shortest_vectors;
  PyArrayObject* multiplicity;
  PyArrayObject* fc3_py;
  PyArrayObject* atc_py;
  PyArrayObject* atc_rec_py;
  PyArrayObject* g_skip_py;
  PyArrayObject* atomic_masses;
  PyArrayObject* p2s_map;
  PyArrayObject* s2p_map;
  PyArrayObject* band_indicies_py;
  double cutoff_frequency, cutoff_hfrequency;
  double cutoff_delta;
  int symmetrize_fc3_q;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOiddd",
			&fc3_normal_squared_py,
			&frequencies,
			&eigenvectors,
			&grid_point_triplets,
			&grid_address_py,
			&mesh_py,
			&fc3_py,
            &atc_py,
            &atc_rec_py,
            &g_skip_py,
			&shortest_vectors,
			&multiplicity,
			&atomic_masses,
			&p2s_map,
			&s2p_map,
			&band_indicies_py,
			&symmetrize_fc3_q,
			&cutoff_frequency,
			&cutoff_hfrequency,
                        &cutoff_delta)) {
    return NULL;
  }


  Darray* fc3_normal_squared = convert_to_darray(fc3_normal_squared_py);
  Darray* freqs = convert_to_darray(frequencies);
  /* npy_cdouble and lapack_complex_double may not be compatible. */
  /* So eigenvectors should not be used in Python side */
  Carray* eigvecs = convert_to_carray(eigenvectors);
  Iarray* triplets = convert_to_iarray(grid_point_triplets);
  const int* grid_address = (int*)grid_address_py->data;
  const int* mesh = (int*)mesh_py->data;
  Darray* fc3 = convert_to_darray(fc3_py);
  const int* atc = (int*)atc_py->data;
  const int* atc_rec = (int*)atc_rec_py->data;
  const char* g_skip = (char*)g_skip_py->data;
  Darray* svecs = convert_to_darray(shortest_vectors);
  Iarray* multi = convert_to_iarray(multiplicity);
  const double* masses = (double*)atomic_masses->data;
  const int* p2s = (int*)p2s_map->data;
  const int* s2p = (int*)s2p_map->data;
  const int* band_indicies = (int*)band_indicies_py->data;

  get_interaction(fc3_normal_squared,
		  freqs,
		  eigvecs,
		  triplets,
		  grid_address,
		  mesh,
		  fc3,
          atc,
          atc_rec,
          g_skip,
		  svecs,
		  multi,
		  masses,
		  p2s,
		  s2p,
		  band_indicies,
		  symmetrize_fc3_q,
		  cutoff_frequency,
		  cutoff_hfrequency,
          cutoff_delta);

  free(fc3_normal_squared);
  free(freqs);
  free(eigvecs);
  free(triplets);
  free(fc3);
  free(svecs);
  free(multi);
  
  Py_RETURN_NONE;
}


static PyObject * py_get_imag_self_energy(PyObject *self, PyObject *args)
{
  PyArrayObject* gamma_py;
  PyArrayObject* fc3_normal_squared_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* grid_point_triplets_py;
  PyArrayObject* triplet_weights_py;
  PyArrayObject* asigma_py;
  double cutoff_gamma, unit_conversion_factor,cutoff_delta, cutoff_frequency, temperature, fpoint;

  if (!PyArg_ParseTuple(args, "OOOOOddOdddd",
			&gamma_py,
			&fc3_normal_squared_py,
			&grid_point_triplets_py,
			&triplet_weights_py,
			&frequencies_py,
			&fpoint,
			&temperature,
			&asigma_py,
			&unit_conversion_factor,
            &cutoff_delta,
			&cutoff_frequency,
			&cutoff_gamma)) {
    return NULL;
  }


  Darray* fc3_normal_squared = convert_to_darray(fc3_normal_squared_py);
  double* gamma = (double*)gamma_py->data;
  const double* frequencies = (double*)frequencies_py->data;
  const int* grid_point_triplets = (int*)grid_point_triplets_py->data;
  const int* triplet_weights = (int*)triplet_weights_py->data;
  const double* asigma = (double*) asigma_py->data;
  get_imag_self_energy(gamma,
		       fc3_normal_squared,
		       fpoint,
		       frequencies,
		       grid_point_triplets,
		       triplet_weights,
		       asigma,
		       temperature,
		       unit_conversion_factor,
               cutoff_delta,
		       cutoff_frequency,
		       cutoff_gamma);

  free(fc3_normal_squared);
  
  Py_RETURN_NONE;
}


static PyObject * py_get_imag_self_energy_at_bands(PyObject *self,
						   PyObject *args)
{
  PyArrayObject* gamma_py;
  PyArrayObject* fc3_normal_squared_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* grid_point_triplets_py;
  PyArrayObject* triplet_weights_py;
  PyArrayObject* band_indices_py;
  PyArrayObject *asigmas_py;
  double cutoff_sigma, unit_conversion_factor,cutoff_delta, cutoff_frequency, temperature, cutoff_hfrequency;

  if (!PyArg_ParseTuple(args, "OOOOOOdOddddd",
			&gamma_py,
			&fc3_normal_squared_py,
			&grid_point_triplets_py,
			&triplet_weights_py,
			&frequencies_py,
			&band_indices_py,
			&temperature,
			&asigmas_py,
			&unit_conversion_factor,
                        &cutoff_delta,
			&cutoff_frequency,
			&cutoff_hfrequency,
			&cutoff_sigma)) {
    return NULL;
  }


  Darray* fc3_normal_squared = convert_to_darray(fc3_normal_squared_py);
  double* gamma = (double*)gamma_py->data;
  const double* asigmas = (double*) asigmas_py->data;
  const double* frequencies = (double*)frequencies_py->data;
  const int* band_indices = (int*)band_indices_py->data;
  const int* grid_point_triplets = (int*)grid_point_triplets_py->data;
  const int* triplet_weights = (int*)triplet_weights_py->data;

  get_imag_self_energy_at_bands(gamma,
				fc3_normal_squared,
				band_indices,
				frequencies,
				grid_point_triplets,
				triplet_weights,
				asigmas,
				temperature,
				unit_conversion_factor,
                                cutoff_delta,
				cutoff_frequency,
				cutoff_hfrequency,
				cutoff_sigma);

  free(fc3_normal_squared);
  
  Py_RETURN_NONE;
}

static PyObject * py_get_jointDOS(PyObject *self, PyObject *args)
{
  PyArrayObject* jointdos;
  PyArrayObject* omegas;
  PyArrayObject* weights;
  PyArrayObject* frequencies;
  double sigma;

  if (!PyArg_ParseTuple(args, "OOOOd",
			&jointdos,
			&omegas,
			&weights,
			&frequencies,
			&sigma)) {
    return NULL;
  }
  
  double* jdos = (double*)jointdos->data;
  const double* o = (double*)omegas->data;
  const int* w = (int*)weights->data;
  const double* f = (double*)frequencies->data;
  const int num_band = (int)frequencies->dimensions[2];
  const int num_omega = (int)omegas->dimensions[0];
  const int num_triplet = (int)weights->dimensions[0];

  get_jointDOS(jdos,
	       num_omega,
	       num_triplet,
	       num_band,
	       o,
	       f,
	       w,
	       sigma);
  
  Py_RETURN_NONE;
}

static PyObject * py_get_isotope_strength(PyObject *self, PyObject *args)
{
  PyArrayObject* collision_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* eigenvectors_py;
  PyArrayObject* band_indices_py;
  PyArrayObject* mass_variances_py;
  PyArrayObject* occupations_py;
  PyArrayObject* ir_grid_points_py;
  int grid_point;
  double cutoff_frequency;
  double sigma;

  if (!PyArg_ParseTuple(args, "OiOOOOOOdd",
			&collision_py,
			&grid_point,
			&ir_grid_points_py,
			&mass_variances_py,
			&frequencies_py,
			&eigenvectors_py,
			&band_indices_py,
			&occupations_py,
			&sigma,
			&cutoff_frequency)) {
    return NULL;
  }


  double* collision = (double*)collision_py->data;
  const double* frequencies = (double*)frequencies_py->data;
  const lapack_complex_double* eigenvectors =
    (lapack_complex_double*)eigenvectors_py->data;
  const int* ir_grid_points = (int*)ir_grid_points_py->data;
  const int* band_indices = (int*)band_indices_py->data;
  const double* mass_variances = (double*)mass_variances_py->data;
  const double* occupations = (double*) occupations_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  const int num_band0 = (int)band_indices_py->dimensions[0];
  const int num_grid_points = (int) collision_py->dimensions[0];
  
   get_isotope_scattering_strength(collision, 
  				  grid_point,
  				  ir_grid_points,
  				  mass_variances,
  				  frequencies,
  				  eigenvectors,
  				  band_indices,
				  occupations,
  				  num_grid_points,
  				  num_band,
  				  num_band0,
  				  sigma,
  				  cutoff_frequency);
  
  Py_RETURN_NONE;
}

static PyObject * py_get_thm_isotope_strength(PyObject *self, PyObject *args)
{
  PyArrayObject* collision_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* eigenvectors_py;
  PyArrayObject* band_indices_py;
  PyArrayObject* mass_variances_py;
  PyArrayObject* ir_grid_points_py;
  PyArrayObject* occupations_py;

  int grid_point;
  double cutoff_frequency;
  PyArrayObject* integration_weights_py;


  if (!PyArg_ParseTuple(args, "OiOOOOOOOd",
			&collision_py,
			&grid_point,
			&ir_grid_points_py,
			&mass_variances_py,
			&frequencies_py,
			&eigenvectors_py,
			&band_indices_py,
			&occupations_py,
			&integration_weights_py,
			&cutoff_frequency)) {
    return NULL;
  }


  double* collision = (double*)collision_py->data;
  const double* frequencies = (double*)frequencies_py->data;
  const int* ir_grid_points = (int*)ir_grid_points_py->data;
  const lapack_complex_double* eigenvectors =
    (lapack_complex_double*)eigenvectors_py->data;
  const int* band_indices = (int*)band_indices_py->data;
  const double* mass_variances = (double*)mass_variances_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  const int num_band0 = (int)band_indices_py->dimensions[0];
  const double *occupations = (double *)occupations_py->data;
  const double* integration_weights = (double*)integration_weights_py->data;
  const int num_ir_grid_points = (int)ir_grid_points_py->dimensions[0];
    
  get_thm_isotope_scattering_strength(collision,
				      grid_point,
				      ir_grid_points,
				      mass_variances,
				      frequencies,
				      eigenvectors,
				      num_ir_grid_points,
				      band_indices,
				      occupations, //occupation
				      num_band,
				      num_band0,
				      integration_weights,
				      cutoff_frequency);
  
  Py_RETURN_NONE;
}

static PyObject * py_get_decay_channel(PyObject *self, PyObject *args)
{
  PyArrayObject* decay_values;
  PyArrayObject* amplitudes;
  PyArrayObject* omegas;
  PyArrayObject* frequencies;
  double sigma, t, cutoff_frequency;

  if (!PyArg_ParseTuple(args, "OOOOddd",
			&decay_values,
			&amplitudes,
			&frequencies,
			&omegas,
			&cutoff_frequency,
			&t,
			&sigma)) {
    return NULL;
  }
  

  double* decay = (double*)decay_values->data;
  const double* amp = (double*)amplitudes->data;
  const double* f = (double*)frequencies->data;
  const double* o = (double*)omegas->data;
  
  const int num_band = (int)amplitudes->dimensions[2];
  const int num_triplet = (int)amplitudes->dimensions[0];
  const int num_omega = (int)omegas->dimensions[0];

  get_decay_channels(decay,
		     num_omega,
		     num_triplet,
		     num_band,
		     o,
		     f,
		     amp,
		     sigma,
		     t,
		     cutoff_frequency);
  
  Py_RETURN_NONE;
}

static PyObject * py_distribute_fc3(PyObject *self, PyObject *args)
{
  PyArrayObject* force_constants_third;
  int third_atom;
  PyArrayObject* rotation_cart_inv;
  PyArrayObject* atom_mapping_py;

  if (!PyArg_ParseTuple(args, "OiOO",
			&force_constants_third,
			&third_atom,
			&atom_mapping_py,
			&rotation_cart_inv)) {
    return NULL;
  }

  double* fc3 = (double*)force_constants_third->data;
  const double* rot_cart_inv = (double*)rotation_cart_inv->data;
  const int* atom_mapping = (int*)atom_mapping_py->data;
  const int num_atom = (int)atom_mapping_py->dimensions[0];

  return PyInt_FromLong((long) distribute_fc3(fc3,
					      third_atom,
					      atom_mapping,
					      num_atom,
					      rot_cart_inv));
}


static PyObject * py_phonopy_zheev(PyObject *self, PyObject *args)
{
  PyArrayObject* dynamical_matrix;
  PyArrayObject* eigenvalues;

  if (!PyArg_ParseTuple(args, "OO",
			&dynamical_matrix,
			&eigenvalues)) {
    return NULL;
  }

  const int dimension = (int)dynamical_matrix->dimensions[0];
  npy_cdouble *dynmat = (npy_cdouble*)dynamical_matrix->data;
  double *eigvals = (double*)eigenvalues->data;

  lapack_complex_double *a;
  int i, info;

  a = (lapack_complex_double*) malloc(sizeof(lapack_complex_double) *
				      dimension * dimension);
  for (i = 0; i < dimension * dimension; i++) {
    a[i] = lapack_make_complex_double(dynmat[i].real, dynmat[i].imag);
  }

  info = phonopy_zheev(eigvals, a, dimension, 'L');

  for (i = 0; i < dimension * dimension; i++) {
    dynmat[i].real = lapack_complex_double_real(a[i]);
    dynmat[i].imag = lapack_complex_double_imag(a[i]);
  }

  free(a);
  
  return PyInt_FromLong((long) info);
}

static PyObject *py_collision(PyObject *self, PyObject *args)
{
  PyArrayObject *py_scatt;
  PyArrayObject *py_interaction;
  PyArrayObject *py_frequency;
  PyArrayObject *py_g;
  double cutoff_frequency;
  double temperature;
  if (!PyArg_ParseTuple(args, "OOOOdd",
			&py_scatt,
			&py_interaction,
			&py_frequency,
			&py_g,
			&temperature,
			&cutoff_frequency))
    return NULL;
  const int num_triplet = (int)py_interaction->dimensions[0];
  const int num_band = (int) py_interaction->dimensions[2];
  const double *interaction = (double*)py_interaction->data;
  const double *frequency = (double*)py_frequency->data;
  const double *g=(double*)py_g->data;
  double *scatt=(double*)py_scatt->data;
  get_collision_at_all_band(scatt,
			    interaction,
			    frequency,
			    g,
			    num_triplet,
			    temperature,
			    num_band,
			    cutoff_frequency);
  Py_RETURN_NONE;
}

static PyObject *py_collision_all_permute(PyObject *self, PyObject *args)
{
  PyArrayObject *py_scatt;
  PyArrayObject *py_occupation;
  PyArrayObject *py_interaction;
  PyArrayObject *py_frequency;
  PyArrayObject *py_integration_weight;
  double cutoff_frequency;
  if (!PyArg_ParseTuple(args, "OOOOOd",
			&py_scatt,
			&py_interaction,
			&py_occupation,
			&py_frequency,
			&py_integration_weight,
			&cutoff_frequency))
    return NULL;
  const int num_triplet = (int)py_interaction->dimensions[0];
  const int num_band = (int) py_interaction->dimensions[2];
  const double *occupation = (double*)py_occupation->data;
  const double *interaction = (double*)py_interaction->data;
  const double *frequency = (double*)py_frequency->data;
  const double *g=(double*)py_integration_weight->data;
  double *scatt=(double*)py_scatt->data;
  get_collision_at_all_band_permute(scatt,
				  interaction,
				  occupation,
				  frequency,
				  num_triplet,
				  num_band,
				  g,
				  cutoff_frequency);
  Py_RETURN_NONE;
}

static PyObject *py_collision_from_reduced(PyObject *self, PyObject *args)
{
  PyArrayObject *py_scatt_at_grid;
  PyArrayObject *py_scatt_all;
  PyArrayObject *py_triplet_mapping;
  PyArrayObject *py_triplet_sequence;
  

  if (!PyArg_ParseTuple(args, "OOOO",
			&py_scatt_at_grid,
			&py_scatt_all,
			&py_triplet_mapping,
			&py_triplet_sequence))
    return NULL;
  const int num_triplet = (int)py_scatt_at_grid->dimensions[0];
  const int num_band = (int) py_scatt_at_grid->dimensions[1];
  
  const double *scattall = (double*)py_scatt_all->data;
  const int *triplet_mapping = (int*)py_triplet_mapping->data;
  const char *triplet_sequence = (char*)py_triplet_sequence->data;
  double *scatt=(double*)py_scatt_at_grid->data;
  
  get_collision_from_reduced(scatt, scattall, triplet_mapping,triplet_sequence,num_triplet, num_band);
  Py_RETURN_NONE;
}

static PyObject *py_collision_degeneracy(PyObject *self, PyObject *args)
{
  PyArrayObject *py_scatt;
  PyArrayObject *py_triplet_degeneracy;
  

  if (!PyArg_ParseTuple(args, "OO",
			&py_scatt,
			&py_triplet_degeneracy))
    return NULL;
  const int num_triplet = (int)py_triplet_degeneracy->dimensions[0];
  const int num_band = (int) py_triplet_degeneracy->dimensions[2];
  
  const int *triplet_degeneracy = (int*)py_triplet_degeneracy->data;
  double *scatt=(double*)py_scatt->data;
  
  collision_degeneracy(scatt,triplet_degeneracy, num_triplet, num_band);
  Py_RETURN_NONE;
}

static PyObject *py_collision_degeneracy_grid(PyObject *self, PyObject *args)
{
  PyArrayObject *py_scatt;
  PyArrayObject *py_degeneracies_all;
  PyArrayObject *py_grid_points2;
  int grid_point;


  if (!PyArg_ParseTuple(args, "OOOi",
			&py_scatt,
			&py_degeneracies_all,
			&py_grid_points2,
			&grid_point))
    return NULL;
  const int num_grid_points2 = (int)py_grid_points2->dimensions[0];
  const int num_band = (int) py_scatt->dimensions[2];
  const int *degeneracies_all = (int*)py_degeneracies_all->data;
  const int *grid_points2 = (int *)py_grid_points2->data;
  double *scatt=(double*)py_scatt->data;
  collision_degeneracy_grid(scatt,degeneracies_all, grid_point, grid_points2, num_grid_points2, num_band);
  Py_RETURN_NONE;
}

static PyObject *py_interaction_from_reduced(PyObject *self, PyObject *args)
{
  PyArrayObject *py_amplitude_at_grid;
  PyArrayObject *py_amplitude_all;
  PyArrayObject *py_triplet_mapping;
  PyArrayObject *py_triplet_sequence;
  

  if (!PyArg_ParseTuple(args, "OOOO",
			&py_amplitude_at_grid,
			&py_amplitude_all,
			&py_triplet_mapping,
			&py_triplet_sequence))
    return NULL;
  const int num_triplet = (int)py_amplitude_at_grid->dimensions[0];
  const int num_band = (int) py_amplitude_at_grid->dimensions[1];
  
  const double *amplitude_all = (double*)py_amplitude_all->data;
  const int *triplet_mapping = (int*)py_triplet_mapping->data;
  const char *triplet_sequence = (char*)py_triplet_sequence->data;
  double *amplitude=(double*)py_amplitude_at_grid->data;
  
  get_interaction_from_reduced(amplitude, amplitude_all, triplet_mapping,triplet_sequence,num_triplet, num_band);
  Py_RETURN_NONE;
}

static PyObject *py_perturbation_next(PyObject *self, PyObject *args)
{
  PyArrayObject *py_scattering;
  PyArrayObject *py_convergence;
  PyArrayObject *py_q1s;
  PyArrayObject *py_cart_inv_rot_sum;
  PyArrayObject *py_spg_mapping_index;
  PyArrayObject *py_F_prev;
  PyArrayObject *py_pf_sum;
  if (!PyArg_ParseTuple(args, "OOOOOOO",
                        &py_pf_sum,
			&py_F_prev,
			&py_spg_mapping_index,
			&py_scattering,
			&py_convergence,
			&py_q1s,
			&py_cart_inv_rot_sum))
    return NULL;
  const int num_triplet = (int)py_scattering->dimensions[0];
  const int num_band = (int) py_scattering->dimensions[1];
  const int num_grid_points = (int)py_spg_mapping_index->dimensions[0];
  
  const int *q1s = (int*) py_q1s->data;
  const double *scatt=(double*)py_scattering->data;
  const int *convergence=(int*)py_convergence->data;
  const int *spg_mapping_index=(int*)py_spg_mapping_index->data;
  const double *F_prev = (double*) py_F_prev->data;
  const double *cart_inv_rot_sum = (double*) py_cart_inv_rot_sum->data;
  double *pf_sum=(double*)py_pf_sum->data;
  get_next_perturbation_at_all_bands(pf_sum, //perturbation flow summation at all triplets
				     F_prev,
				     scatt,
				     convergence,
				     q1s,
				     spg_mapping_index, 
				     cart_inv_rot_sum, 
				     num_grid_points,
				     num_triplet,
				     num_band);
  Py_RETURN_NONE;
}

static PyObject * py_phonon_multiply_dmatrix_gbb_dvector_gb3(PyObject *self, PyObject *args)
{
  PyArrayObject *py_vector0;
  PyArrayObject *py_matrix;
  PyArrayObject *py_vector1;
  PyArrayObject *py_rots;
  PyArrayObject *py_weights;
  PyArrayObject *py_rec_lat;
  if (!PyArg_ParseTuple(args, "OOOOOO",
                        &py_vector0,
			&py_matrix,
			&py_vector1,
			&py_weights,
			&py_rots,
			&py_rec_lat))
    return NULL;
  double * vector0 = (double*) py_vector0->data;
  const int num_grids = (int)py_vector1->dimensions[0];
  const int num_bands = (int)py_vector1->dimensions[1];
  const double *matrix = (double*) py_matrix->data;
  const double *vector1 = (double*) py_vector1->data;
  const double *rots = (double*) py_rots->data;
  const int *weights = (int*) py_weights->data;
  const double *rec_lat = (double*) py_rec_lat->data;
  phonon_multiply_dmatrix_gbb_dvector_gb3(vector0, 
				     matrix,
				     vector1,
				     weights,
				     rots, //the rots should be a matrix to be multiplied with vector1 first
				     rec_lat,
				     num_grids,
				     num_bands);
  Py_RETURN_NONE;
}

static PyObject * py_phonon_3_multiply_dvector_gb3_dvector_gb3(PyObject *self, PyObject *args)
{
  PyArrayObject *py_vector0;
  PyArrayObject *py_vector1;
  PyArrayObject *py_vector2;
  PyArrayObject *py_mapping;
  PyArrayObject *py_rots;
  PyArrayObject *py_rec_lat;
  
  if (!PyArg_ParseTuple(args, "OOOOOO",
                        &py_vector0,
			&py_vector1,
			&py_vector2,
			&py_mapping,
			&py_rots, 
			&py_rec_lat))
    return NULL;
  
  const int num_grids = (int)py_mapping->dimensions[0]; // the number of all grids
  const int num_bands = (int)py_vector1->dimensions[1];
  double *vector0 = (double*) py_vector0->data;
  const double *vector1 = (double*) py_vector1->data;
  const double * vector2 = (double*) py_vector2->data;
  const int * rots = (int*) py_rots->data;
  const int *mapping = (int*)py_mapping->data;
  const double *rec_lat = (double*) py_rec_lat->data;
  phonon_3_multiply_dvector_gb3_dvector_gb3(vector0, vector1, vector2, rots, rec_lat,  mapping,num_grids,  num_bands);
  Py_RETURN_NONE;
}

static PyObject * py_phonon_gb33_multiply_dvector_gb3_dvector_gb3(PyObject *self, PyObject *args)
{
  PyArrayObject *py_vector0;
  PyArrayObject *py_vector1;
  PyArrayObject *py_vector2;
  PyArrayObject *py_mapping;
  PyArrayObject *py_rots;
  PyArrayObject *py_rec_lat;
  
  if (!PyArg_ParseTuple(args, "OOOOOO",
                        &py_vector0,
			&py_vector1,
			&py_vector2,
			&py_mapping,
			&py_rots, 
			&py_rec_lat))
    return NULL;
  
  const int num_grids = (int)py_mapping->dimensions[0]; // the number of all grids
  const int num_ir_grids = (int)py_vector1->dimensions[0]; // number of irreducible grids
  const int num_bands = (int)py_vector1->dimensions[1];
  double *vector0 = (double*) py_vector0->data;
  const double *vector1 = (double*) py_vector1->data;
  const double * vector2 = (double*) py_vector2->data;
  const int * rots = (int*) py_rots->data;
  const int *mapping = (int*)py_mapping->data;
  const double *rec_lat = (double*) py_rec_lat->data;
  phonon_gb33_multiply_dvector_gb3_dvector_gb3(vector0, vector1, vector2, rots, rec_lat,  mapping,num_grids, num_ir_grids, num_bands);
  Py_RETURN_NONE;
}




static PyObject *py_get_thermal_conductivity_at_grid(PyObject *self, PyObject *args)
{
  PyArrayObject* py_kappa;
  PyArrayObject* py_mfp;
  PyArrayObject* py_heat_capacity;
  PyArrayObject* py_group_velocity;
  PyArrayObject* py_kpt_rotations_at_q;
  PyArrayObject* py_rec_lat;
  PyArrayObject* py_degeneracy;
  
  if (!PyArg_ParseTuple(args, "OOOOOOO",
			&py_kappa,
			&py_kpt_rotations_at_q,
			&py_rec_lat,
			&py_heat_capacity,
			&py_mfp,
			&py_group_velocity,
			&py_degeneracy))
    return NULL;
  Iarray *kpt_rotations_at_q = convert_to_iarray(py_kpt_rotations_at_q);
  const int num_temp = (int)py_kappa->dimensions[0];
  const int num_band = (int)py_kappa->dimensions[1];
  const double *rec_lat = (double*)py_rec_lat->data;
  const double *mfp=(double*)py_mfp->data;
  const double *heat_capacity=(double*)py_heat_capacity->data;
  const double *gv=(double*)py_group_velocity->data;
  const int *degeneracy = (int*) py_degeneracy->data;
  double *kappa= (double*)py_kappa->data;
  get_kappa_at_grid_point(kappa,
			  kpt_rotations_at_q,
			  rec_lat, 
			  gv,
			  heat_capacity, 
			  mfp, 
			  degeneracy,
			  num_band,
			  num_temp);
  Py_RETURN_NONE;
}


static PyObject *py_get_thermal_conductivity(PyObject *self, PyObject *args)
{
  PyArrayObject* py_kappa;
  PyArrayObject* py_F;
  PyArrayObject* py_heat_capacity;
  PyArrayObject* py_group_velocity;
  PyArrayObject* py_index_mappings;
  PyArrayObject* py_kpt_rotations;
  PyArrayObject* py_rec_lat;
  PyArrayObject* py_degeneracies;

  if (!PyArg_ParseTuple(args, "OOOOOOOO",
			&py_kappa,
			&py_F,
			&py_heat_capacity,
			&py_group_velocity,
			&py_rec_lat,
			&py_index_mappings,
			&py_kpt_rotations,
			&py_degeneracies))
    return NULL;
  const int num_grid = (int)py_kappa->dimensions[0];
  const int num_all_grids = (int)py_index_mappings->dimensions[0];
  const int num_temp = (int)py_kappa->dimensions[1];
  const int num_band = (int)py_kappa->dimensions[2];
  const double *rec_lat = (double*)py_rec_lat->data;
  const int *index_mappings = (int*)py_index_mappings->data;
  const int *kpt_rotations= (int*)py_kpt_rotations->data;
  const double *F=(double*)py_F->data;
  const double *heat_capacity=(double*)py_heat_capacity->data;
  const double *gv=(double*)py_group_velocity->data;
  const int *degeneracies = (int*) py_degeneracies->data;
  double *kappa= (double*)py_kappa->data;
  get_kappa(kappa,
	    F,
	    heat_capacity,
	    gv,
	    rec_lat, 
	    index_mappings,
	    kpt_rotations, 
	    degeneracies,
	    num_grid,
	    num_all_grids,
	    num_band,
	    num_temp);
  Py_RETURN_NONE;
}

static PyObject *
py_set_triplets_integration_weights_with_asigma(PyObject *self, PyObject *args)
{
  PyArrayObject* iw_py;
  PyArrayObject* frequency_points_py;
  PyArrayObject* triplets_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* asigmas_py;
  if (!PyArg_ParseTuple(args, "OOOOO",
			&iw_py,
			&frequency_points_py,
			&triplets_py,
			&frequencies_py,
			&asigmas_py)) {
    return NULL;
  }

  double *iw = (double*)iw_py->data;
  const double *frequency_points = (double*)frequency_points_py->data;
  const double *asigmas = (double*)asigmas_py->data;
  const int num_band0 = frequency_points_py->dimensions[0];
  SPGCONST int (*triplets)[3] = (int(*)[3])triplets_py->data;
  const int num_triplets = (int)triplets_py->dimensions[0];
  const double *frequencies = (double*)frequencies_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  
  int i, j, k, l, adrs_shift;
  double f0, f1, f2, g0, g1, g2, sigma;

#pragma omp parallel for private(j, k, l, adrs_shift, f0, f1, f2, g0, g1, g2, sigma)
  for (i = 0; i < num_triplets; i++) {
    for (j = 0; j < num_band0; j++) {
      f0 = frequency_points[j];
      for (k = 0; k < num_band; k++) {
	f1 = frequencies[triplets[i][1] * num_band + k];
	for (l = 0; l < num_band; l++) {
	  f2 = frequencies[triplets[i][2] * num_band + l];
	  adrs_shift = i * num_band0 * num_band * num_band +
	    j * num_band * num_band + k * num_band + l;
	  sigma = asigmas[adrs_shift];
	  g0 = gaussian(f0 - f1 - f2, sigma);
	  g1 = gaussian(f0 - f1 + f2, sigma);
	  g2 = gaussian(f0 + f1 - f2, sigma);
	 
	  iw[adrs_shift] = g0;
	  adrs_shift += num_triplets * num_band0 * num_band * num_band;
	  iw[adrs_shift] = g1;
	  adrs_shift += num_triplets * num_band0 * num_band * num_band;
	  iw[adrs_shift] = g2;
	}
      }
    }
  }

  Py_RETURN_NONE;
}

static PyObject *
py_set_triplets_integration_weights_with_sigma(PyObject *self, PyObject *args)
{
  PyArrayObject* iw_py;
  PyArrayObject* frequency_points_py;
  PyArrayObject* triplets_py;
  PyArrayObject* frequencies_py;
  double sigma;
  if (!PyArg_ParseTuple(args, "OOOOd",
			&iw_py,
			&frequency_points_py,
			&triplets_py,
			&frequencies_py,
			&sigma)) {
    return NULL;
  }

  double *iw = (double*)iw_py->data;
  const double *frequency_points = (double*)frequency_points_py->data;
  const int num_band0 = frequency_points_py->dimensions[0];
  SPGCONST int (*triplets)[3] = (int(*)[3])triplets_py->data;
  const int num_triplets = (int)triplets_py->dimensions[0];
  const double *frequencies = (double*)frequencies_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  
  int i, j, k, l, adrs_shift;
  double f0, f1, f2, g0, g1, g2;

#pragma omp parallel for private(j, k, l, adrs_shift, f0, f1, f2, g0, g1, g2)
  for (i = 0; i < num_triplets; i++) {
    for (j = 0; j < num_band0; j++) {
      f0 = frequency_points[j];
      for (k = 0; k < num_band; k++) {
	f1 = frequencies[triplets[i][1] * num_band + k];
	for (l = 0; l < num_band; l++) {
	  f2 = frequencies[triplets[i][2] * num_band + l];

	  g0 = gaussian(f0 - f1 - f2, sigma);
	  g1 = gaussian(f0 - f1 + f2, sigma);
	  g2 = gaussian(f0 + f1 - f2, sigma);
	  adrs_shift = i * num_band0 * num_band * num_band +
	    j * num_band * num_band + k * num_band + l;
	  iw[adrs_shift] = g0;
	  adrs_shift += num_triplets * num_band0 * num_band * num_band;
	  iw[adrs_shift] = g1;
	  adrs_shift += num_triplets * num_band0 * num_band * num_band;
	  iw[adrs_shift] = g2;
	}
      }
    }
  }

  Py_RETURN_NONE;
}

static PyObject * py_set_integration_weights(PyObject *self, PyObject *args)
{
  PyArrayObject* iw_py;
  PyArrayObject* frequency_points_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* grid_points_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* bz_map_py;
  PyArrayObject* weight_correction_py;
  PyArrayObject* tratra_center_indices_py;
  int is_linear;
  if (!PyArg_ParseTuple(args, "OOOOOOOOOOi",
			&iw_py,
			&frequency_points_py,
			&relative_grid_address_py,
			&weight_correction_py,
			&tratra_center_indices_py,
			&mesh_py,
			&grid_points_py,
			&frequencies_py,
			&grid_address_py, 
			&bz_map_py,
			&is_linear)) {
    return NULL;
  }

  double *iw = (double *)iw_py->data;
  double iw_tmp[120][4];
  const double *frequency_points = (double*)frequency_points_py->data;
  const int num_band0 = frequency_points_py->dimensions[0];
  SPGCONST int (*relative_grid_address)[20][3] =
    (int(*)[20][3])relative_grid_address_py->data;
  const int *mesh = (int*)mesh_py->data;
  const double (*weight_correction)[4] = (double (*)[4]) weight_correction_py->data;
  const int *center_indices = (int *)tratra_center_indices_py->data;
  SPGCONST int *grid_points = (int*)grid_points_py->data;
  const int num_gp = (int)grid_points_py->dimensions[0];
  SPGCONST int (*grid_address)[3] = (int(*)[3])grid_address_py->data;
  const double *frequencies = (double*)frequencies_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  const int *bz_map = (int*)bz_map_py->data;
  int i, j, k, bi;
  int vertices[120][20];
  double freq_vertices[120][20], freq_tetra[120][4];
  int na, nt;
  na = (is_linear)? 24: 120;
  nt = (is_linear)? 4: 20;

#pragma omp parallel for private(j, k, bi, vertices, freq_vertices)
  for (i = 0; i < num_gp; i++) {
    for (j = 0; j < na; j++) {
      get_neighboring_grid_points(vertices[j],
				      grid_points[i],
				      relative_grid_address[j],
				      nt,
				      mesh,
				      grid_address,
				      bz_map);
    }
    for (bi = 0; bi < num_band; bi++) {
      for (j = 0; j < na; j++) {
        for (k = 0; k < nt; k++) {
	      freq_vertices[j][k] = frequencies[vertices[j][k] * num_band + bi];
	    }
      }
      get_corrected_frequencies(freq_tetra, freq_vertices, weight_correction, is_linear);

      for (j = 0; j < num_band0; j++) {
	    thm_get_integration_weight(iw_tmp,
	                             frequency_points[j],
	                             freq_tetra,
	                             is_linear,
	                             'I');
        iw[i * num_band0 * num_band + j * num_band + bi] +=
         get_corrected_integration_weights(iw_tmp, center_indices, weight_correction, is_linear);
      }
    }
  }

  Py_RETURN_NONE;
}

static PyObject * py_get_thm_imag_self_energy(PyObject *self, PyObject *args)
{
  PyArrayObject* gamma_py;
  PyArrayObject* fc3_normal_squared_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* grid_point_triplets_py;
  PyArrayObject* triplet_weights_py;
  PyArrayObject* g_py;
  double unit_conversion_factor, cutoff_frequency, temperature;

  if (!PyArg_ParseTuple(args, "OOOOOdOdd",
			&gamma_py,
			&fc3_normal_squared_py,
			&grid_point_triplets_py,
			&triplet_weights_py,
			&frequencies_py,
			&temperature,
			&g_py,
			&unit_conversion_factor,
			&cutoff_frequency)) {
    return NULL;
  }

  Darray* fc3_normal_squared = convert_to_darray(fc3_normal_squared_py);
  double* gamma = (double*)gamma_py->data;
  const double* g = (double*)g_py->data;
  const double* frequencies = (double*)frequencies_py->data;
  const int* grid_point_triplets = (int*)grid_point_triplets_py->data;
  const int* triplet_weights = (int*)triplet_weights_py->data;

  get_thm_imag_self_energy_at_bands(gamma,
				    fc3_normal_squared,
				    frequencies,
				    grid_point_triplets,
				    triplet_weights,
				    g,
				    temperature,
				    unit_conversion_factor,
				    cutoff_frequency);

  free(fc3_normal_squared);
  
  Py_RETURN_NONE;
}
//
//static PyObject *
//py_set_triplets_integration_weights(PyObject *self, PyObject *args)
//{
//  PyArrayObject* iw_py;
//  PyArrayObject* frequency_points_py;
//  PyArrayObject* relative_grid_address_py;
//  PyArrayObject* weight_correction_py;
//  PyArrayObject* tratra_center_indices_py;
//  PyArrayObject* tetra_mapping_py;
//  PyArrayObject* mesh_py;
//  PyArrayObject* triplets_py;
//  PyArrayObject* frequencies_py;
//  PyArrayObject* grid_address_py;
//  PyArrayObject* bz_map_py;
//  if (!PyArg_ParseTuple(args, "OOOOOOOOOOO",
//			&iw_py,
//			&frequency_points_py,
//			&relative_grid_address_py,
//			&weight_correction_py,
//			&tratra_center_indices_py,
//			&tetra_mapping_py,
//			&mesh_py,
//			&triplets_py,
//			&frequencies_py,
//			&grid_address_py,
//			&bz_map_py)) {
//    return NULL;
//  }
//  const int num_band0 = frequency_points_py->dimensions[0];
//  const int num_band = (int)frequencies_py->dimensions[1];
//  const int num_triplets = (int)triplets_py->dimensions[0];
//  double (*iw)[num_triplets][num_band0][num_band][num_band] = (double (*)[num_triplets][num_band0][num_band][num_band]) iw_py->data;
//  const double *frequency_points = (double*)frequency_points_py->data;
//  SPGCONST int (*relative_grid_address)[20][3] =
//    (int(*)[20][3])relative_grid_address_py->data;
//  const double (*weight_correction)[4] = (double (*)[4])weight_correction_py->data; //(20, 4)
//  const int *tratra_center_indices = (int *)tratra_center_indices_py->data; //(120)
//  const int *mesh = (int*)mesh_py->data;
//  SPGCONST int (*triplets)[3] = (int(*)[3])triplets_py->data;
//
//  SPGCONST int (*grid_address)[3] = (int(*)[3])grid_address_py->data;
//  const double *frequencies = (double*)frequencies_py->data;
//  const int (*tetra_mapping)[120] = (int (*)[120]) tetra_mapping_py->data;
//  const int *bz_map = (int*)bz_map_py->data;
////   const int num_iw = (int)iw_py->dimensions[0];
//
//  int i, j, k, l, b12,  b1, b2, sign, na=0, is_found=0;
//  int tp_relative_grid_address[2][120][20][3];
//  int vertices[2][120][20];
//  int (*avert)[2][20], (*tmap)[120], *tmap_uniq;
//  double f0, f1, f2, g0[120][4], g1[120][4], g2[120][4];
//  double freq_vertices[3][120][20], freq_tetra[3][120][4];
//  double (*iw_all)[3][num_band0][4];
//  char *iw_done, skip[120];
//  avert = (int (*)[2][20]) malloc(sizeof(int) * num_triplets * 120 * 2 * 20);
//  tmap = (int (*)[120]) malloc(sizeof(int) * num_triplets * 120);
//  tmap_uniq = (int *) malloc(sizeof(int) * num_triplets * 120);
//
//  for (i=0; i < num_triplets * 120; i++)
//  {
//    tmap_uniq[i] = -1;
//    for (j = 0; j < 2; j++)
//      for (k = 0; k < 20; k++)
//        avert[i][j][k] = 0;
//  }
//  for (i = 0; i < num_triplets; i++)
//    for (j = 0; j < 120; j++)
//    {
//      tmap[i][j] = -1;
//      }
//
//
//  for (i = 0; i < 2; i++) {
//    sign = 1 - i * 2;
//    for (j = 0; j < 120; j++) {
//      for (k = 0; k < 20; k++) {
//	for (l = 0; l < 3; l++) {
//	  tp_relative_grid_address[i][j][k][l] =
//	    relative_grid_address[j][k][l] * sign;
//	}
//      }
//    }
//  }
//
//  for (i = 0; i < num_triplets; i++) {
//    get_triplet_tetrahedra_vertices(vertices,
//				    tp_relative_grid_address,
//				    mesh,
//				    triplets[i],
//				    grid_address,
//				    bz_map);
//
//    for (j = 0; j < 120; j++)
//    {
//
//      is_found = 0;
//
//      for (k = 0; k < na; k++){
//        if (tetra_mapping[triplets[i][1]][j] == tmap_uniq[k])
//        {
//          is_found = 1;
//          tmap[i][j] = k;
//          break;
//        }
//      }
//
//      if (!is_found){
//
//        for (k = 0; k < 20; k++){
//          avert[na][0][k] = vertices[0][j][k];
//          avert[na][1][k] = vertices[1][j][k];
//        }
//
//        tmap[i][j] = na;
//        tmap_uniq[na] = tetra_mapping[triplets[i][1]][j];
//        na++;
//      }
//
//    }
//
//  }
////  printf("NUMBER OF IRREDUCIBLE TETRA: %d!...\n", na);
//  // na is the number of all tetra hedra
//
//
//  #pragma omp parallel private(i, j, k, l, b1, b2, f0, f1, f2, g0, g1, g2, freq_vertices, freq_tetra, iw_done, skip, iw_all)
//  {
//    iw_done = (char *)malloc(sizeof(char) * na);
//    iw_all = (double (*)[3][num_band0][4])malloc(sizeof(double) * na * 3 * num_band0 * 4);
//    #pragma omp for
//    for (b12 = 0; b12 < num_band * num_band; b12++) {
//      b1 = b12 / num_band; b2 = b12 % num_band;
//      for (i = 0; i < na; i++)
//      {
//        iw_done[i] = 0;
//        for (j = 0; j < num_band0; j++)
//          for (k = 0; k < 4; k++)
//            for (l = 0; l < 3; l++)
//              iw_all[i][l][j][k] =0.;
//      }
//      for (i = 0; i < num_triplets; i++) {
//        for (j = 0; j < 120; j++) {
//          skip[j] = 0;
//          if (iw_done[tmap[i][j]]) {
//            skip[j] = 1;
//            continue;
//          }
//          for (k = 0; k < 20; k++) {
//            l = tmap[i][j];
//            f1 = frequencies[avert[l][0][k] * num_band + b1];
//            f2 = frequencies[avert[l][1][k] * num_band + b2];
//            freq_vertices[0][j][k] = f1 + f2;
//            freq_vertices[1][j][k] = f1 - f2;
//            freq_vertices[2][j][k] = -f1 + f2;
//          }
//
//          //omega_p(4) = omega(20) .dot P(20,4)
//          for (k = 0; k < 4; k++){
//            freq_tetra[0][j][k] = 0;
//            freq_tetra[1][j][k] = 0;
//            freq_tetra[2][j][k] = 0;
//            for (l = 0; l < 20; l++){
//              freq_tetra[0][j][k] += freq_vertices[0][j][l] * weight_correction[l][k];
//              freq_tetra[1][j][k] += freq_vertices[1][j][l] * weight_correction[l][k];
//              freq_tetra[2][j][k] += freq_vertices[2][j][l] * weight_correction[l][k];
//            }
//          }
//        }
//
//        for (j = 0; j < num_band0; j++) {
//          f0 = frequency_points[j];
//          thm_get_integration_weight(g0, f0, freq_tetra[0], skip, 'I');
//          thm_get_integration_weight(g1, f0, freq_tetra[1], skip, 'I');
//          thm_get_integration_weight(g2, f0, freq_tetra[2], skip, 'I');
//          for (k = 0; k < 120; k++)
//          {
//            l = tmap[i][k];
//            if (skip[k]){
//              mat_copy_vector_dn(g0[k], iw_all[l][0][j], 4);
//              mat_copy_vector_dn(g1[k], iw_all[l][1][j], 4);
//              mat_copy_vector_dn(g2[k], iw_all[l][2][j], 4);
//            }
//            else{
//              mat_copy_vector_dn(iw_all[l][0][j], g0[k], 4);
//              mat_copy_vector_dn(iw_all[l][1][j], g1[k], 4);
//              mat_copy_vector_dn(iw_all[l][2][j], g2[k], 4);
//              iw_done[l] = 1;
//            }
//            iw[0][i][j][b1][b2] += mat_multiply_vector_vector_dn(g0[k], weight_correction[tratra_center_indices[k]], 4);
//            iw[1][i][j][b1][b2] += mat_multiply_vector_vector_dn(g1[k], weight_correction[tratra_center_indices[k]], 4);
//            iw[2][i][j][b1][b2] += mat_multiply_vector_vector_dn(g2[k], weight_correction[tratra_center_indices[k]], 4);
//          }
//        }
//      }
//    }
//    free(iw_done);
//    free(iw_all);
//  }
//  free(avert);
//  free(tmap);
//
//  Py_RETURN_NONE;
//}


static PyObject *
py_set_triplets_integration_weights(PyObject *self, PyObject *args)
{
  PyArrayObject* iw_py;
  PyArrayObject* frequency_points_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* weight_correction_py;
  PyArrayObject* tratra_center_indices_py;
  PyArrayObject* mesh_py;
  PyArrayObject* triplets_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* bz_map_py;
  int is_linear;
  if (!PyArg_ParseTuple(args, "OOOOOOOOOOi",
			&iw_py,
			&frequency_points_py,
			&relative_grid_address_py,
			&weight_correction_py,
			&tratra_center_indices_py,
			&mesh_py,
			&triplets_py,
			&frequencies_py,
			&grid_address_py,
			&bz_map_py,
			&is_linear)) {
    return NULL;
  }
  double *iw = (double*) iw_py->data;
  const double *frequency_points = (double*)frequency_points_py->data;
  const int num_band0 = frequency_points_py->dimensions[0];
  SPGCONST int (*relative_grid_address)[20][3] =
    (int(*)[20][3])relative_grid_address_py->data;
  const double (*weight_correction)[4] = (double (*)[4])weight_correction_py->data; //(20, 4)
  const int *tratra_center_indices = (int *)tratra_center_indices_py->data; //(120)
  const int *mesh = (int*)mesh_py->data;
  SPGCONST int (*triplets)[3] = (int(*)[3])triplets_py->data;
  const int num_triplets = (int)triplets_py->dimensions[0];
  SPGCONST int (*grid_address)[3] = (int(*)[3])grid_address_py->data;
  const double *frequencies = (double*)frequencies_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  const int *bz_map = (int*)bz_map_py->data;
//   const int num_iw = (int)iw_py->dimensions[0];

  int i, j, k, l, b1, b2, sign, na, nt;
  int tp_relative_grid_address[2][120][20][3];
  int vertices[2][120][20];
  int adrs_shift, tbbb=num_triplets * num_band0 * num_band * num_band;
  double f0, f1, f2, g0[120][4], g1[120][4], g2[120][4];
  double freq_vertices[3][120][20], freq_tetra[3][120][4];
  na = (is_linear)? 24: 120;
  nt = (is_linear)? 4: 20;
  for (i = 0; i < 2; i++) {
    sign = 1 - i * 2;
    for (j = 0; j < na; j++) {
      for (k = 0; k < nt; k++) {
        for (l = 0; l < 3; l++) {
          tp_relative_grid_address[i][j][k][l] =
            relative_grid_address[j][k][l] * sign;
        }
      }
    }
  }


#pragma omp parallel for private(j, k, l, b1, b2, vertices, adrs_shift, f0, f1, f2, g0, g1, g2, freq_vertices, freq_tetra)
  for (i = 0; i < num_triplets; i++) {
    get_triplet_tetrahedra_vertices(vertices,
				    tp_relative_grid_address,
				    mesh,
				    triplets[i],
				    grid_address,
				    bz_map,
				    is_linear);
    for (b1 = 0; b1 < num_band; b1++) {
      for (b2 = 0; b2 < num_band; b2++) {
        for (j = 0; j < na; j++) {
          for (k = 0; k < nt; k++) {
            f1 = frequencies[vertices[0][j][k] * num_band + b1];
            f2 = frequencies[vertices[1][j][k] * num_band + b2];
            freq_vertices[0][j][k] = f1 + f2;
            freq_vertices[1][j][k] = f1 - f2;
            freq_vertices[2][j][k] = -f1 + f2;
          }
        }

          //omega_p(4) = omega(20) .dot P(20,4)

        get_corrected_frequencies(freq_tetra[0], freq_vertices[0], weight_correction, is_linear);
        get_corrected_frequencies(freq_tetra[1], freq_vertices[1], weight_correction, is_linear);
        get_corrected_frequencies(freq_tetra[2], freq_vertices[2], weight_correction, is_linear);


        for (j = 0; j < num_band0; j++) {
          f0 = frequency_points[j];
          thm_get_integration_weight(g0, f0, freq_tetra[0], is_linear, 'I');
          thm_get_integration_weight(g1, f0, freq_tetra[1], is_linear, 'I');
          thm_get_integration_weight(g2, f0, freq_tetra[2], is_linear, 'I');
          adrs_shift = i * num_band0 * num_band * num_band +
            j * num_band * num_band + b1 * num_band + b2;
          iw[adrs_shift] += get_corrected_integration_weights(g0, tratra_center_indices, weight_correction, is_linear);
          iw[adrs_shift + tbbb] += get_corrected_integration_weights(g1, tratra_center_indices, weight_correction, is_linear);
          iw[adrs_shift + 2 * tbbb] += get_corrected_integration_weights(g2, tratra_center_indices, weight_correction, is_linear);
        }
      }
    }
  }

  Py_RETURN_NONE;
}


//
//static PyObject *
//py_set_triplets_integration_weights_sym(PyObject *self, PyObject *args)
//{
//  PyArrayObject* iw_py;
//  PyArrayObject* relative_grid_address_py;
//  PyArrayObject* weight_correction_py;
//  PyArrayObject* tratra_center_indices_py;
//  PyArrayObject* mesh_py;
//  PyArrayObject* triplets_py;
//  PyArrayObject* frequencies_py;
//  PyArrayObject* grid_address_py;
//  PyArrayObject* bz_map_py;
//  if (!PyArg_ParseTuple(args, "OOOOOOOOO",
//			&iw_py,
//			&relative_grid_address_py,
//			&weight_correction_py,
//			&tratra_center_indices_py,
//			&mesh_py,
//			&triplets_py,
//			&frequencies_py,
//			&grid_address_py,
//			&bz_map_py)) {
//    return NULL;
//  }
//
//  double *iw = (double*)iw_py->data;
//  SPGCONST int (*relative_grid_address)[20][3] =
//    (int(*)[20][3])relative_grid_address_py->data;
//  const double (*weight_correction)[4] = (double (*)[4])weight_correction_py->data; //(20, 4)
//  const int *tratra_center_indices = (int *)tratra_center_indices_py->data; //(120)
//  const int *mesh = (int*)mesh_py->data;
//  SPGCONST int (*triplets)[3] = (int(*)[3])triplets_py->data;
//  const int num_triplets = (int)triplets_py->dimensions[0];
//  SPGCONST int (*grid_address)[3] = (int(*)[3])grid_address_py->data;
//  const double *frequencies = (double*)frequencies_py->data;
//  const int num_band = (int)frequencies_py->dimensions[1];
//  const int *bz_map = (int*)bz_map_py->data;
//  VecINT* triplets_permute = mat_alloc_VecINT(num_triplets);
//  int i, j, k, l, b1, b2, sign, index, tbbb=num_triplets*num_band*num_band*num_band;
//  int tp_relative_grid_address[2][120][20][3];
//  int vertices[2][120][20];
//  int adrs_shift=0;
//  double f0, f1, f2, g0[120][4], g1[120][4], g2[120][4];
//  double freq_vertices[3][120][20], freq_tetra[3][120][4] ;
//
//  for (i = 0; i < 2; i++) {
//    sign = 1 - i * 2;
//    for (j = 0; j < 120; j++) {
//      for (k = 0; k < 20; k++) {
//        for (l = 0; l < 3; l++) {
//          tp_relative_grid_address[i][j][k][l] =
//            relative_grid_address[j][k][l] * sign;
//        }
//      }
//    }
//  }
//  for (index=0; index<3; index++)
//  {
//     for (i=0; i< num_triplets; i++)
//       for (j = 0; j < 3; j++)
//         triplets_permute->vec[i][j] = triplets[i][(j+index) % 3];
//
//#pragma omp parallel for private(j, k, l, b1, b2, vertices, adrs_shift, f0, f1, f2, g0, g1, g2, freq_vertices)
//      for (i = 0; i < num_triplets; i++) {
//        get_triplet_tetrahedra_vertices(vertices,
//                        tp_relative_grid_address,
//                        mesh,
//                        triplets_permute->vec[i],
//                        grid_address,
//                        bz_map);
//        for (b1 = 0; b1 < num_band; b1++) {
//          for (b2 = 0; b2 < num_band; b2++) {
//            for (j = 0; j < 120; j++) {
//              for (k = 0; k < 4; k++) {
//                f1 = frequencies[vertices[0][j][k] * num_band + b1];
//                f2 = frequencies[vertices[1][j][k] * num_band + b2];
//                freq_vertices[0][j][k] = f1 + f2;
//                freq_vertices[1][j][k] = f1 - f2;
//                freq_vertices[2][j][k] = -f1 + f2;
//              }
//            //omega_p(4) = omega(20) .dot P(20,4)
//              for (k = 0; k < 4; k++){
//                freq_tetra[0][j][k] = 0;
//                freq_tetra[1][j][k] = 0;
//                freq_tetra[2][j][k] = 0;
//                for (l = 0; l < 20; l++){
//                  freq_tetra[0][j][k] += freq_vertices[0][j][l] * weight_correction[l][k];
//                  freq_tetra[1][j][k] += freq_vertices[1][j][l] * weight_correction[l][k];
//                  freq_tetra[2][j][k] += freq_vertices[2][j][l] * weight_correction[l][k];
//                }
//	          }
//            }
//            for (j = 0; j < num_band; j++) {
//              f0 = frequencies[triplets_permute->vec[i][0] * num_band + j];
//              thm_get_integration_weight(g0, f0, freq_tetra[0], 'I');
//              thm_get_integration_weight(g1, f0, freq_tetra[1], 'I');
//              thm_get_integration_weight(g2, f0, freq_tetra[2], 'I');
//              if (index==0)
//                adrs_shift = i * num_band * num_band * num_band +
//                    j * num_band * num_band + b1 * num_band + b2;
//              else if (index==1)
//                adrs_shift = i * num_band * num_band * num_band +
//                    b2 * num_band * num_band + j * num_band + b1;
//              else if (index==2)
//                adrs_shift = i * num_band * num_band * num_band +
//                    b1 * num_band * num_band + b2 * num_band + j;
//              for (k = 0; k < 120; k++){
//                iw[index*tbbb+adrs_shift] +=
//                   mat_multiply_vector_vector_dn(g0[k], weight_correction[tratra_center_indices[k]], 4);
//                iw[(1+index)%3 *tbbb+adrs_shift] +=
//                   mat_multiply_vector_vector_dn(g1[k], weight_correction[tratra_center_indices[k]], 4);
//                iw[(2+index)%3 *tbbb+adrs_shift] +=
//                   mat_multiply_vector_vector_dn(g2[k], weight_correction[tratra_center_indices[k]], 4);
//
//              }
//            }
//          }
//        }
//      }
//  }
//  mat_free_VecINT(triplets_permute);
//  Py_RETURN_NONE;
//}

static PyObject * py_get_neighboring_gird_points(PyObject *self, PyObject *args)
{
  PyArrayObject* relative_grid_points_py;
  PyArrayObject* grid_points_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* bz_map_py;
  if (!PyArg_ParseTuple(args, "OOOOOO",
			&relative_grid_points_py,
			&grid_points_py,
			&relative_grid_address_py,
			&mesh_py,
			&grid_address_py,
			&bz_map_py)) {
    return NULL;
  }

  int* relative_grid_points = (int*)relative_grid_points_py->data;
  const int *grid_points = (int*)grid_points_py->data;
  const int num_grid_points = (int)grid_points_py->dimensions[0];
  SPGCONST int (*relative_grid_address)[3] =
    (int(*)[3])relative_grid_address_py->data;
  const int num_relative_grid_address = relative_grid_address_py->dimensions[0];
  const int *mesh = (int*)mesh_py->data;
  SPGCONST int (*grid_address)[3] = (int(*)[3])grid_address_py->data;
  const int *bz_map = (int*)bz_map_py->data;
  int i;
#pragma omp parallel for
  for (i = 0; i < num_grid_points; i++) {
    get_neighboring_grid_points
      (relative_grid_points + i * num_relative_grid_address,
       grid_points[i],
       relative_grid_address,
       num_relative_grid_address,
       mesh,
       grid_address,
       bz_map);
  }
  
  Py_RETURN_NONE;
}

static PyObject * py_get_uniq_neighboring_tetrahedra(PyObject *self, PyObject *args)
{
  PyArrayObject* uniq_tetrahedra_py;
  PyArrayObject* tetrahedra_mapping_py;
  PyArrayObject* grid_points_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* bz_map_py;
  if (!PyArg_ParseTuple(args, "OOOOOOO",
			&uniq_tetrahedra_py,
			&tetrahedra_mapping_py,
			&grid_points_py,
			&relative_grid_address_py,
			&mesh_py,
			&grid_address_py,
			&bz_map_py)) {
    return NULL;
  }
  int (*uniq_tetrahedra)[20] = (int (*)[20])uniq_tetrahedra_py->data;
  int (*tetrahedra_mapping)[120] = (int (*)[120])tetrahedra_mapping_py->data;
  const int *grid_points = (int*)grid_points_py->data;
  const int num_grid_points = (int)grid_points_py->dimensions[0];
  SPGCONST int (*relative_grid_address)[20][3] =
    (int(*)[20][3])relative_grid_address_py->data;
  const int *mesh = (int*)mesh_py->data;
  SPGCONST int (*grid_address)[3] = (int(*)[3])grid_address_py->data;
  const int *bz_map = (int*)bz_map_py->data;
  int i, j, k, is_found, na=0;
  int relative_grid_points[20];

  for (i = 0; i < num_grid_points; i++) {

    for (j = 0; j < 120; j++)
    {
      get_neighboring_grid_points
        (relative_grid_points,
         grid_points[i],
         relative_grid_address[j],
         20,
         mesh,
         grid_address,
         bz_map);
      is_found = 0;
      for (k = 0; k < na; k++){
        if (relative_grid_points[0] == uniq_tetrahedra[k][0] &&
          relative_grid_points[1] == uniq_tetrahedra[k][1] &&
          relative_grid_points[2] == uniq_tetrahedra[k][2] &&
          relative_grid_points[3] == uniq_tetrahedra[k][3])
        {
          is_found = 1;
          tetrahedra_mapping[i][j] = k;
          break;
        }
      }

      if (!is_found){
        for (k = 0; k < 20; k++)
          uniq_tetrahedra[na][k] = relative_grid_points[k];
        tetrahedra_mapping[i][j] = na;
        na++;
      }
    }
  }

  return PyInt_FromLong((long)na);
}


static void get_triplet_tetrahedra_vertices
  (int vertices[2][120][20],
   SPGCONST int relative_grid_address[2][120][20][3],
   const int mesh[3],
   const int triplet[3],
   SPGCONST int grid_address[][3],
   const int bz_map[],
   const int is_linear)
{
  int i, j, nt, na;
  if (is_linear) {
    na = 24;
    nt = 4;
  }
  else{
    na = 120;
    nt = 20;
  }
  for (i = 0; i < 2; i++) {
    for (j = 0; j < na; j++) {
      get_neighboring_grid_points(vertices[i][j],
				      triplet[i + 1],
				      relative_grid_address[i][j],
				      nt,
				      mesh,
				      grid_address,
				      bz_map);
    }
  }
}

static void get_neighboring_grid_points(int neighboring_grid_points[],
				     const int grid_point,
				     SPGCONST int relative_grid_address[][3],
				     const int num_relative_grid_address,
				     const int mesh[3],
				     SPGCONST int bz_grid_address[][3],
				     const int bz_map[])
{
  int mesh_double[3], bzmesh[3], bzmesh_double[3],
    address_double[3], bz_address_double[3];
  int i, j, bz_gp;

  for (i = 0; i < 3; i++) {
    mesh_double[i] = mesh[i] * 2;
    bzmesh[i] = mesh[i] * 2;
    bzmesh_double[i] = bzmesh[i] * 2;
  }
  for (i = 0; i < num_relative_grid_address; i++) {
    for (j = 0; j < 3; j++) {
      address_double[j] = (bz_grid_address[grid_point][j] +
			   relative_grid_address[i][j]) * 2;
      bz_address_double[j] = address_double[j];
    }
    // get vector modulo
//     for (j = 0; j < 3; j++) {
//       address_double[j] = address_double[j] % mesh_double[j];
//       if (address_double[j] < 0)
// 	address_double[j] += mesh_double[j];
//     }
    
    get_vector_modulo(bz_address_double, bzmesh_double);
    bz_gp = bz_map[get_grid_point_double_mesh(bz_address_double, bzmesh)];
    if (bz_gp == -1) {
      get_vector_modulo(address_double, mesh_double);
      neighboring_grid_points[i] =
	get_grid_point_double_mesh(address_double, mesh);
    } else {
      neighboring_grid_points[i] = bz_gp;
    }
  }
}
static int get_grid_point_double_mesh(const int address_double[3],
				      const int mesh[3])
{
  int i, address[3];

  for (i = 0; i < 3; i++) {
    if (address_double[i] % 2 == 0) {
      address[i] = address_double[i] / 2;
    } else {
      address[i] = (address_double[i] - 1) / 2;
    }
  }
  return get_grid_point_single_mesh(address, mesh);
}

static int get_grid_point_single_mesh(const int address[3],
				      const int mesh[3])
{  
#ifndef GRID_ORDER_XYZ
  return address[2] * mesh[0] * mesh[1] + address[1] * mesh[0] + address[0];
#else
  return address[0] * mesh[1] * mesh[2] + address[1] * mesh[2] + address[2];
#endif  
}

static void get_vector_modulo(int v[3], const int m[3])
{
  int i;

  for (i = 0; i < 3; i++) {
    v[i] = v[i] % m[i];

    if (v[i] < 0)
      v[i] += m[i];
  }
}
static void get_corrected_frequencies(double freq_tetra[120][4],
                                      const double freq_vertices[120][20],
                                      const double weight_correction[20][4],
                                      const int is_linear)
{
  int i, j;
  int na = (is_linear)? 24: 120;
  int nt = (is_linear)? 4: 20;
  for (i = 0; i < na; i++)
    for (j = 0; j < 4; j++)
      freq_tetra[i][j] = 0.;
  for (i = 0; i < na; i++)
    for (j = 0; j < nt; j++)
    {
      freq_tetra[i][0] += freq_vertices[i][j] * weight_correction[j][0];
      freq_tetra[i][1] += freq_vertices[i][j] * weight_correction[j][1];
      freq_tetra[i][2] += freq_vertices[i][j] * weight_correction[j][2];
      freq_tetra[i][3] += freq_vertices[i][j] * weight_correction[j][3];
    }
}

static double get_corrected_integration_weights(const double tetra_weights[120][4],
                                                const int center_indices[120],
                                                const double weight_correction[20][4],
                                                const int is_linear)
{
  int i, j, k;
  double iw=0.;
  int na = (is_linear)? 24: 120;
  for (i = 0; i < na; i++)
  {
    k = center_indices[i];
    for (j = 0; j < 4; j++)
      iw += tetra_weights[i][j] * weight_correction[k][j];
  }
  return iw;
}




// static void get_neighboring_grid_points(int neighboring_grid_points[],
// 				     const int grid_point,
// 				     SPGCONST int relative_grid_address[][3],
// 				     const int num_relative_grid_address,
// 				     const int mesh[3],
// 				     SPGCONST int grid_address[][3])
// {
//   int mesh_double[3],
//     address_double[3],address[3];
//   int i, j;
// 
//   for (i = 0; i < 3; i++) {
//     mesh_double[i] = mesh[i] * 2;
//   }
//   for (i = 0; i < num_relative_grid_address; i++) {
//     for (j = 0; j < 3; j++)
//       address_double[j] = (grid_address[grid_point][j] +
// 			   relative_grid_address[i][j]) * 2;
// 
//     // get vector modulo
//     for (j = 0; j < 3; j++) {
//       address_double[j] = address_double[j] % mesh_double[j];
//       if (address_double[j] < 0)
// 	address_double[j] += mesh_double[j];
//     }
//     
//     for (j = 0; j < 3; j++)
//       if (address_double[j] % 2 == 0)
// 	address[j] = address_double[j] / 2;
//       else
// 	address[j] = (address_double[j] - 1) / 2;
//     neighboring_grid_points[i] = address[2] * mesh[0] * mesh[1] + address[1] * mesh[0] + address[0];
//   }
// }
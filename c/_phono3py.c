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
static PyObject * py_get_decay_channel_thm(PyObject *self, PyObject *args);
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
static PyObject *
py_set_triplets_integration_weights_1D(PyObject *self, PyObject *args);
static PyObject *
py_set_triplets_integration_weights_1D_frequency_points(PyObject *self, PyObject *args);
static PyObject *
py_set_triplets_integration_weights(PyObject *self, PyObject *args);
static PyObject *
py_set_triplets_integration_weights_frequency_points(PyObject *self, PyObject *args);
static PyObject *py_interaction_degeneracy_grid(PyObject *self, PyObject *args);
static void get_triplet_tetrahedra_vertices
  (int vertices[2][24][4],
   SPGCONST int relative_grid_address[2][24][4][3],
   const int mesh[3],
   const int triplet[3],
   SPGCONST int grid_address[][3],
   const int bz_map[]);
static void get_triplet_tetrahedra_vertices_1D
  (int vertices[2][2][2],
   SPGCONST int relative_grid_address[2][2][2][3],
   const int mesh[3],
   const int triplet[3],
   SPGCONST int grid_address[][3],
   const int bz_map[]);
static void get_vector_modulo(int v[3], const int m[3]);
static int get_grid_point_double_mesh(const int address_double[3],
				      const int mesh[3]);
static int get_grid_point_single_mesh(const int address[3],
				      const int mesh[3]);
static void get_neighboring_grid_points(int neighboring_grid_points[],
				     const int grid_point,
				     SPGCONST int relative_grid_address[][3],
				     const int num_relative_grid_address,
				     const int mesh[3],
				     SPGCONST int grid_address[][3], 
				     const int bz_map[]);
static void set_triplet_integration_1D_at_triplet(double *iw_triplet, //shape: num_band0, num_band1, num_band2
                                                  const int triplet[3],
                                                  const int vertices[2][2][2], //neighbors
                                                  const double *frequencies,
                                                  const int *band_indices[3],
                                                  const int num_bands[3],
                                                  const int num_band);
static PyMethodDef functions[] = {
  {"joint_dos", py_get_jointDOS, METH_VARARGS, "Calculate joint density of states"},
  {"decay_channel", py_get_decay_channel, METH_VARARGS, "Calculate decay of phonons"},
  {"decay_channel_thm", py_get_decay_channel_thm, METH_VARARGS, "Calculate decay of phonons with integration weights given"},
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
  {"interaction_degeneracy_grid", py_interaction_degeneracy_grid, METH_VARARGS, "Integration symmetrization considering the degeneracy" },
  {"interaction_from_reduced", py_interaction_from_reduced, METH_VARARGS, "interaction strength from reduced triplets" },
  {"perturbation_next", py_perturbation_next, METH_VARARGS, "Calculate the next perturbation flow"},
  {"thermal_conductivity_at_grid",py_get_thermal_conductivity_at_grid, METH_VARARGS, "thermal conductivity calculation at a grid point" },
  {"thermal_conductivity",py_get_thermal_conductivity, METH_VARARGS, "thermal conductivity calculation at all grid points" },
  {"phonon_multiply_dmatrix_gbb_dvector_gb3",py_phonon_multiply_dmatrix_gbb_dvector_gb3, METH_VARARGS,
    "phonon multiplicity between a double matrix(grid, band, band) and another vector (grid, band, 3)"},
  {"triplets_integration_weights", py_set_triplets_integration_weights, METH_VARARGS,
   "Integration weights of tetrahedron method for triplets"},
  {"triplets_integration_weights_1D", py_set_triplets_integration_weights_1D, METH_VARARGS,
   "Integration weights of tetrahedron method in 1D for triplets"},
  {"triplets_integration_weights_1D_fpoints", py_set_triplets_integration_weights_1D_frequency_points, METH_VARARGS,
   "Integration weights of tetrahedron method in 1D for triplets"},
  {"triplets_integration_weights", py_set_triplets_integration_weights, METH_VARARGS,
   "Integration weights of tetrahedron method for triplets with interchange symmetry"},
  {"triplets_integration_weights_fpoints", py_set_triplets_integration_weights_frequency_points, METH_VARARGS,
   "Integration weights of tetrahedron method for triplets with interchange symmetry"},
  {"triplets_integration_weights_with_sigma",py_set_triplets_integration_weights_with_sigma, METH_VARARGS,
   "Integration weights of smearing method for triplets"},
   {"triplets_integration_weights_with_asigma",py_set_triplets_integration_weights_with_asigma, METH_VARARGS,
   "Integration weights of smearing method (with sigma self-adaption) for triplets"},
  {"neighboring_grid_points", py_get_neighboring_gird_points, METH_VARARGS,
   "Neighboring grid points by relative grid addresses"},
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
  PyArrayObject* g_skip_py;
  PyArrayObject* atomic_masses;
  PyArrayObject* p2s_map;
  PyArrayObject* s2p_map;
  PyArrayObject* band_indicies_py;
  double cutoff_frequency, cutoff_hfrequency;
  double cutoff_delta;
  int symmetrize_fc3_q;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOiddd",
			&fc3_normal_squared_py,
			&frequencies,
			&eigenvectors,
			&grid_point_triplets,
			&grid_address_py,
			&mesh_py,
			&fc3_py,
            &atc_py,
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

  if (!PyArg_ParseTuple(args, "OOOOOdddddddd",
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

  get_imag_self_energy(gamma,
		       fc3_normal_squared,
		       fpoint,
		       frequencies,
		       grid_point_triplets,
		       triplet_weights,
		       asigma_py,
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

static PyObject * py_get_decay_channel_thm(PyObject *self, PyObject *args)
{
  PyArrayObject* decay_values;
  PyArrayObject* amplitudes;
  PyArrayObject* omegas;
  PyArrayObject* frequencies;
  PyArrayObject* integration_weights;
  double sigma, t, cutoff_frequency;

  if (!PyArg_ParseTuple(args, "OOOOOdd",
			&decay_values,
			&amplitudes,
			&frequencies,
			&omegas,
			&integration_weights,
			&cutoff_frequency,
			&t)) {
    return NULL;
  }


  double* decay = (double*)decay_values->data;
  const double* amp = (double*)amplitudes->data;
  const double* f = (double*)frequencies->data;
  const double* o = (double*)omegas->data;
  const double* g = (double *)integration_weights->data;
  const int num_band = (int)amplitudes->dimensions[2];
  const int num_triplet = (int)amplitudes->dimensions[0];
  const int num_omega = (int)omegas->dimensions[0];

  get_decay_channels_thm(decay,
                         num_omega,
                         num_triplet,
                         num_band,
                         o,
                         f,
                         amp,
                         g,
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
  PyArrayObject *py_collision;
  PyArrayObject *py_interaction;
  PyArrayObject *py_frequency;
  PyArrayObject *py_g;
  double cutoff_frequency;
  double temperature;
  if (!PyArg_ParseTuple(args, "OOOOdd",
			&py_collision,
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
  double *collision=(double*)py_collision->data;
  get_collision_at_all_band(collision,
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
  PyArrayObject *py_collision;
  PyArrayObject *py_occupation;
  PyArrayObject *py_interaction;
  PyArrayObject *py_frequency;
  PyArrayObject *py_integration_weight;
  double cutoff_frequency;
  if (!PyArg_ParseTuple(args, "OOOOOd",
			&py_collision,
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
  double *collision=(double*)py_collision->data;
  get_collision_at_all_band_permute(collision,
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
  PyArrayObject *py_collision_at_grid;
  PyArrayObject *py_collision_all;
  PyArrayObject *py_triplet_mapping;
  PyArrayObject *py_triplet_sequence;
  

  if (!PyArg_ParseTuple(args, "OOOO",
			&py_collision_at_grid,
			&py_collision_all,
			&py_triplet_mapping,
			&py_triplet_sequence))
    return NULL;
  const int num_triplet = (int)py_collision_at_grid->dimensions[0];
  const int num_band = (int) py_collision_at_grid->dimensions[1];
  
  const double *collision_all = (double*)py_collision_all->data;
  const int *triplet_mapping = (int*)py_triplet_mapping->data;
  const char *triplet_sequence = (char*)py_triplet_sequence->data;
  double *collision=(double*)py_collision_at_grid->data;
  
  get_collision_from_reduced(collision, collision_all, triplet_mapping,triplet_sequence,num_triplet, num_band);
  Py_RETURN_NONE;
}

static PyObject *py_collision_degeneracy(PyObject *self, PyObject *args)
{
  PyArrayObject *py_collision;
  PyArrayObject *py_degeneracies_all;
  PyArrayObject *py_triplets;
  int is_permute;

  if (!PyArg_ParseTuple(args, "OOOi",
			&py_collision,
			&py_degeneracies_all,
			&py_triplets,
			&is_permute))
    return NULL;
  const int num_triplet = (int)py_triplets->dimensions[0];
  const int (*triplets)[3] = (int (*)[3]) py_triplets->data;
  const int num_band = (int) py_degeneracies_all->dimensions[1];
  const int *degeneracies_all = (int*)py_degeneracies_all->data;
  double *collision=(double*)py_collision->data;
  collision_degeneracy(collision, degeneracies_all, triplets, num_triplet, num_band, is_permute);
  Py_RETURN_NONE;
}

static PyObject *py_interaction_degeneracy_grid(PyObject *self, PyObject *args)
{
  PyArrayObject *py_interaction;
  PyArrayObject *py_degeneracies_all;
  PyArrayObject *py_triplets_at_grid;
  PyArrayObject *py_band_indices;


  if (!PyArg_ParseTuple(args, "OOOO",
			&py_interaction,
			&py_degeneracies_all,
			&py_triplets_at_grid,
			&py_band_indices))
    return NULL;
  const int num_triplets = (int)py_triplets_at_grid->dimensions[0];
  const int num_band = (int) py_interaction->dimensions[2];
  const int num_band0 = (int) py_band_indices->dimensions[0];
  const int* band_indices = (int*) py_band_indices->data;
  const int *degeneracies_all = (int*)py_degeneracies_all->data;
  const int (*triplets_at_grid)[3] = (int (*)[3])py_triplets_at_grid->data;
  double *interaction=(double*)py_interaction->data;
  interaction_degeneracy_grid(interaction,degeneracies_all, triplets_at_grid, band_indices, num_triplets, num_band0, num_band);
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
  const int num_band0 = (int) py_amplitude_at_grid->dimensions[1];
  const int num_band = (int) py_amplitude_at_grid->dimensions[2];
  
  const double *amplitude_all = (double*)py_amplitude_all->data;
  const int *triplet_mapping = (int*)py_triplet_mapping->data;
  const char *triplet_sequence = (char*)py_triplet_sequence->data;
  double *amplitude=(double*)py_amplitude_at_grid->data;
  
  get_interaction_from_reduced(amplitude, amplitude_all, triplet_mapping,triplet_sequence,num_triplet, num_band0, num_band);
  Py_RETURN_NONE;
}

static PyObject *py_perturbation_next(PyObject *self, PyObject *args)
{
  PyArrayObject *py_collision;
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
			&py_collision,
			&py_convergence,
			&py_q1s,
			&py_cart_inv_rot_sum))
    return NULL;
  const int num_triplet = (int)py_collision->dimensions[0];
  const int num_band = (int) py_collision->dimensions[1];
  const int num_grid_points = (int)py_spg_mapping_index->dimensions[0];
  
  const int *q1s = (int*) py_q1s->data;
  const double *collision=(double*)py_collision->data;
  const int *convergence=(int*)py_convergence->data;
  const int *spg_mapping_index=(int*)py_spg_mapping_index->data;
  const double *F_prev = (double*) py_F_prev->data;
  const double *cart_inv_rot_sum = (double*) py_cart_inv_rot_sum->data;
  double *pf_sum=(double*)py_pf_sum->data;
  get_next_perturbation_at_all_bands(pf_sum, //perturbation flow summation at all triplets
				     F_prev,
				     collision,
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
  const int num_iw = (int)iw_py->dimensions[0];
  
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
  const int num_iw = (int)iw_py->dimensions[0];
  const int tbbb = num_triplets * num_band0 * num_band * num_band;
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
          adrs_shift = i * num_band0 * num_band * num_band +
	        j * num_band * num_band + k * num_band + l;
          if (fabs(f0 - f1 - f2) < 10 * sigma)
          {
	        g0 = gaussian(f0 - f1 - f2, sigma);
	        iw[adrs_shift] = g0;
	      }
	      if (fabs(f0 - f1 + f2) < 10 * sigma)
          {
	        g1 = gaussian(f0 - f1 + f2, sigma);
	        iw[tbbb + adrs_shift] = g1;
	      }

	      if (fabs(f0 + f1 - f2) < 10 * sigma)
          {
	        g2 = gaussian(f0 + f1 - f2, sigma);
	        iw[2 * tbbb + adrs_shift] = g2;
	      }
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
  if (!PyArg_ParseTuple(args, "OOOOOOOO",
			&iw_py,
			&frequency_points_py,
			&relative_grid_address_py,
			&mesh_py,
			&grid_points_py,
			&frequencies_py,
			&grid_address_py, 
			&bz_map_py)) {
    return NULL;
  }

  double *iw = (double*)iw_py->data;
  const double *frequency_points = (double*)frequency_points_py->data;
  const int num_band0 = frequency_points_py->dimensions[0];
  SPGCONST int (*relative_grid_address)[4][3] =
    (int(*)[4][3])relative_grid_address_py->data;
  const int *mesh = (int*)mesh_py->data;
  SPGCONST int *grid_points = (int*)grid_points_py->data;
  const int num_gp = (int)grid_points_py->dimensions[0];
  SPGCONST int (*grid_address)[3] = (int(*)[3])grid_address_py->data;
  const double *frequencies = (double*)frequencies_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  const int *bz_map = (int*)bz_map_py->data;
  int i, j, k, bi;
  int vertices[24][4];
  double freq_vertices[24][4];

#pragma omp parallel for private(j, k, bi, vertices, freq_vertices)
  for (i = 0; i < num_gp; i++) {
    for (j = 0; j < 24; j++) {
      get_neighboring_grid_points(vertices[j],
				      grid_points[i],
				      relative_grid_address[j],
				      4,
				      mesh,
				      grid_address,
				      bz_map);
    }
    for (bi = 0; bi < num_band; bi++) {
      for (j = 0; j < 24; j++) {
	for (k = 0; k < 4; k++) {
	  freq_vertices[j][k] = frequencies[vertices[j][k] * num_band + bi];
	}
      }
      for (j = 0; j < num_band0; j++) {
	iw[i * num_band0 * num_band + j * num_band + bi] =
	  thm_get_integration_weight(frequency_points[j], freq_vertices, 'I');
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
  PyArrayObject* band_indices_py;
  double unit_conversion_factor, cutoff_frequency, temperature;

  if (!PyArg_ParseTuple(args, "OOOOOdOOdd",
			&gamma_py,
			&fc3_normal_squared_py,
			&grid_point_triplets_py,
			&triplet_weights_py,
			&frequencies_py,
			&temperature,
			&g_py,
			&band_indices_py,
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
  const int* band_indices = (int*) band_indices_py->data;

  get_thm_imag_self_energy_at_bands(gamma,
				    fc3_normal_squared,
				    frequencies,
				    grid_point_triplets,
				    triplet_weights,
				    g,
				    band_indices,
				    temperature,
				    unit_conversion_factor,
				    cutoff_frequency);

  free(fc3_normal_squared);
  
  Py_RETURN_NONE;
}

static PyObject *
py_set_triplets_integration_weights_frequency_points(PyObject *self, PyObject *args)
{
  PyArrayObject* iw_py;
  PyArrayObject* frequency_points_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* triplets_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* bz_map_py;
  if (!PyArg_ParseTuple(args, "OOOOOOOO",
			&iw_py,
			&frequency_points_py,
			&relative_grid_address_py,
			&mesh_py,
			&triplets_py,
			&frequencies_py,
			&grid_address_py,
			&bz_map_py)) {
    return NULL;
  }

  double *iw = (double*)iw_py->data;
  const double *frequency_points = (double*)frequency_points_py->data;
  const int num_band0 = frequency_points_py->dimensions[0];
  SPGCONST int (*relative_grid_address)[4][3] =
    (int(*)[4][3])relative_grid_address_py->data;
  const int *mesh = (int*)mesh_py->data;
  SPGCONST int (*triplets)[3] = (int(*)[3])triplets_py->data;
  const int num_triplets = (int)triplets_py->dimensions[0];
  SPGCONST int (*grid_address)[3] = (int(*)[3])grid_address_py->data;
  const double *frequencies = (double*)frequencies_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  const int *bz_map = (int*)bz_map_py->data;
//   const int num_iw = (int)iw_py->dimensions[0];

  int i, j, k, l, b1, b2, sign;
  int tp_relative_grid_address[2][24][4][3];
  int vertices[2][24][4];
  int adrs_shift;
  double f0, f1, f2, g0, g1, g2;
  double freq_vertices[3][24][4];
    
  for (i = 0; i < 2; i++) {
    sign = 1 - i * 2;
    for (j = 0; j < 24; j++) {
      for (k = 0; k < 4; k++) {
	for (l = 0; l < 3; l++) {
	  tp_relative_grid_address[i][j][k][l] = 
	    relative_grid_address[j][k][l] * sign;
	}
      }
    }
  }

#pragma omp parallel for private(j, k, b1, b2, vertices, adrs_shift, f0, f1, f2, g0, g1, g2, freq_vertices)
  for (i = 0; i < num_triplets; i++) {
    get_triplet_tetrahedra_vertices(vertices,
				    tp_relative_grid_address,
				    mesh,
				    triplets[i],
				    grid_address, 
				    bz_map);
    for (b1 = 0; b1 < num_band; b1++) {
      for (b2 = 0; b2 < num_band; b2++) {
	for (j = 0; j < 24; j++) {
	  for (k = 0; k < 4; k++) {
	    f1 = frequencies[vertices[0][j][k] * num_band + b1];
	    f2 = frequencies[vertices[1][j][k] * num_band + b2];
	    freq_vertices[0][j][k] = f1 + f2;
	    freq_vertices[1][j][k] = f1 - f2;
	    freq_vertices[2][j][k] = -f1 + f2;
	  }
	}
	for (j = 0; j < num_band0; j++) {
	  f0 = frequency_points[j];
	  g0 = thm_get_integration_weight(f0, freq_vertices[0], 'I');
	  g1 = thm_get_integration_weight(f0, freq_vertices[1], 'I');
	  g2 = thm_get_integration_weight(f0, freq_vertices[2], 'I');
	  adrs_shift = i * num_band0 * num_band * num_band +
	    j * num_band * num_band + b1 * num_band + b2;
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
py_set_triplets_integration_weights_1D_frequency_points(PyObject *self, PyObject *args)
{
  PyArrayObject* iw_py;
  PyArrayObject* frequency_points_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* triplets_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* bz_map_py;
  if (!PyArg_ParseTuple(args, "OOOOOOOO",
			&iw_py,
			&frequency_points_py,
			&relative_grid_address_py,
			&mesh_py,
			&triplets_py,
			&frequencies_py,
			&grid_address_py,
			&bz_map_py)) {
    return NULL;
  }

  double *iw = (double*)iw_py->data;
  const double *frequency_points = (double*)frequency_points_py->data;
  const int num_band0 = frequency_points_py->dimensions[0];
  SPGCONST int (*relative_grid_address)[2][3] =
    (int(*)[2][3])relative_grid_address_py->data;
  const int *mesh = (int*)mesh_py->data;
  SPGCONST int (*triplets)[3] = (int(*)[3])triplets_py->data;
  const int num_triplets = (int)triplets_py->dimensions[0];
  SPGCONST int (*grid_address)[3] = (int(*)[3])grid_address_py->data;
  const double *frequencies = (double*)frequencies_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  const int *bz_map = (int*)bz_map_py->data;
//   const int num_iw = (int)iw_py->dimensions[0];

  int i, j, k, l, b1, b2, sign;
  int tp_relative_grid_address[2][2][2][3];
  int vertices[2][2][2];
  int adrs_shift;
  double f0, f1, f2, g0, g1, g2;
  double freq_vertices[3][2][2];

  for (i = 0; i < 2; i++) {
    sign = 1 - i * 2;
    for (j = 0; j < 2; j++) {
      for (k = 0; k < 2; k++) {
	for (l = 0; l < 3; l++) {
	  tp_relative_grid_address[i][j][k][l] =
	    relative_grid_address[j][k][l] * sign;
	}
      }
    }
  }

#pragma omp parallel for private(j, k, b1, b2, vertices, adrs_shift, f0, f1, f2, g0, g1, g2, freq_vertices)
  for (i = 0; i < num_triplets; i++) {
    get_triplet_tetrahedra_vertices_1D(vertices,
				    tp_relative_grid_address,
				    mesh,
				    triplets[i],
				    grid_address,
				    bz_map);
    for (b1 = 0; b1 < num_band; b1++) {
      for (b2 = 0; b2 < num_band; b2++) {
	for (j = 0; j < 2; j++) {
	  for (k = 0; k < 2; k++) {
	    f1 = frequencies[vertices[0][j][k] * num_band + b1];
	    f2 = frequencies[vertices[1][j][k] * num_band + b2];
	    freq_vertices[0][j][k] = f1 + f2;
	    freq_vertices[1][j][k] = f1 - f2;
	    freq_vertices[2][j][k] = -f1 + f2;
	  }
	}
	for (j = 0; j < num_band0; j++) {
	  f0 = frequency_points[j];
	  g0 = thm_get_integration_weight_1D(f0, freq_vertices[0], 'I');
	  g1 = thm_get_integration_weight_1D(f0, freq_vertices[1], 'I');
	  g2 = thm_get_integration_weight_1D(f0, freq_vertices[2], 'I');
	  adrs_shift = i * num_band0 * num_band * num_band +
	    j * num_band * num_band + b1 * num_band + b2;
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
//
//static PyObject *
//py_set_triplets_integration_weights_1D(PyObject *self, PyObject *args)
//{
//  PyArrayObject* iw_py;
//  PyArrayObject* relative_grid_address_py;
//  PyArrayObject* mesh_py;
//  PyArrayObject* triplets_py;
//  PyArrayObject* frequencies_py;
//  PyArrayObject* band_indices_py;
//  PyArrayObject* grid_address_py;
//  PyArrayObject* bz_map_py;
//  int is_sym;
//  if (!PyArg_ParseTuple(args, "OOOOOOOOi",
//			&iw_py,
//			&relative_grid_address_py,
//			&mesh_py,
//			&triplets_py,
//			&frequencies_py,
//			&band_indices_py,
//			&grid_address_py,
//			&bz_map_py,
//			&is_sym)) {
//    return NULL;
//  }
//
//  double *iw = (double*)iw_py->data;
//  const int num_band0 = band_indices_py->dimensions[0];
//  const int* band_indices = band_indices_py->data;
//  SPGCONST int (*relative_grid_address)[2][3] =
//    (int(*)[2][3])relative_grid_address_py->data;
//  const int *mesh = (int*)mesh_py->data;
//  SPGCONST int (*triplets)[3] = (int(*)[3])triplets_py->data;
//  const int num_triplets = (int)triplets_py->dimensions[0];
//  SPGCONST int (*grid_address)[3] = (int(*)[3])grid_address_py->data;
//  const double *frequencies = (double*)frequencies_py->data;
//  const int num_band = (int)frequencies_py->dimensions[1];
//  const int *bz_map = (int*)bz_map_py->data;
////   const int num_iw = (int)iw_py->dimensions[0];
//  int nb[3][3] = {{num_band0, num_band, num_band},
//                  {num_band, num_band, num_band0},
//                  {num_band, num_band0, num_band}};
//  int i, j, k, l, m, n, b0, b1, b2, index, sign, tbbb=num_triplets*num_band0*num_band*num_band;
//  int tp_relative_grid_address[2][2][2][3];
//  int vertices[2][2][2];
//  int adrs_shift;
//  double f0, f1, f2, g0, g1, g2, *iw_tmp;
//  double freq_vertices[3][2][2];
//  int triplets_permute[num_triplets][3];
//  for (i = 0; i < 2; i++) {
//    sign = 1 - i * 2;
//    for (j = 0; j < 2; j++) {
//      for (k = 0; k < 2; k++) {
//	for (l = 0; l < 3; l++) {
//	  tp_relative_grid_address[i][j][k][l] =
//	    relative_grid_address[j][k][l] * sign;
//	}
//      }
//    }
//  }
//  if (is_sym)
//    iw_tmp = (double *)malloc(sizeof(double) * 3 * num_triplets * 3 * num_band0 * num_band * num_band);
//  else
//    iw_tmp = (double *)malloc(sizeof(double) * num_triplets * 3 * num_band0 * num_band * num_band);
//
//  if (!iw_tmp) printf("Error, %s, %d\n", __FILE__, __LINE__);
//  for (index=0; index < (is_sym? 3: 1); index++)
//  { // 012->120->201
//    for (i=0; i< num_triplets; i++)
//      for (j = 0; j < 3; j++)
//        triplets_permute[i][j] = triplets[i][(j+index) % 3];
//    #pragma omp parallel for private(j, k, l, m, n, b0, b1, b2, vertices, adrs_shift, f0, f1, f2, g0, g1, g2, freq_vertices)
//    for (i = 0; i < num_triplets; i++) {
//      get_triplet_tetrahedra_vertices_1D(vertices,
//                      tp_relative_grid_address,
//                      mesh,
//                      triplets_permute[i],
//                      grid_address,
//                      bz_map);
//      for (m = 0; m < nb[index][1]; m++) {
//        b1 = (nb[index][1] == num_band0)?  band_indices[m]: m;
//        for (n = 0; n < nb[index][2]; n++) {
//          b2 = (nb[index][2] == num_band0)?  band_indices[n]: n;
//          for (j = 0; j < 2; j++) {
//            for (k = 0; k < 2; k++) {
//              f1 = frequencies[vertices[0][j][k] * num_band + b1];
//              f2 = frequencies[vertices[1][j][k] * num_band + b2];
//              freq_vertices[0][j][k] = f1 + f2;
//              freq_vertices[1][j][k] = f1 - f2;
//              freq_vertices[2][j][k] = -f1 + f2;
//            }
//          }
//          for (j = 0; j < nb[index][0]; j++) {
//            b0 = (nb[index][0] == num_band0)?  band_indices[j]: j;
//            f0 = frequencies[triplets_permute[i][0] * num_band + b0];
//	        g0 = thm_get_integration_weight_1D(f0, freq_vertices[0], 'I');
//	        g1 = thm_get_integration_weight_1D(f0, freq_vertices[1], 'I');
//	        g2 = thm_get_integration_weight_1D(f0, freq_vertices[2], 'I');
//            if (index==0)
//              adrs_shift = i * num_band0 * num_band * num_band +
//                  j * num_band * num_band + m * num_band + n; // j is the branch of first phonon
//            else if (index==1)
//              adrs_shift = i * num_band0 * num_band * num_band +
//                  n * num_band * num_band + j * num_band + m; // n is the branch of first phonon
//            else if (index==2)
//              adrs_shift = i * num_band0 * num_band * num_band +
//                  m * num_band * num_band + n * num_band + j; // m is the branch of first phonon
////            iw[index *tbbb+adrs_shift] += g0;
////            iw[(1+index)%3 *tbbb+adrs_shift] += g1;
////            iw[(2+index)%3 *tbbb+adrs_shift] += g2;
//            iw_tmp[index * 3 * tbbb + index *tbbb+adrs_shift] = g0;
//            iw_tmp[index * 3 * tbbb + (1+index)%3 *tbbb+adrs_shift] = g1;
//            iw_tmp[index * 3 * tbbb + (2+index)%3 *tbbb+adrs_shift] = g2;
//          }
//        }
//      }
//    }
//  }
////  if (is_sym)
////    for (i = 0; i < 3 * tbbb ; i++)
////      iw[i] /= 3;
//  if (is_sym)
//  {
//    for (i = 0; i < 3 * tbbb ; i++)
//    {
//      iw[i] = (iw_tmp[i] + iw_tmp[3 * tbbb + i] + iw_tmp[2 * 3 * tbbb + i] -
//       fmax(fmax(iw_tmp[i], iw_tmp[3 * tbbb + i]), iw_tmp[2 * 3 * tbbb + i])-
//       fmin(fmin(iw_tmp[i], iw_tmp[3 * tbbb + i]), iw_tmp[2 * 3 * tbbb + i])
//       );
//    }
//  }
//  else
//    for (i = 0; i < 3 * tbbb ; i++)
//      iw[i] = iw_tmp[i];
//  free(iw_tmp);
//  Py_RETURN_NONE;
//}


static void set_triplet_integration_1D_at_triplet(double *iw_triplet, //shape: num_band0, num_band1, num_band2
                                                  const int triplet[3],
                                                  const int vertices[2][2][2], //neighbors
                                                  const double *frequencies,
                                                  const int *band_indices[3],
                                                  const int num_bands[3],
                                                  const int num_band)
{
  int i, j, k, l, b0, b1, b2, address_shift;
  int bbb = num_bands[0] * num_bands[1] * num_bands[2];
  double freq_vertices[3][2][2], f0, f1, f2, g0, g1, g2;

  for (i = 0; i < 3 * bbb; i++) iw_triplet[i] = 0; //initialization
  for (j = 0; j < num_bands[1]; j++) {
    b1 = band_indices[1][j];
    for (k = 0; k < num_bands[2]; k++) {
      b2 = band_indices[2][k];
      for (i = 0; i < 2; i++) {
        for (l = 0; l < 2; l++) {
          f1 = frequencies[vertices[0][i][l] * num_band + b1];
          f2 = frequencies[vertices[1][i][l] * num_band + b2];
          freq_vertices[0][i][l] = f1 + f2;
          freq_vertices[1][i][l] = f1 - f2;
          freq_vertices[2][i][l] = -f1 + f2;
        }
      }
      for (i = 0; i < num_bands[0]; i++) {
        b0 = band_indices[0][i];
        f0 = frequencies[triplet[0] * num_band + b0];
        g0 = thm_get_integration_weight_1D(f0, freq_vertices[0], 'I');
        g1 = thm_get_integration_weight_1D(f0, freq_vertices[1], 'I');
        g2 = thm_get_integration_weight_1D(f0, freq_vertices[2], 'I');
        address_shift = i * num_bands[1] * num_bands[2] + j * num_bands[2] + k;
        iw_triplet[address_shift] = g0;
        iw_triplet[bbb + address_shift] = g1;
        iw_triplet[2 * bbb + address_shift] = g2;
      }
    }
  }
}



static PyObject *
py_set_triplets_integration_weights_1D(PyObject *self, PyObject *args)
{
  PyArrayObject* iw_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* triplets_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* band_indices_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* bz_map_py;
  int is_sym;
  if (!PyArg_ParseTuple(args, "OOOOOOOOi",
			&iw_py,
			&relative_grid_address_py,
			&mesh_py,
			&triplets_py,
			&frequencies_py,
			&band_indices_py,
			&grid_address_py,
			&bz_map_py,
			&is_sym)) {
    return NULL;
  }

  double *iw = (double*)iw_py->data;
  const int num_band0 = band_indices_py->dimensions[0];
  const int* band_indices = band_indices_py->data;
  SPGCONST int (*relative_grid_address)[2][3] =
    (int(*)[2][3])relative_grid_address_py->data;
  const int *mesh = (int*)mesh_py->data;
  SPGCONST int (*triplets)[3] = (int(*)[3])triplets_py->data;
  const int num_triplets = (int)triplets_py->dimensions[0];
  SPGCONST int (*grid_address)[3] = (int(*)[3])grid_address_py->data;
  const double *frequencies = (double*)frequencies_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  const int *bz_map = (int*)bz_map_py->data;

  int t, i, j, k, index, sign, tbbb=num_triplets*num_band0*num_band*num_band, bbb = num_band0*num_band*num_band;
  int ijk, jki, kij;
  int tp_relative_grid_address[2][2][2][3];
  int vertices[2][2][2];
  double v[3];
  int triplet[3];
  int *band_indices_all = (int*)malloc(sizeof(int) * num_band);
  double *iw_triplet;
  int *band_indices_tmp[3], num_bands[3], axis_b0;

  for (t = 0; t < 2; t++) {
    sign = 1 - t * 2;
    for (i = 0; i < 2; i++) {
      for (j = 0; j < 2; j++) {
	for (k = 0; k < 3; k++) {
	  tp_relative_grid_address[t][i][j][k] =
	    relative_grid_address[i][j][k] * sign;
	}
      }
    }
  }
  for (i = 0; i < num_band; i++) band_indices_all[i] = i;

  #pragma omp parallel private(t, index, i, j, k, axis_b0, ijk, jki, kij, band_indices_tmp, iw_triplet, num_bands, triplet, vertices, v)
  {
    iw_triplet = (double*)malloc(sizeof(double) * 3 * 3 * num_band0 * num_band * num_band);
    #pragma omp for
    for (t = 0; t < num_triplets; t++)
    {
      for (index=0; index < (is_sym? 3: 1); index++)
      { // 012->120->201
        axis_b0 = ((index==0)? 0: ((index==1)? 2: 1));
        for (i = 0; i < 3; i++)
        {
          if (i == axis_b0)
          {
            band_indices_tmp[i] = band_indices;
            num_bands[i] = num_band0;
          }
          else
          {
            band_indices_tmp[i] = band_indices_all;
            num_bands[i] = num_band;
          }
        }
        for (i = 0; i < 3; i++)
          triplet[i] = triplets[t][(i+index) % 3];
        get_triplet_tetrahedra_vertices_1D(vertices,
                                           tp_relative_grid_address,
                                           mesh,
                                           triplet,
                                           grid_address,
                                           bz_map);

        set_triplet_integration_1D_at_triplet(iw_triplet + index * 3 * bbb, //shape: num_band0, num_band1, num_band2
                                            triplet,
                                            vertices, //neighbors
                                            frequencies,
                                            band_indices_tmp,
                                            num_bands,
                                            num_band);
      }
      if (is_sym)
      {
        for (i = 0; i < num_band0; i++)
          for (j = 0; j < num_band; j++)
            for (k = 0; k < num_band; k++)
            {
              ijk = i * num_band * num_band + j * num_band + k;
              jki = j * num_band * num_band0 + k * num_band0 + i;
              kij = k * num_band0 * num_band + i * num_band + j;
              v[0] = iw_triplet[0 * 3 * bbb + 0 * bbb + ijk];
              v[1] = iw_triplet[1 * 3 * bbb + 2 * bbb + jki];
              v[2] = iw_triplet[2 * 3 * bbb + 1 * bbb + kij];
              iw[0 * tbbb + t * bbb + ijk] =
                v[0] + v[1] + v[2] - fmax(fmax(v[0], v[1]), v[2]) - fmin(fmin(v[0], v[1]), v[2]);


              v[0] = iw_triplet[0 * 3 * bbb + 1 * bbb + ijk];
              v[1] = iw_triplet[1 * 3 * bbb + 0 * bbb + jki];
              v[2] = iw_triplet[2 * 3 * bbb + 2 * bbb + kij];
              iw[1 * tbbb + t * bbb + ijk] =
                v[0] + v[1] + v[2] - fmax(fmax(v[0], v[1]), v[2]) - fmin(fmin(v[0], v[1]), v[2]);
              v[0] = iw_triplet[0 * 3 * bbb + 2 * bbb + ijk];
              v[1] = iw_triplet[1 * 3 * bbb + 1 * bbb + jki];
              v[2] = iw_triplet[2 * 3 * bbb + 0 * bbb + kij];
              iw[2 * tbbb + t * bbb + ijk] =
                v[0] + v[1] + v[2] - fmax(fmax(v[0], v[1]), v[2]) - fmin(fmin(v[0], v[1]), v[2]);
            }
      }
      else
      {
        for (i = 0; i < 3; i++)
          for (j = 0; j < bbb; j++)
            iw[i * tbbb + t * bbb + j] = iw_triplet[i * bbb + j];
      }
    }
    free(iw_triplet);
  }
  free(band_indices_all);
  Py_RETURN_NONE;
}

static PyObject *
py_set_triplets_integration_weights(PyObject *self, PyObject *args)
{
  PyArrayObject* iw_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* triplets_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* band_indices_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* bz_map_py;
  int is_sym;
  if (!PyArg_ParseTuple(args, "OOOOOOOOi",
			&iw_py,
			&relative_grid_address_py,
			&mesh_py,
			&triplets_py,
			&frequencies_py,
			&band_indices_py,
			&grid_address_py,
			&bz_map_py,
			&is_sym)) {
    return NULL;
  }

  double *iw = (double*)iw_py->data;
  SPGCONST int (*relative_grid_address)[4][3] =
    (int(*)[4][3])relative_grid_address_py->data;
  const int *mesh = (int*)mesh_py->data;
  SPGCONST int (*triplets)[3] = (int(*)[3])triplets_py->data;
  const int num_triplets = (int)triplets_py->dimensions[0];
  SPGCONST int (*grid_address)[3] = (int(*)[3])grid_address_py->data;
  const double *frequencies = (double*)frequencies_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  const int num_band0 = (int)iw_py->dimensions[2];
  const int* band_indices = (int*) band_indices_py->data;
  const int *bz_map = (int*)bz_map_py->data;
  int triplets_permute[num_triplets][3];
  int i, j, k, l, b0, b1, b2, m, n, sign, index, tbbb=num_triplets*num_band0*num_band*num_band;
  int tp_relative_grid_address[2][24][4][3];
  int vertices[2][24][4];
  int nb[3][3] = {{num_band0, num_band, num_band},
                  {num_band, num_band, num_band0},
                  {num_band, num_band0, num_band}};
  int adrs_shift=0;
  double f0, f1, f2, g0, g1, g2;
  double freq_vertices[3][24][4];

  for (i = 0; i < 2; i++) {
    sign = 1 - i * 2;
    for (j = 0; j < 24; j++) {
      for (k = 0; k < 4; k++) {
        for (l = 0; l < 3; l++) {
          tp_relative_grid_address[i][j][k][l] =
            relative_grid_address[j][k][l] * sign;
        }
      }
    }
  }
  for (index=0; index < (is_sym? 3: 1); index++)
  { // 012->120->201
    for (i=0; i< num_triplets; i++)
      for (j = 0; j < 3; j++)
        triplets_permute[i][j] = triplets[i][(j+index) % 3];
    #pragma omp parallel for private(j, k, l, m, n, b0, b1, b2, vertices, adrs_shift, f0, f1, f2, g0, g1, g2, freq_vertices)
    for (i = 0; i < num_triplets; i++) {
      get_triplet_tetrahedra_vertices(vertices,
                      tp_relative_grid_address,
                      mesh,
                      triplets_permute[i],
                      grid_address,
                      bz_map);
      for (m = 0; m < nb[index][1]; m++) {
        b1 = (nb[index][1] == num_band0)?  band_indices[m]: m;
        for (n = 0; n < nb[index][2]; n++) {
          b2 = (nb[index][2] == num_band0)?  band_indices[n]: n;
          for (j = 0; j < 24; j++) {
            for (k = 0; k < 4; k++) {
              f1 = frequencies[vertices[0][j][k] * num_band + b1];
              f2 = frequencies[vertices[1][j][k] * num_band + b2];
              freq_vertices[0][j][k] = f1 + f2;
              freq_vertices[1][j][k] = f1 - f2;
              freq_vertices[2][j][k] = -f1 + f2;
            }
          }
          for (j = 0; j < nb[index][0]; j++) {
            b0 = (nb[index][0] == num_band0)?  band_indices[j]: j;
            f0 = frequencies[triplets_permute[i][0] * num_band + b0];
            g0 = thm_get_integration_weight(f0, freq_vertices[0], 'I');
            g1 = thm_get_integration_weight(f0, freq_vertices[1], 'I');
            g2 = thm_get_integration_weight(f0, freq_vertices[2], 'I');
            if (index==0)
              adrs_shift = i * num_band0 * num_band * num_band +
                  j * num_band * num_band + m * num_band + n;
            else if (index==1)
              adrs_shift = i * num_band0 * num_band * num_band +
                  n * num_band * num_band + j * num_band + m;
            else if (index==2)
              adrs_shift = i * num_band0 * num_band * num_band +
                  m * num_band * num_band + n * num_band + j;
            iw[index *tbbb+adrs_shift] += g0;
            iw[(1+index)%3 *tbbb+adrs_shift] += g1;
            iw[(2+index)%3 *tbbb+adrs_shift] += g2;
          }
        }
      }
    }
  }
  if (is_sym)
    for (i = 0; i < 3 * tbbb ; i++)
      iw[i] /= 3;
  Py_RETURN_NONE;
}

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

static void get_triplet_tetrahedra_vertices
  (int vertices[2][24][4],
   SPGCONST int relative_grid_address[2][24][4][3],
   const int mesh[3],
   const int triplet[3],
   SPGCONST int grid_address[][3],
   const int bz_map[])
{
  int i, j;

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 24; j++) {
      get_neighboring_grid_points(vertices[i][j],
				      triplet[i + 1],
				      relative_grid_address[i][j],
				      4,
				      mesh,
				      grid_address,
				      bz_map);
    }
  }
}

static void get_triplet_tetrahedra_vertices_1D
  (int vertices[2][2][2],
   SPGCONST int relative_grid_address[2][2][2][3],
   const int mesh[3],
   const int triplet[3],
   SPGCONST int grid_address[][3],
   const int bz_map[])
{
  int i, j;

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      get_neighboring_grid_points(vertices[i][j],
				      triplet[i + 1],
				      relative_grid_address[i][j],
				      2,
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
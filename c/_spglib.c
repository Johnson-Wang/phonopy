#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include <spglib.h>


static PyObject * get_dataset(PyObject *self, PyObject *args);
static PyObject * get_spacegroup(PyObject *self, PyObject *args);
static PyObject * get_pointgroup(PyObject *self, PyObject *args);
static PyObject * refine_cell(PyObject *self, PyObject *args);
static PyObject * get_symmetry(PyObject *self, PyObject *args);
static PyObject * get_symmetry_with_collinear_spin(PyObject *self, PyObject *args);
static PyObject * find_primitive(PyObject *self, PyObject *args);
static PyObject * get_ir_kpoints(PyObject *self, PyObject *args);
static PyObject * get_ir_reciprocal_mesh(PyObject *self, PyObject *args);
static PyObject * get_stabilized_reciprocal_mesh(PyObject *self, PyObject *args);
static PyObject * get_triplets_reciprocal_mesh_at_q(PyObject *self, PyObject *args);
static PyObject * get_grid_triplets_at_q(PyObject *self, PyObject *args);
static PyObject * get_kpointgroup(PyObject *self, PyObject *args);
static PyObject * get_unique_tetrahedra(PyObject *self, PyObject *args);
static PyObject * get_reduced_triplets_permute_sym(PyObject *self, PyObject *args);
static PyObject * get_reduced_pairs_permute_sym(PyObject *self, PyObject *args);
static PyObject * get_BZ_grid_points_by_rotations(PyObject *self, PyObject *args);
static PyObject * relocate_BZ_grid_address(PyObject *self, PyObject *args);
static PyObject * get_BZ_triplets_at_q(PyObject *self, PyObject *args);
static PyObject * get_neighboring_grid_points(PyObject *self, PyObject *args);
static PyObject *
get_tetrahedra_relative_grid_address(PyObject *self, PyObject *args);
static PyObject *
get_all_tetrahedra_relative_grid_address(PyObject *self, PyObject *args);
static PyObject *
get_tetrahedra_integration_weight(PyObject *self, PyObject *args);
static PyObject *
get_tetrahedra_integration_weight_at_omegas(PyObject *self, PyObject *args);

static PyMethodDef functions[] = {
  {"dataset", get_dataset, METH_VARARGS,
   "Dataset for crystal symmetry"},
  {"spacegroup", get_spacegroup, METH_VARARGS,
   "International symbol"},
  {"pointgroup", get_pointgroup, METH_VARARGS,
   "International symbol of pointgroup"},
  {"kpointgroup", get_kpointgroup, METH_VARARGS,
   "Get the kpoint group from the real-space point group"},
  {"refine_cell", refine_cell, METH_VARARGS,
   "Refine cell"},
  {"symmetry", get_symmetry, METH_VARARGS,
   "Symmetry operations"},
  {"symmetry_with_collinear_spin", get_symmetry_with_collinear_spin,
   METH_VARARGS, "Symmetry operations with collinear spin magnetic moments"},
  {"primitive", find_primitive, METH_VARARGS,
   "Find primitive cell in the input cell"},
  {"ir_kpoints", get_ir_kpoints, METH_VARARGS,
   "Irreducible k-points"},
  {"ir_reciprocal_mesh", get_ir_reciprocal_mesh, METH_VARARGS,
   "Reciprocal mesh points with map"},
  {"stabilized_reciprocal_mesh", get_stabilized_reciprocal_mesh, METH_VARARGS,
   "Reciprocal mesh points with map"},
  {"BZ_grid_address", relocate_BZ_grid_address, METH_VARARGS,
   "Relocate grid addresses inside Brillouin zone"},
  {"unique_tetrahedra", get_unique_tetrahedra, METH_VARARGS,
   "Get the unique tetrahedrons inside a Brillouin zone"},
  {"triplets_reciprocal_mesh_at_q", get_triplets_reciprocal_mesh_at_q,
   METH_VARARGS, "Triplets on reciprocal mesh points at a specific q-point"},
  {"reduce_triplets_permute_sym", get_reduced_triplets_permute_sym, 
   METH_VARARGS, "Reduce the total number of triplets by considering permutation symmetry"},
   {"reduce_pairs_permute_sym", get_reduced_pairs_permute_sym, 
   METH_VARARGS, "Reduce the total number of pairs by considering permutation symmetry"},
  {"grid_triplets_at_q", get_grid_triplets_at_q,
   METH_VARARGS, "Grid point triplets on reciprocal mesh points at a specific q-point are set from output variables of triplets_reciprocal_mesh_at_q"},
  {"BZ_grid_points_by_rotations", get_BZ_grid_points_by_rotations, METH_VARARGS,
   "Rotated grid points in BZ are returned"},
  {"BZ_triplets_at_q", get_BZ_triplets_at_q,
   METH_VARARGS, "Triplets in reciprocal primitive lattice are transformed to those in BZ."},
  {"neighboring_grid_points", get_neighboring_grid_points,
   METH_VARARGS, "Neighboring grid points by relative grid addresses"},
  {"tetrahedra_relative_grid_address", get_tetrahedra_relative_grid_address,
   METH_VARARGS, "Relative grid addresses of vertices of 24 tetrahedra"},
  {"all_tetrahedra_relative_grid_address",
   get_all_tetrahedra_relative_grid_address,
   METH_VARARGS,
   "4 (all) sets of relative grid addresses of vertices of 24 tetrahedra"},
  {"tetrahedra_integration_weight", get_tetrahedra_integration_weight,
   METH_VARARGS, "Integration weight for tetrahedron method"},
  {"tetrahedra_integration_weight_at_omegas",
   get_tetrahedra_integration_weight_at_omegas,
   METH_VARARGS, "Integration weight for tetrahedron method at omegas"},
   {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_spglib(void)
{
  Py_InitModule3("_spglib", functions, "C-extension for spglib\n\n...\n");
  return;
}

static PyObject * get_dataset(PyObject *self, PyObject *args)
{
  int i, j, k;
  double symprec, angle_tolerance;
  SpglibDataset *dataset;
  PyArrayObject* lattice;
  PyArrayObject* position;
  PyArrayObject* atom_type;
  PyObject* array, *vec, *mat, *rot, *trans, *wyckoffs, *equiv_atoms;
  
  if (!PyArg_ParseTuple(args, "OOOdd",
			&lattice,
			&position,
			&atom_type,
			&symprec,
			&angle_tolerance)) {
    return NULL;
  }

  SPGCONST double (*lat)[3] = (double(*)[3])lattice->data;
  SPGCONST double (*pos)[3] = (double(*)[3])position->data;
  const int num_atom = position->dimensions[0];
  const int* typat = (int*)atom_type->data;

  dataset = spgat_get_dataset(lat,
			      pos,
			      typat,
			      num_atom,
			      symprec,
			      angle_tolerance);

  array = PyList_New(9);

  /* Space group number, international symbol, hall symbol */
  PyList_SetItem(array, 0, PyInt_FromLong((long) dataset->spacegroup_number));
  PyList_SetItem(array, 1, PyString_FromString(dataset->international_symbol));
  PyList_SetItem(array, 2, PyString_FromString(dataset->hall_symbol));

  /* Transformation matrix */
  mat = PyList_New(3);
  for (i = 0; i < 3; i++) {
    vec = PyList_New(3);
    for (j = 0; j < 3; j++) {
      PyList_SetItem(vec, j, PyFloat_FromDouble(dataset->transformation_matrix[i][j]));
    }
    PyList_SetItem(mat, i, vec);
  }
  PyList_SetItem(array, 3, mat);

  /* Origin shift */
  vec = PyList_New(3);
  for (i = 0; i < 3; i++) {
    PyList_SetItem(vec, i, PyFloat_FromDouble(dataset->origin_shift[i]));
  }
  PyList_SetItem(array, 4, vec);

  /* Rotation matrices */
  rot = PyList_New(dataset->n_operations);
  for (i = 0; i < dataset->n_operations; i++) {
    mat = PyList_New(3);
    for (j = 0; j < 3; j++) {
      vec = PyList_New(3);
      for (k = 0; k < 3; k++) {
	PyList_SetItem(vec, k, PyInt_FromLong((long) dataset->rotations[i][j][k]));
      }
      PyList_SetItem(mat, j, vec);
    }
    PyList_SetItem(rot, i, mat);
  }
  PyList_SetItem(array, 5, rot);

  /* Translation vectors */
  trans = PyList_New(dataset->n_operations);
  for (i = 0; i < dataset->n_operations; i++) {
    vec = PyList_New(3);
    for (j = 0; j < 3; j++) {
      PyList_SetItem(vec, j, PyFloat_FromDouble(dataset->translations[i][j]));
    }
    PyList_SetItem(trans, i, vec);
  }
  PyList_SetItem(array, 6, trans);

  /* Wyckoff letters, Equivalent atoms */
  wyckoffs = PyList_New(dataset->n_atoms);
  equiv_atoms = PyList_New(dataset->n_atoms);
  for (i = 0; i < dataset->n_atoms; i++) {
    PyList_SetItem(wyckoffs, i, PyInt_FromLong((long) dataset->wyckoffs[i]));
    PyList_SetItem(equiv_atoms, i, PyInt_FromLong((long) dataset->equivalent_atoms[i]));
  }
  PyList_SetItem(array, 7, wyckoffs);
  PyList_SetItem(array, 8, equiv_atoms);
  spg_free_dataset(dataset);

  return array;
}

static PyObject * get_spacegroup(PyObject *self, PyObject *args)
{
  double symprec, angle_tolerance;
  char symbol[26];
  PyArrayObject* lattice;
  PyArrayObject* position;
  PyArrayObject* atom_type;
  if (!PyArg_ParseTuple(args, "OOOdd",
			&lattice,
			&position,
			&atom_type,
			&symprec,
			&angle_tolerance)) {
    return NULL;
  }

  SPGCONST double (*lat)[3] = (double(*)[3])lattice->data;
  SPGCONST double (*pos)[3] = (double(*)[3])position->data;
  const int num_atom = position->dimensions[0];
  const int* typat = (int*)atom_type->data;

  const int num_spg = spgat_get_international(symbol,
					      lat,
					      pos,
					      typat,
					      num_atom,
					      symprec,
					      angle_tolerance);
  sprintf(symbol, "%s (%d)", symbol, num_spg);

  return PyString_FromString(symbol);
}

static PyObject * get_pointgroup(PyObject *self, PyObject *args)
{
  PyArrayObject* rotations;
  if (! PyArg_ParseTuple(args, "O", &rotations)) {
    return NULL;
  }

  int *rot_int = (int*)rotations->data;

  int i, j, k;
  int trans_mat[3][3];
  char symbol[6];
  PyObject* array, * mat, * vec;
    
  const int num_rot = rotations->dimensions[0];
  int rot[num_rot][3][3];
  for (i = 0; i < num_rot; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	rot[i][j][k] = (int) rot_int[i*9 + j*3 + k];
      }
    }
  }
  

  const int ptg_num = spg_get_pointgroup(symbol, trans_mat, rot, num_rot);

  /* Transformation matrix */
  mat = PyList_New(3);
  for (i = 0; i < 3; i++) {
    vec = PyList_New(3);
    for (j = 0; j < 3; j++) {
      PyList_SetItem(vec, j, PyInt_FromLong((long)trans_mat[i][j]));
    }
    PyList_SetItem(mat, i, vec);
  }

  array = PyList_New(3);
  PyList_SetItem(array, 0, PyString_FromString(symbol));
  PyList_SetItem(array, 1, PyInt_FromLong((long) ptg_num));
  PyList_SetItem(array, 2, mat);

  return array;
}

static PyObject * get_kpointgroup(PyObject *self, PyObject *args)
{
  PyArrayObject* rotations;
  PyArrayObject* reci_rotations_py;
  PyArrayObject* mesh_py;
  PyArrayObject* kpoints_py;
  int is_time_reversal;
  if (! PyArg_ParseTuple(args, "OOOOi",
                         &reci_rotations_py,
                         &rotations,
			 &mesh_py,
			 &kpoints_py,
			 &is_time_reversal)) {
    return NULL;
  }
  const int num_kpoint = kpoints_py->dimensions[0];
  const int num_rot = rotations->dimensions[0];
  const int *rot_int = (int*)rotations->data;
  const int *mesh = (int*)mesh_py->data;
  double *kpoints = (double*) kpoints_py->data;
  int *reci_rotations = (int*)reci_rotations_py->data;
  int i, j, k, num_reci_rot;
  int rot[num_rot][3][3], reci_rot[num_rot][3][3];
  double kpts[num_kpoint][3];
  for (i = 0; i < num_kpoint; i++){
    for (j = 0; j < 3; j++)
      kpts[i][j] = kpoints[i * 3 + j];
  }
  for (i = 0; i < num_rot; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	rot[i][j][k] = (int) rot_int[i*9 + j*3 + k];
      }
    }
  }
  
  num_reci_rot = spg_get_kpoint_group_at_q(reci_rot,
					   kpts,
					   mesh,
					   is_time_reversal,
					   num_rot,
					   num_kpoint,
					   rot);
  
  for (i = 0; i < num_reci_rot; i++)
    for (j = 0; j < 3; j++)
      for (k = 0; k < 3; k++)
	reci_rotations[i * 9 + j * 3 + k] = reci_rot[i][j][k];
  return PyInt_FromLong((long) num_reci_rot);
}

static PyObject * refine_cell(PyObject *self, PyObject *args)
{
  int num_atom;
  double symprec, angle_tolerance;
  PyArrayObject* lattice;
  PyArrayObject* position;
  PyArrayObject* atom_type;
  if (!PyArg_ParseTuple(args, "OOOidd",
			&lattice,
			&position,
			&atom_type,
			&num_atom,
			&symprec,
			&angle_tolerance)) {
    return NULL;
  }

  double (*lat)[3] = (double(*)[3])lattice->data;
  SPGCONST double (*pos)[3] = (double(*)[3])position->data;
  int* typat = (int*)atom_type->data;

  int num_atom_brv = spgat_refine_cell(lat,
				       pos,
				       typat,
				       num_atom,
				       symprec,
				       angle_tolerance);

  return PyInt_FromLong((long) num_atom_brv);
}


static PyObject * find_primitive(PyObject *self, PyObject *args)
{
  double symprec, angle_tolerance;
  PyArrayObject* lattice;
  PyArrayObject* position;
  PyArrayObject* atom_type;
  if (!PyArg_ParseTuple(args, "OOOdd",
			&lattice,
			&position,
			&atom_type,
			&symprec,
			&angle_tolerance)) {
    return NULL;
  }

  double (*lat)[3] = (double(*)[3])lattice->data;
  double (*pos)[3] = (double(*)[3])position->data;
  int num_atom = position->dimensions[0];
  int* types = (int*)atom_type->data;

  int num_atom_prim = spgat_find_primitive(lat,
					   pos,
					   types,
					   num_atom,
					   symprec,
					   angle_tolerance);

  return PyInt_FromLong((long) num_atom_prim);
}

static PyObject * get_symmetry(PyObject *self, PyObject *args)
{
  int i, j, k;
  double symprec, angle_tolerance;
  PyArrayObject* lattice;
  PyArrayObject* position;
  PyArrayObject* rotation;
  PyArrayObject* translation;
  PyArrayObject* atom_type;
  if (!PyArg_ParseTuple(args, "OOOOOdd",
			&rotation,
			&translation,
			&lattice,
			&position,
			&atom_type,
			&symprec,
			&angle_tolerance)) {
    return NULL;
  }

  SPGCONST double (*lat)[3] = (double(*)[3])lattice->data;
  SPGCONST double (*pos)[3] = (double(*)[3])position->data;
  const int* types = (int*)atom_type->data;
  const int num_atom = position->dimensions[0];
  int *rot_int = (int*)rotation->data;
  double (*trans)[3] = (double(*)[3])translation->data;
  const int num_sym_from_array_size = rotation->dimensions[0];

  int rot[num_sym_from_array_size][3][3];
  
  /* num_sym has to be larger than num_sym_from_array_size. */
  const int num_sym = spgat_get_symmetry(rot,
					 trans,
					 num_sym_from_array_size,
					 lat,
					 pos,
					 types,
					 num_atom,
					 symprec,
					 angle_tolerance);
  for (i = 0; i < num_sym; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	rot_int[i*9 + j*3 + k] = (int)rot[i][j][k];
      }
    }
  }

  return PyInt_FromLong((long) num_sym);
}

static PyObject * get_symmetry_with_collinear_spin(PyObject *self,
						   PyObject *args)
{
  int i, j, k;
  double symprec, angle_tolerance;
  PyArrayObject* lattice;
  PyArrayObject* position;
  PyArrayObject* rotation;
  PyArrayObject* translation;
  PyArrayObject* atom_type;
  PyArrayObject* magmom;
  if (!PyArg_ParseTuple(args, "OOOOOOdd",
			&rotation,
			&translation,
			&lattice,
			&position,
			&atom_type,
			&magmom,
			&symprec,
			&angle_tolerance)) {
    return NULL;
  }

  SPGCONST double (*lat)[3] = (double(*)[3])lattice->data;
  SPGCONST double (*pos)[3] = (double(*)[3])position->data;
  const double *spins = (double*) magmom->data;
  const int* types = (int*)atom_type->data;
  const int num_atom = position->dimensions[0];
  int *rot_int = (int*)rotation->data;
  double (*trans)[3] = (double(*)[3])translation->data;
  const int num_sym_from_array_size = rotation->dimensions[0];

  int rot[num_sym_from_array_size][3][3];
  
  /* num_sym has to be larger than num_sym_from_array_size. */
  const int num_sym = 
    spgat_get_symmetry_with_collinear_spin(rot,
					   trans,
					   num_sym_from_array_size,
					   lat,
					   pos,
					   types,
					   spins,
					   num_atom,
					   symprec,
					   angle_tolerance);
  for (i = 0; i < num_sym; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	rot_int[i*9 + j*3 + k] = (int)rot[i][j][k];
      }
    }
  }

  return PyInt_FromLong((long) num_sym);
}

static PyObject * get_ir_kpoints(PyObject *self, PyObject *args)
{
  double symprec;
  int is_time_reversal;
  PyArrayObject* kpoint;
  PyArrayObject* kpoint_map;
  PyArrayObject* lattice;
  PyArrayObject* position;
  PyArrayObject* atom_type;
  if (!PyArg_ParseTuple(args, "OOOOOid", &kpoint_map, &kpoint, &lattice, &position,
			&atom_type, &is_time_reversal, &symprec))
    return NULL;

  SPGCONST double (*lat)[3] = (double(*)[3])lattice->data;
  SPGCONST double (*pos)[3] = (double(*)[3])position->data;
  SPGCONST double (*kpts)[3] = (double(*)[3])kpoint->data;
  const int num_kpoint = kpoint->dimensions[0];
  const int* types = (int*)atom_type->data;
  const int num_atom = position->dimensions[0];
  int *map = (int*)kpoint_map->data;

  /* num_sym has to be larger than num_sym_from_array_size. */
  const int num_ir_kpt = spg_get_ir_kpoints(map,
					    kpts,
					    num_kpoint,
					    lat,
					    pos,
					    types,
					    num_atom,
					    is_time_reversal,
					    symprec);

  return PyInt_FromLong((long) num_ir_kpt);
}

static PyObject * get_ir_reciprocal_mesh(PyObject *self, PyObject *args)
{
  int i, j;
  double symprec;
  PyArrayObject* grid_point;
  PyArrayObject* map;
  PyArrayObject* mesh;
  PyArrayObject* is_shift;
  int is_time_reversal;
  PyArrayObject* lattice;
  PyArrayObject* position;
  PyArrayObject* atom_type;
  if (!PyArg_ParseTuple(args, "OOOOiOOOd",
			&grid_point,
			&map,
			&mesh,
			&is_shift,
			&is_time_reversal,
			&lattice,
			&position,
			&atom_type,
			&symprec))
    return NULL;

  SPGCONST double (*lat)[3] = (double(*)[3])lattice->data;
  SPGCONST double (*pos)[3] = (double(*)[3])position->data;
  const int num_grid = grid_point->dimensions[0];
  const int* types = (int*)atom_type->data;
  const int* mesh_int = (int*)mesh->data;
  const int* is_shift_int = (int*)is_shift->data;
  const int num_atom = position->dimensions[0];
  int *grid_pint = (int*)grid_point->data;
  int grid_int[num_grid][3];
  int*map_int = (int*)map->data;

  /* Check memory space */
  if (mesh_int[0] * mesh_int[1] * mesh_int[2] > num_grid) {
    return NULL;
  }

  /* num_sym has to be larger than num_sym_from_array_size. */
  const int num_ir = spg_get_ir_reciprocal_mesh(grid_int,
						map_int,
						mesh_int,
						is_shift_int,
						is_time_reversal,
						lat,
						pos,
						types,
						num_atom,
						symprec);
  
  for (i = 0; i < mesh_int[0] * mesh_int[1] * mesh_int[2]; i++) {
    for (j = 0; j < 3; j++) {
      grid_pint[i*3 + j] = (int) grid_int[i][j];
    }
  }
  
  return PyInt_FromLong((long) num_ir);
}

static PyObject * get_stabilized_reciprocal_mesh(PyObject *self, PyObject *args)
{
  int i, j, k;
  PyArrayObject* grid_point;
  PyArrayObject* map;
  PyArrayObject* rot_map_py;
  PyArrayObject* mesh;
  PyArrayObject* is_shift;
  int is_time_reversal;
  PyArrayObject* rotations;
  PyArrayObject* qpoints;
  if (!PyArg_ParseTuple(args, "OOOOOiOO",
			&grid_point,
			&map,
			&rot_map_py,
			&mesh,
			&is_shift,
			&is_time_reversal,
			&rotations,
			&qpoints)) {
    return NULL;
  }

  int *grid_pint = (int*)grid_point->data;
  const int num_grid = grid_point->dimensions[0];
  int grid_int[num_grid][3];
  int *map_int = (int*)map->data;
  int *rot_map = (int*) rot_map_py->data;
  const int* mesh_int = (int*)mesh->data;
  const int* is_shift_int = (int*)is_shift->data;
  const int* rot_int = (int*)rotations->data;
  const int num_rot = rotations->dimensions[0];
  int rot[num_rot][3][3];
  for (i = 0; i < num_rot; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	rot[i][j][k] = rot_int[i*9 + j*3 + k];
      }
    }
  }


  SPGCONST double (*q)[3] = (double(*)[3])qpoints->data;
  const int num_q = qpoints->dimensions[0];

  /* Check memory space */
  if (mesh_int[0] * mesh_int[1] * mesh_int[2] > num_grid) {
    return NULL;
  }

  const int num_ir = spg_get_stabilized_reciprocal_mesh(grid_int,
							map_int,
							rot_map,
							mesh_int,
							is_shift_int,
							is_time_reversal,
							num_rot,
							rot,
							num_q,
							q);

  for (i = 0; i < mesh_int[0] * mesh_int[1] * mesh_int[2]; i++) {
    for (j = 0; j < 3; j++) {
      grid_pint[i*3 + j] = grid_int[i][j];
    }
  }
  return PyInt_FromLong((long) num_ir);
}

static PyObject *get_unique_tetrahedra(PyObject *self, PyObject *args)
{
  PyArrayObject *bz_grid_address_py;
  PyArrayObject *relative_address_py;
  PyArrayObject *bz_map_py;
  PyArrayObject *mesh_py;
  PyArrayObject *unique_vertices_py;
  if (!PyArg_ParseTuple(args, "OOOOO",
			&unique_vertices_py,
			&bz_grid_address_py,
			&bz_map_py,
			&relative_address_py,
			&mesh_py)) {
    return NULL;
  }
  const int *bz_grid_address = (int*)bz_grid_address_py->data;
  const int num_grids = bz_grid_address_py->dimensions[0];
  const int *relative_address = (int*) relative_address_py->data;
  const int dim0 = relative_address_py->dimensions[0];
  const int dim1 = relative_address_py->dimensions[1];
  const int *mesh = (int*) mesh_py->data;
  const int *bz_map=(int*) bz_map_py->data;
  int *unique_vertices=(int*) unique_vertices_py->data;
  int num_unique=spg_get_unique_tetrahedra(unique_vertices,
					   bz_grid_address,
					   bz_map, 
					   relative_address,
					   mesh, 
					   num_grids,
					   dim0,
					   dim1);
  return PyInt_FromLong((long)num_unique);
          
}

static PyObject * relocate_BZ_grid_address(PyObject *self, PyObject *args)
{
  PyArrayObject* bz_grid_address_py;
  PyArrayObject* bz_map_py;
  PyArrayObject* bz_map_orig_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* is_shift_py;
  PyArrayObject* reciprocal_lattice_py;
  if (!PyArg_ParseTuple(args, "OOOOOOO",
			&bz_grid_address_py,
			&bz_map_py,
			&bz_map_orig_py,
			&grid_address_py,
			&mesh_py,
			&reciprocal_lattice_py,
			&is_shift_py)) {
    return NULL;
  }

  int (*bz_grid_address)[3] = (int(*)[3])bz_grid_address_py->data;
  int *bz_map = (int*)bz_map_py->data;
  int *bz_map_orig = (int*) bz_map_orig_py->data;
  SPGCONST int (*grid_address)[3] = (int(*)[3])grid_address_py->data;
  const int* mesh = (int*)mesh_py->data;
  const int* is_shift = (int*)is_shift_py->data;
  SPGCONST double (*reciprocal_lattice)[3]  =
    (double(*)[3])reciprocal_lattice_py->data;
  int num_ir_gp;
  
  num_ir_gp = spg_relocate_BZ_grid_address(bz_grid_address,
					   bz_map,
					   bz_map_orig,
					   grid_address,
					   mesh,
					   reciprocal_lattice,
					   is_shift);

  return PyInt_FromLong((long) num_ir_gp);
}

static PyObject * get_triplets_reciprocal_mesh_at_q(PyObject *self, PyObject *args)
{
  PyArrayObject* weights;
  PyArrayObject* grid_points;
  PyArrayObject* third_q;
  PyArrayObject* map_q_py;
  PyArrayObject* rot_map_q_py;
  int fixed_grid_number;
  PyArrayObject* mesh;
  int is_time_reversal;
  PyArrayObject* rotations;
  if (!PyArg_ParseTuple(args, "OOOOOiOiO",
			&weights,
			&grid_points,
			&third_q,
			&map_q_py,
			&rot_map_q_py,
			&fixed_grid_number,
			&mesh,
			&is_time_reversal,
			&rotations)) {
    return NULL;
  }

  int i, j, k;
  const int num_grid = grid_points->dimensions[0];
  int (*grid_points_int)[3] = (int(*)[3])grid_points->data;
  int *weights_int = (int*)weights->data;
  int *third_q_int = (int*)third_q->data;
  int *map_q = (int*)map_q_py->data;
  int *rot_map_q = (int*)rot_map_q_py->data;
  const int* mesh_int = (int*)mesh->data;
  const int* rot_int = (int*)rotations->data;
  const int num_rot = rotations->dimensions[0];
  SPGCONST int (*rot)[3][3] = (int(*)[3][3])rotations->data;
  const int num_ir = 
    spg_get_triplets_reciprocal_mesh_at_q(weights_int,
					  grid_points_int,
					  third_q_int,
					  map_q,
					  rot_map_q,
					  fixed_grid_number,
					  mesh_int,
					  is_time_reversal,
					  num_rot,
					  rot);
  return PyInt_FromLong((long) num_ir);
}

static PyObject * get_BZ_triplets_at_q(PyObject *self, PyObject *args)
{
  PyArrayObject* triplets_py;
  PyArrayObject* bz_grid_address_py;
  PyArrayObject* bz_map_py;
  PyArrayObject* map_triplets_py;
  PyArrayObject* mesh_py;
  int grid_point;
  if (!PyArg_ParseTuple(args, "OiOOOO",
			&triplets_py,
			&grid_point,
			&bz_grid_address_py,
			&bz_map_py,
			&map_triplets_py,
			&mesh_py)) {
    return NULL;
  }

  int (*triplets)[3] = (int(*)[3])triplets_py->data;
  SPGCONST int (*bz_grid_address)[3] = (int(*)[3])bz_grid_address_py->data;
  const int *bz_map = (int*)bz_map_py->data;
  const int *map_triplets = (int*)map_triplets_py->data;
  const int num_map_triplets = (int)map_triplets_py->dimensions[0];
  const int *mesh = (int*)mesh_py->data;
  int num_ir;

  num_ir = spg_get_BZ_triplets_at_q(triplets,
				    grid_point,
				    bz_grid_address,
				    bz_map,
				    map_triplets,
				    num_map_triplets,
				    mesh);

  return PyLong_FromLong((long) num_ir);
}


static PyObject * get_grid_triplets_at_q(PyObject *self, PyObject *args)
{
  PyArrayObject* triplets_py;
  PyArrayObject* grid_points_py;
  PyArrayObject* third_q_py;
  PyArrayObject* weights_py;
  PyArrayObject* mesh_py;
  int q_grid_point;
  if (!PyArg_ParseTuple(args, "OiOOOO",
			&triplets_py,
			&q_grid_point,
			&grid_points_py,
			&third_q_py,
			&weights_py,
			&mesh_py)) {
    return NULL;
  }

  int i, j;
  
  int *p_triplets = (int*)triplets_py->data;
  const int num_ir_triplets = (int)triplets_py->dimensions[0];
  const int *p_grid_points = (int*)grid_points_py->data;
  const int num_grid_points = (int)grid_points_py->dimensions[0];
  const int *third_q = (int*)third_q_py->data;
  const int *weights = (int*)weights_py->data;
  const int *mesh = (int*)mesh_py->data;
  int triplets[num_ir_triplets][3];
  int grid_points[num_grid_points][3];
  for (i = 0; i < num_grid_points; i++) {
    for (j = 0; j < 3; j++) {
      grid_points[i][j] = p_grid_points[i * 3 + j];
    }
  }
  
  spg_set_grid_triplets_at_q(triplets,
			     q_grid_point,
			     grid_points,
			     third_q,
			     weights,
			     mesh);
  
  for (i = 0; i < num_ir_triplets; i++) {
    for (j = 0; j < 3; j++) {
      p_triplets[i * 3 + j] = triplets[i][j];
    }
  }

  Py_RETURN_NONE;
}

static PyObject * get_reduced_triplets_permute_sym(PyObject *self, PyObject *args)
{
  PyArrayObject* triplet_mappings_py;
  PyArrayObject* sequence_py;
  PyArrayObject* triplets_py;
  PyArrayObject* triplet_numbers_py;
  PyArrayObject* grid_points_py;
  PyArrayObject* mesh_py;
  PyArrayObject* first_mapping_py;
  PyArrayObject* first_rotation_py;
  PyArrayObject* second_mapping_py;
  int (*triplets)[3], (*first_rotation)[3][3], num_irred_triplets;
  if (!PyArg_ParseTuple(args, "OOOOOOOOO",
                        &triplet_mappings_py,
			&sequence_py,
			&triplets_py,
			&grid_points_py,
			&triplet_numbers_py,
			&mesh_py,
			&first_mapping_py,
			&first_rotation_py,
			&second_mapping_py)) {
    return NULL;
  }
  int* triplet_mappings = (int*)triplet_mappings_py->data;
  const int num_grid = (int)grid_points_py->dimensions[0];
  char* sequence = (char(*)[3])sequence_py->data;
  const int* triplet_numbers = (int*) triplet_numbers_py->data;
  const int* mesh = (int*)mesh_py->data;
  const int* first_mapping = (int*) first_mapping_py->data;
  const int* second_mapping = (int*) second_mapping_py->data;
  const int num_grid_all = (int)second_mapping_py->dimensions[1];
  const int* grid_points = (int*)grid_points_py->data;
  triplets = (int(*)[3])triplets_py->data;
  first_rotation = (int(*)[3][3])first_rotation_py->data;
  
  num_irred_triplets = spg_reduce_triplets_permute_sym(triplet_mappings,
						       sequence,
						       triplets,
						       grid_points,
						       triplet_numbers,
						       mesh,
						       first_mapping,
						       first_rotation,     
						       second_mapping,
						       num_grid,
						       num_grid_all);
  PyInt_FromLong((long) num_irred_triplets);
}

static PyObject * get_reduced_pairs_permute_sym(PyObject *self, PyObject *args)
{
  PyArrayObject* pairs_mappings;
  PyArrayObject* sequence_py;
  PyArrayObject* pairs_py;
  PyArrayObject* pair_numbers_py;
  PyArrayObject* grid_points_py;
  PyArrayObject* mesh_py;
  PyArrayObject* first_mapping_py;
  PyArrayObject* first_rotation_py;
  PyArrayObject* second_mapping_py;
  int (*pairs)[2], (*first_rotation)[3][3], num_irred_pairs;
  if (!PyArg_ParseTuple(args, "OOOOOOOOO",
                        &pairs_mappings,
			&sequence_py,
			&pairs_py,
			&grid_points_py,
			&pair_numbers_py,
			&mesh_py,
			&first_mapping_py,
			&first_rotation_py,
			&second_mapping_py)) {
    return NULL;
  }
  int* pair_mappings = (int*)pairs_mappings->data;
  const int num_grid = (int)grid_points_py->dimensions[0];
  char* sequence = (char(*)[2])sequence_py->data;
  const int* pair_numbers = (int*) pair_numbers_py->data;
  const int* mesh = (int*)mesh_py->data;
  const int* first_mapping = (int*) first_mapping_py->data;
  const int* second_mapping = (int*) second_mapping_py->data;
  const int num_grid_all = (int)second_mapping_py->dimensions[1];
  const int* grid_points = (int*)grid_points_py->data;
  pairs = (int(*)[2])pairs_py->data;
  first_rotation = (int(*)[3][3])first_rotation_py->data;

  num_irred_pairs = spg_reduce_pairs_permute_sym(pair_mappings,
						       sequence,
						       pairs,
						       grid_points,
						       pair_numbers,
						       mesh,
						       first_mapping,
						       first_rotation,     
						       second_mapping,
						       num_grid,
						       num_grid_all);
  PyInt_FromLong((long) num_irred_pairs);
}

static PyObject * get_BZ_grid_points_by_rotations(PyObject *self, PyObject *args)
{
  PyArrayObject* rot_grid_points_py;
  PyArrayObject* address_orig_py;
  PyArrayObject* rot_reciprocal_py;
  PyArrayObject* mesh_py;
  PyArrayObject* is_shift_py;
  PyArrayObject* bz_map_py;
  if (!PyArg_ParseTuple(args, "OOOOOO",
			&rot_grid_points_py,
			&address_orig_py,
			&rot_reciprocal_py,
			&mesh_py,
			&is_shift_py,
			&bz_map_py)) {
    return NULL;
  }

  int *rot_grid_points = (int*)rot_grid_points_py->data;
  const int *address_orig = (int*)address_orig_py->data;
  SPGCONST int (*rot_reciprocal)[3][3] = (int(*)[3][3])rot_reciprocal_py->data;
  const int num_rot = rot_reciprocal_py->dimensions[0];
  const int* mesh = (int*)mesh_py->data;
  const int* is_shift = (int*)is_shift_py->data;
  const int* bz_map = (int*)bz_map_py->data;

  spg_get_BZ_grid_points_by_rotations(rot_grid_points,
				      address_orig,
				      num_rot,
				      rot_reciprocal,
				      mesh,
				      is_shift,
				      bz_map);
  Py_RETURN_NONE;
}

static PyObject *get_neighboring_grid_points(PyObject *self, PyObject *args)
{
  PyArrayObject* relative_grid_points_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* bz_grid_address_py;
  PyArrayObject* bz_map_py;
  int grid_point;
  if (!PyArg_ParseTuple(args, "OiOOOO",
			&relative_grid_points_py,
			&grid_point,
			&relative_grid_address_py,
			&mesh_py,
			&bz_grid_address_py,
			&bz_map_py)) {
    return NULL;
  }

  int* relative_grid_points = (int*)relative_grid_points_py->data;
  SPGCONST int (*relative_grid_address)[3] =
    (int(*)[3])relative_grid_address_py->data;
  const int num_relative_grid_address = relative_grid_address_py->dimensions[0];
  const int *mesh = (int*)mesh_py->data;
  SPGCONST int (*bz_grid_address)[3] = (int(*)[3])bz_grid_address_py->data;
  const int *bz_map = (int*)bz_map_py->data;

  spg_get_neighboring_grid_points(relative_grid_points,
				  grid_point,
				  relative_grid_address,
				  num_relative_grid_address,
				  mesh,
				  bz_grid_address,
				  bz_map);
  Py_RETURN_NONE;
}

static PyObject *
get_tetrahedra_relative_grid_address(PyObject *self, PyObject *args)
{
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* reciprocal_lattice_py;

  if (!PyArg_ParseTuple(args, "OO",
			&relative_grid_address_py,
			&reciprocal_lattice_py)) {
    return NULL;
  }

  int (*relative_grid_address)[4][3] =
    (int(*)[4][3])relative_grid_address_py->data;
  SPGCONST double (*reciprocal_lattice)[3] =
    (double(*)[3])reciprocal_lattice_py->data;

  spg_get_tetrahedra_relative_grid_address(relative_grid_address,
					   reciprocal_lattice);

  Py_RETURN_NONE;
}

static PyObject *
get_all_tetrahedra_relative_grid_address(PyObject *self, PyObject *args)
{
  PyArrayObject* relative_grid_address_py;

  if (!PyArg_ParseTuple(args, "O",
			&relative_grid_address_py)) {
    return NULL;
  }

  int (*relative_grid_address)[24][4][3] =
    (int(*)[24][4][3])relative_grid_address_py->data;

  spg_get_all_tetrahedra_relative_grid_address(relative_grid_address);

  Py_RETURN_NONE;
}

static PyObject *
get_tetrahedra_integration_weight(PyObject *self, PyObject *args)
{
  double omega;
  PyArrayObject* tetrahedra_omegas_py;
  char function;
  double iw;
  if (!PyArg_ParseTuple(args, "dOc",
			&omega,
			&tetrahedra_omegas_py,
			&function)) {
    return NULL;
  }
  const int num_adjacent = tetrahedra_omegas_py->dimensions[0];
  const int num_vertices = tetrahedra_omegas_py->dimensions[1];
  SPGCONST double (*tetrahedra_omegas)[num_vertices] =
    (double(*)[num_vertices])tetrahedra_omegas_py->data;
  if (num_adjacent == 24 && num_vertices == 4){ // 3 dimensional
      iw = spg_get_tetrahedra_integration_weight(omega,
                                    tetrahedra_omegas,
                                    function);
  }
  else if (num_adjacent == 6 && num_vertices == 3){ // 2 dimensional
  }
  else if (num_adjacent == 2 && num_vertices == 2){ // 1 dimensional
      iw = spg_get_tetrahedra_integration_weight_1D(omega,
                                    tetrahedra_omegas,
                                    function);
  }

  return PyFloat_FromDouble(iw);
}

static PyObject *
get_tetrahedra_integration_weight_at_omegas(PyObject *self, PyObject *args)
{
  PyArrayObject* integration_weights_py;
  PyArrayObject* omegas_py;
  PyArrayObject* tetrahedra_omegas_py;
  char function;
  if (!PyArg_ParseTuple(args, "OOOc",
			&integration_weights_py,
			&omegas_py,
			&tetrahedra_omegas_py,
			&function)) {
    return NULL;
  }
  const int num_adjacent = tetrahedra_omegas_py->dimensions[0];
  const int num_vertices = tetrahedra_omegas_py->dimensions[1];
  const double *omegas = (double*)omegas_py->data;
  double *iw = (double*)integration_weights_py->data;
  const int num_omegas = (int)omegas_py->dimensions[0];
  SPGCONST double (*tetrahedra_omegas)[num_vertices] =
    (double(*)[num_vertices])tetrahedra_omegas_py->data;

  if (num_adjacent == 24 && num_vertices == 4){ // 3 dimensional
      spg_get_tetrahedra_integration_weight_at_omegas(iw,
                              num_omegas,
                              omegas,
                              tetrahedra_omegas,
                              function);
  }
  else if (num_adjacent == 6 && num_vertices == 3){ // 2 dimensional
  }
  else if (num_adjacent == 2 && num_vertices == 2){ // 1 dimensional
      spg_get_tetrahedra_integration_weight_at_omegas_1D(iw,
                              num_omegas,
                              omegas,
                              tetrahedra_omegas,
                              function);
  }

  Py_RETURN_NONE;
}
/* kpoint.c */
/* Copyright (C) 2008 Atsushi Togo */

#include <stdio.h>
#include <stdlib.h>
#include "mathfunc.h"
#include "symmetry.h"
#include "kpoint.h"

#include "debug.h"
#define NUM_DIM_SEARCH 125

static int search_space[NUM_DIM_SEARCH][3] = {
  { 0,  0,  0},
  { 0,  0,  1},
  { 0,  0,  2},
  { 0,  0, -2},
  { 0,  0, -1},
  { 0,  1,  0},
  { 0,  1,  1},
  { 0,  1,  2},
  { 0,  1, -2},
  { 0,  1, -1},
  { 0,  2,  0},
  { 0,  2,  1},
  { 0,  2,  2},
  { 0,  2, -2},
  { 0,  2, -1},
  { 0, -2,  0},
  { 0, -2,  1},
  { 0, -2,  2},
  { 0, -2, -2},
  { 0, -2, -1},
  { 0, -1,  0},
  { 0, -1,  1},
  { 0, -1,  2},
  { 0, -1, -2},
  { 0, -1, -1},
  { 1,  0,  0},
  { 1,  0,  1},
  { 1,  0,  2},
  { 1,  0, -2},
  { 1,  0, -1},
  { 1,  1,  0},
  { 1,  1,  1},
  { 1,  1,  2},
  { 1,  1, -2},
  { 1,  1, -1},
  { 1,  2,  0},
  { 1,  2,  1},
  { 1,  2,  2},
  { 1,  2, -2},
  { 1,  2, -1},
  { 1, -2,  0},
  { 1, -2,  1},
  { 1, -2,  2},
  { 1, -2, -2},
  { 1, -2, -1},
  { 1, -1,  0},
  { 1, -1,  1},
  { 1, -1,  2},
  { 1, -1, -2},
  { 1, -1, -1},
  { 2,  0,  0},
  { 2,  0,  1},
  { 2,  0,  2},
  { 2,  0, -2},
  { 2,  0, -1},
  { 2,  1,  0},
  { 2,  1,  1},
  { 2,  1,  2},
  { 2,  1, -2},
  { 2,  1, -1},
  { 2,  2,  0},
  { 2,  2,  1},
  { 2,  2,  2},
  { 2,  2, -2},
  { 2,  2, -1},
  { 2, -2,  0},
  { 2, -2,  1},
  { 2, -2,  2},
  { 2, -2, -2},
  { 2, -2, -1},
  { 2, -1,  0},
  { 2, -1,  1},
  { 2, -1,  2},
  { 2, -1, -2},
  { 2, -1, -1},
  {-2,  0,  0},
  {-2,  0,  1},
  {-2,  0,  2},
  {-2,  0, -2},
  {-2,  0, -1},
  {-2,  1,  0},
  {-2,  1,  1},
  {-2,  1,  2},
  {-2,  1, -2},
  {-2,  1, -1},
  {-2,  2,  0},
  {-2,  2,  1},
  {-2,  2,  2},
  {-2,  2, -2},
  {-2,  2, -1},
  {-2, -2,  0},
  {-2, -2,  1},
  {-2, -2,  2},
  {-2, -2, -2},
  {-2, -2, -1},
  {-2, -1,  0},
  {-2, -1,  1},
  {-2, -1,  2},
  {-2, -1, -2},
  {-2, -1, -1},
  {-1,  0,  0},
  {-1,  0,  1},
  {-1,  0,  2},
  {-1,  0, -2},
  {-1,  0, -1},
  {-1,  1,  0},
  {-1,  1,  1},
  {-1,  1,  2},
  {-1,  1, -2},
  {-1,  1, -1},
  {-1,  2,  0},
  {-1,  2,  1},
  {-1,  2,  2},
  {-1,  2, -2},
  {-1,  2, -1},
  {-1, -2,  0},
  {-1, -2,  1},
  {-1, -2,  2},
  {-1, -2, -2},
  {-1, -2, -1},
  {-1, -1,  0},
  {-1, -1,  1},
  {-1, -1,  2},
  {-1, -1, -2},
  {-1, -1, -1}
};
// static int search_space[][3] = {
//   {0, 0, 0},
//   {0, 0, 1},
//   {0, 1, -1},
//   {0, 1, 0},
//   {0, 1, 1},
//   {1, -1, -1},
//   {1, -1, 0},
//   {1, -1, 1},
//   {1, 0, -1},
//   {1, 0, 0},
//   {1, 0, 1},
//   {1, 1, -1},
//   {1, 1, 0},
//   {1, 1, 1},
//   {-1, -1, -1},
//   {-1, -1, 0},
//   {-1, -1, 1},
//   {-1, 0, -1},
//   {-1, 0, 0},
//   {-1, 0, 1},
//   {-1, 1, -1},
//   {-1, 1, 0},
//   {-1, 1, 1},
//   {0, -1, -1},
//   {0, -1, 0},
//   {0, -1, 1},
//   {0, 0, -1}
// };

/* #define GRID_ORDER_XYZ */
/* The addressing order of mesh grid is defined as running left */
/* element first. But when GRID_ORDER_XYZ is defined, it is changed to right */ 
/* element first. */

static PointSymmetry get_point_group_reciprocal(const MatINT * rotations,
						const int is_time_reversal);
static PointSymmetry
get_point_group_reciprocal_with_q(SPGCONST PointSymmetry * pointgroup,
				  const double symprec,
				  const int num_q,
				  SPGCONST double qpoints[][3]);
static int get_ir_kpoints(int map[],
			  SPGCONST double kpoints[][3],
			  const int num_kpoint,
			  SPGCONST PointSymmetry * point_symmetry,
			  const double symprec);
static int get_ir_reciprocal_mesh(int grid_points[][3],
				  int map[],
				  const int mesh[3],
				  const int is_shift[3],
				  SPGCONST PointSymmetry * point_symmetry);
static int get_grid_point_double_mesh(const int address_double[3],
				      const int mesh[3]);
static int get_grid_point_single_mesh(const int address[3],
				      const int mesh[3]);
static int
get_ir_reciprocal_mesh_openmp(int grid_points[][3],
			      int map[],
			      const int mesh[3],
			      const int is_shift[3],
			      SPGCONST PointSymmetry * point_symmetry);
static int get_third_q_of_triplets_at_q(int bz_address[3][3],
					const int q_index,
					const int bz_map[],
					const int mesh[3],
					const int bzmesh[3],
					const int bzmesh_double[3]);
static int relocate_BZ_grid_address(int bz_grid_address[][3],
				    int bz_map[],
				    int bz_map_to_pp[],
				    SPGCONST int grid_address[][3],
				    const int mesh[3],
				    SPGCONST double rec_lattice[3][3],
				    const int is_shift[3]);
static int get_unique_tetrahedra(int *unique_vertices,
				 const int *bz_grid_address,
				 const int *bz_map, 
				 const int *relative_address,
				 const int *mesh, 
				 const int num_grid, 
				 const int dim0, 
				 const int dim1);
static double get_tolerance_for_BZ_reduction(SPGCONST double rec_lattice[3][3], const int mesh[3]);
static int get_ir_mesh_with_rot_map(int grid_points[][3],
				    int map[],
				    int rot_map[],
				    const int mesh[3],
				    const int is_shift[3],
				    SPGCONST PointSymmetry * point_symmetry);
static int
get_ir_mesh_openmp_with_rot_map(int grid_points[][3],
				int map[],
				int rot_map[],
				const int mesh[3],
				const int is_shift[3],
				SPGCONST PointSymmetry * point_symmetry);
static int get_ir_triplets_at_q(int weights[],
				int grid_points[][3],
				int third_q[],
				int map[],
				int rot_map_q[],
				const int grid_point,
				const int mesh[3],
				SPGCONST PointSymmetry * pointgroup);
static void set_grid_triplets_at_q(int triplets[][3],
				   const int q_grid_point,
				   SPGCONST int grid_points[][3],
				   const int third_q[],
				   const int weights[],
				   const int mesh[3]);
static void grid_point_to_address_double(int address_double[3],
			    const int address,
			    const int mesh[3],
			    const int is_shift[3]);
static void get_grid_address(int grid_point[3],
			    const int grid[3],
			    const int mesh[3]);
static void get_vector_modulo(int v[3], const int m[3]);
static int get_BZ_triplets_at_q(int triplets[][3],
				const int grid_point,
				SPGCONST int bz_grid_address[][3],
				const int bz_map[],
				const int map_triplets[],
				const int num_map_triplets,
				const int mesh[3]);
int kpt_get_irreducible_kpoints(int map[],
				SPGCONST double kpoints[][3],
				const int num_kpoint,
				const Symmetry * symmetry,
				const int is_time_reversal,
				const double symprec)
{
  int i;
  PointSymmetry point_symmetry;
  MatINT *rotations;
  
  rotations = mat_alloc_MatINT(symmetry->size);
  for (i = 0; i < symmetry->size; i++) {
    mat_copy_matrix_i3(rotations->mat[i], symmetry->rot[i]);
  }

  point_symmetry = get_point_group_reciprocal(rotations,
					      is_time_reversal);
  mat_free_MatINT(rotations);

  return get_ir_kpoints(map, kpoints, num_kpoint, &point_symmetry, symprec);
}

/* grid_point (e.g. 4x4x4 mesh)                               */
/*    [[ 0  0  0]                                             */
/*     [ 1  0  0]                                             */
/*     [ 2  0  0]                                             */
/*     [-1  0  0]                                             */
/*     [ 0  1  0]                                             */
/*     [ 1  1  0]                                             */
/*     [ 2  1  0]                                             */
/*     [-1  1  0]                                             */
/*     ....      ]                                            */
/*                                                            */
/* Each value of 'map' correspnds to the index of grid_point. */
int kpt_get_irreducible_reciprocal_mesh(int grid_points[][3],
					int map[],
					const int mesh[3],
					const int is_shift[3],
					const int is_time_reversal,
					const Symmetry * symmetry)
{
  int i;
  PointSymmetry point_symmetry, kpoint_symmetry;
  MatINT *rotations;
  
  rotations = mat_alloc_MatINT(symmetry->size);
  for (i = 0; i < symmetry->size; i++) {
    mat_copy_matrix_i3(rotations->mat[i], symmetry->rot[i]);
  }

  point_symmetry = get_point_group_reciprocal(rotations,
					      is_time_reversal);
	kpoint_symmetry = arbi_transform_pointsymmetry(&point_symmetry,mesh);
  mat_free_MatINT(rotations);

#ifdef _OPENMP
  return get_ir_reciprocal_mesh_openmp(grid_points,
				       map,
				       mesh,
				       is_shift,
				       &kpoint_symmetry);
#else
  return get_ir_reciprocal_mesh(grid_points,
				map,
				mesh,
				is_shift,
				&kpoint_symmetry);
#endif
  
}

int kpt_get_stabilized_reciprocal_mesh(int grid_points[][3],
				       int map[],
				       int rot_map[],
				       const int mesh[3],
				       const int is_shift[3],
				       const int is_time_reversal,
				       const MatINT * rotations,
				       const int num_q,
				       SPGCONST double qpoints[][3])
{
  PointSymmetry pointgroup, kpointgroup, pointgroup_q;
  double tolerance;
  
  pointgroup = get_point_group_reciprocal(rotations,
					  is_time_reversal);
  kpointgroup = arbi_transform_pointsymmetry(&pointgroup,mesh);
  tolerance = 0.01 / (mesh[0] + mesh[1] + mesh[2]);
  pointgroup_q = get_point_group_reciprocal_with_q(&kpointgroup,
						   tolerance,
						   num_q,
						   qpoints);

#ifdef _OPENMP
  return get_ir_mesh_openmp_with_rot_map(grid_points,
					 map,
					 rot_map,
					 mesh,
					 is_shift,
					 &pointgroup_q);
#else
  return get_ir_mesh_with_rot_map(grid_points,
				  map,
				  rot_map,
				  mesh,
				  is_shift,
				  &pointgroup_q);
#endif

}

int kpt_relocate_BZ_grid_address(int bz_grid_address[][3],
				 int bz_map[],
				 int bz_map_to_pp[],
				 SPGCONST int grid_address[][3],
				 const int mesh[3],
				 SPGCONST double rec_lattice[3][3],
				 const int is_shift[3])
{
  return relocate_BZ_grid_address(bz_grid_address,
				  bz_map,
				  bz_map_to_pp,
				  grid_address,
				  mesh,
				  rec_lattice,
				  is_shift);
}

int kpt_get_unique_tetrahedra(int *unique_vertices,
			      const int *bz_grid_address,
			      const int *bz_map, 
			      const int *relative_address,
			      const int *mesh, 
			      const int num_grid, 
			      const int dim0, 
			      const int dim1)
{
  return get_unique_tetrahedra(unique_vertices,
			       bz_grid_address,
			       bz_map,
			       relative_address,
			       mesh,
			       num_grid,
			       dim0, 
			       dim1);
}

int kpt_get_ir_triplets_at_q(int weights[],
			     int grid_points[][3],
			     int third_q[],
			     int map_q[],
			     int rot_map_q[],
			     const int grid_point,
			     const int mesh[3],
			     const int is_time_reversal,
			     const MatINT * rotations)
{
  PointSymmetry pointgroup,kpointgroup;
  pointgroup = get_point_group_reciprocal(rotations,
					  is_time_reversal);
  kpointgroup = arbi_transform_pointsymmetry(&pointgroup,mesh);
  
  return get_ir_triplets_at_q(weights,
			      grid_points,
			      third_q,
			      map_q,
			      rot_map_q,
			      grid_point,
			      mesh,
			      &kpointgroup);
}

int kpt_get_kpoint_group_at_q(int reverse_rotations[][3][3],
			      const double kpoints[][3],
			      const int mesh[3],
			      const int is_time_reversal,
			      const int num_rot,
			      const int num_q,
			      const MatINT * rotations)
{
  double tolerance; 
  int i;
  PointSymmetry pointgroup,kpointgroup, pointgroup_q;
  pointgroup = get_point_group_reciprocal(rotations,
					  is_time_reversal);
  kpointgroup = arbi_transform_pointsymmetry(&pointgroup,mesh);
  tolerance = 0.1 / (mesh[0] + mesh[1] + mesh[2]);
  pointgroup_q = get_point_group_reciprocal_with_q(&kpointgroup,
						   tolerance,
						   num_q,
						   kpoints);
  for (i=0; i<pointgroup_q.size; i++)
     mat_copy_matrix_i3(reverse_rotations[i], pointgroup_q.rot[i]);
  return pointgroup_q.size;
  
}

void kpt_set_grid_triplets_at_q(int triplets[][3],
				const int q_grid_point,
				SPGCONST int grid_points[][3],
				const int third_q[],
				const int weights[],
				const int mesh[3])
{
  set_grid_triplets_at_q(triplets,
			 q_grid_point,
			 grid_points,
			 third_q,
			 weights,
			 mesh);
}


      

/* qpoints are used to find stabilizers (operations). */
/* num_q is the number of the qpoints. */
static PointSymmetry get_point_group_reciprocal(const MatINT * rotations,
						const int is_time_reversal)
{
  int i, j, num_pt = 0;
  MatINT *rot_reciprocal;
  PointSymmetry point_symmetry;
  SPGCONST int inversion[3][3] = {
    {-1, 0, 0 },
    { 0,-1, 0 },
    { 0, 0,-1 }
  };
  
  if (is_time_reversal) {
    rot_reciprocal = mat_alloc_MatINT(rotations->size * 2);
  } else {
    rot_reciprocal = mat_alloc_MatINT(rotations->size);
  }

  for (i = 0; i < rotations->size; i++) {
    mat_transpose_matrix_i3(rot_reciprocal->mat[i], rotations->mat[i]);
    
    if (is_time_reversal) {
      mat_multiply_matrix_i3(rot_reciprocal->mat[rotations->size+i],
			     inversion,
			     rot_reciprocal->mat[i]);
    }
  }


  for (i = 0; i < rot_reciprocal->size; i++) {
    for (j = 0; j < num_pt; j++) {
      if (mat_check_identity_matrix_i3(point_symmetry.rot[j],
				       rot_reciprocal->mat[i])) {
	goto escape;
      }
    }
    
    mat_copy_matrix_i3(point_symmetry.rot[num_pt],
		       rot_reciprocal->mat[i]);
    num_pt++;
  escape:
    ;
  }

  point_symmetry.size = num_pt;

  mat_free_MatINT(rot_reciprocal);

  return point_symmetry;
}

static PointSymmetry
get_point_group_reciprocal_with_q(SPGCONST PointSymmetry * pointgroup,
				  const double symprec,
				  const int num_q,
				  SPGCONST double qpoints[][3])
{
  int i, j, k, l, is_all_ok=0, num_ptq = 0;
  double q_rot[3], diff[3];
  PointSymmetry pointgroup_q;

  for (i = 0; i < pointgroup->size; i++) {
    for (j = 0; j < num_q; j++) {
      is_all_ok = 0;
      mat_multiply_matrix_vector_id3(q_rot,
				     pointgroup->rot[i],
				     qpoints[j]);

      for (k = 0; k < num_q; k++) {
	for (l = 0; l < 3; l++) {
	  diff[l] = q_rot[l] - qpoints[k][l];
	  diff[l] -= mat_Nint(diff[l]);
	}
	
	if (mat_Dabs(diff[0]) < symprec && 
	    mat_Dabs(diff[1]) < symprec &&
	    mat_Dabs(diff[2]) < symprec) {
	  is_all_ok = 1;
	  break;
	}
      }
      
      if (! is_all_ok) {
	break;
      }
    }

    if (is_all_ok) {
      mat_copy_matrix_i3(pointgroup_q.rot[num_ptq], pointgroup->rot[i]);
      num_ptq++;
    }
  }
  pointgroup_q.size = num_ptq;

  return pointgroup_q;
}


static int get_ir_kpoints(int map[],
			  SPGCONST double kpoints[][3],
			  const int num_kpoint,
			  SPGCONST PointSymmetry * point_symmetry,
			  const double symprec)
{
  int i, j, k, l, num_ir_kpoint = 0, is_found;
  int *ir_map;
  double kpt_rot[3], diff[3];

  ir_map = (int*)malloc(num_kpoint*sizeof(int));

  for (i = 0; i < num_kpoint; i++) {

    map[i] = i;

    is_found = 1;

    for (j = 0; j < point_symmetry->size; j++) {
      mat_multiply_matrix_vector_id3(kpt_rot, point_symmetry->rot[j], kpoints[i]);

      for (k = 0; k < 3; k++) {
	diff[k] = kpt_rot[k] - kpoints[i][k];
	diff[k] = diff[k] - mat_Nint(diff[k]);
      }

      if (mat_Dabs(diff[0]) < symprec && 
	  mat_Dabs(diff[1]) < symprec && 
	  mat_Dabs(diff[2]) < symprec) {
	continue;
      }
      
      for (k = 0; k < num_ir_kpoint; k++) {
	mat_multiply_matrix_vector_id3(kpt_rot, point_symmetry->rot[j], kpoints[i]);

	for (l = 0; l < 3; l++) {
	  diff[l] = kpt_rot[l] - kpoints[ir_map[k]][l];
	  diff[l] = diff[l] - mat_Nint(diff[l]);
	}

	if (mat_Dabs(diff[0]) < symprec && 
	    mat_Dabs(diff[1]) < symprec && 
	    mat_Dabs(diff[2]) < symprec) {
	  is_found = 0;
	  map[i] = ir_map[k];
	  break;
	}
      }

      if (! is_found)
	break;
    }

    if (is_found) {
      ir_map[num_ir_kpoint] = i;
      num_ir_kpoint++;
    }
  }

  free(ir_map);
  ir_map = NULL;

  return num_ir_kpoint;
}

static int get_ir_mesh_with_rot_map(int grid_address[][3],
				  int map[],
				  int rot_map[],
				  const int mesh[3],
				  const int is_shift[3],
				  SPGCONST PointSymmetry * point_symmetry)
{
  /* In the following loop, mesh is doubled. */
  /* Even and odd mesh numbers correspond to */
  /* is_shift[i] = 0 and 1, respectively. */
  /* is_shift = [0,0,0] gives Gamma center mesh. */
  /* grid: reducible grid points */
  /* map: the mapping from each point to ir-point. */

  int i, j, k, l, grid_point, grid_point_rot, num_ir = 0;
  int grid_address_double[3], grid_address_rot[3], mesh_double[3];

  for (i = 0; i < 3; i++) {
    mesh_double[i] = mesh[i] * 2;
  }

  /* "-1" means the element is not touched yet. */
  for (i = 0; i < mesh[0] * mesh[1] * mesh[2]; i++) {
    map[i] = -1;
  }

#ifndef GRID_ORDER_XYZ
  for (i = 0; i < mesh[2]; i++) {
    for (j = 0; j < mesh[1]; j++) {
      for (k = 0; k < mesh[0]; k++) {
	grid_address_double[0] = k * 2 + is_shift[0];
	grid_address_double[1] = j * 2 + is_shift[1];
	grid_address_double[2] = i * 2 + is_shift[2];
#else
  for (i = 0; i < mesh[0]; i++) {
    for (j = 0; j < mesh[1]; j++) {
      for (k = 0; k < mesh[2]; k++) {
  	grid_address_double[0] = i * 2 + is_shift[0];
  	grid_address_double[1] = j * 2 + is_shift[1];
  	grid_address_double[2] = k * 2 + is_shift[2];
#endif	

	grid_point = get_grid_point_double_mesh(grid_address_double, mesh);
	get_grid_address(grid_address[grid_point], grid_address_double, mesh);

	for (l = 0; l < point_symmetry->size; l++) {
	  mat_multiply_matrix_vector_i3(grid_address_rot,
					point_symmetry->rot[l],	grid_address_double);
	  get_vector_modulo(grid_address_rot, mesh_double);
	  grid_point_rot = get_grid_point_double_mesh(grid_address_rot, mesh);

	  if (grid_point_rot > -1) { /* Invalid if even --> odd or odd --> even */
	    if (map[grid_point_rot] > -1) {
	      map[grid_point] = map[grid_point_rot];
	      rot_map[grid_point] = 0;
	      break;
	    }
	  }
	}
	
	if (map[grid_point] == -1) {
	  map[grid_point] = grid_point;
	  rot_map[grid_point] = l;
	  num_ir++;
	}
      }
    }
  }

  return num_ir;
}

static int
get_ir_mesh_openmp_with_rot_map(int grid_address[][3],
			      int map[],
			      int rot_map[],
			      const int mesh[3],
			      const int is_shift[3],
			      SPGCONST PointSymmetry * point_symmetry)
{
  int i, j, k, l, grid_point, grid_point_rot, num_ir;
  int grid_address_double[3], grid_address_rot[3], mesh_double[3];

  for (i = 0; i < 3; i++) {
    mesh_double[i] = mesh[i] * 2;
  }

#ifndef GRID_ORDER_XYZ
#pragma omp parallel for private(j, k, l, grid_point, grid_point_rot, grid_address_double, grid_address_rot)
  for (i = 0; i < mesh[2]; i++) {
    for (j = 0; j < mesh[1]; j++) {
      for (k = 0; k < mesh[0]; k++) {
	grid_address_double[0] = k * 2 + is_shift[0];
	grid_address_double[1] = j * 2 + is_shift[1];
	grid_address_double[2] = i * 2 + is_shift[2];
#else
#pragma omp parallel for private(j, k, l, grid_point, grid_point_rot, grid_address_double, grid_address_rot)
  for (i = 0; i < mesh[0]; i++) {
    for (j = 0; j < mesh[1]; j++) {
      for (k = 0; k < mesh[2]; k++) {
  	grid_address_double[0] = i * 2 + is_shift[0];
  	grid_address_double[1] = j * 2 + is_shift[1];
  	grid_address_double[2] = k * 2 + is_shift[2];
#endif	

	grid_point = get_grid_point_double_mesh(grid_address_double, mesh);
	map[grid_point] = grid_point;
	rot_map[grid_point] = 0; //eye Matrix
	get_grid_address(grid_address[grid_point], grid_address_double, mesh);

	for (l = 0; l < point_symmetry->size; l++) {
	  mat_multiply_matrix_vector_i3(grid_address_rot,
					point_symmetry->rot[l],	grid_address_double);
	  get_vector_modulo(grid_address_rot, mesh_double);
	  grid_point_rot = get_grid_point_double_mesh(grid_address_rot, mesh);

	  if (grid_point_rot > -1) { /* Invalid if even --> odd or odd --> even */
	    if (grid_point_rot < map[grid_point]) {
	      map[grid_point] = grid_point_rot;
	      rot_map[grid_point] = l;
	    }
	  }
	}
      }
    }
  }

  num_ir = 0;

#pragma omp parallel for reduction(+:num_ir)
  for (i = 0; i < mesh[0] * mesh[1] * mesh[2]; i++) {
    if (map[i] == i) {
      num_ir++;
    }
  }
  
  return num_ir;
}


static int get_ir_reciprocal_mesh(int grid_address[][3],
				  int map[],
				  const int mesh[3],
				  const int is_shift[3],
				  SPGCONST PointSymmetry * point_symmetry)
{
  /* In the following loop, mesh is doubled. */
  /* Even and odd mesh numbers correspond to */
  /* is_shift[i] = 0 and 1, respectively. */
  /* is_shift = [0,0,0] gives Gamma center mesh. */
  /* grid: reducible grid points */
  /* map: the mapping from each point to ir-point. */

  int i, j, k, l, grid_point, grid_point_rot, num_ir = 0;
  int grid_address_double[3], grid_address_rot[3], mesh_double[3];

  for (i = 0; i < 3; i++) {
    mesh_double[i] = mesh[i] * 2;
  }

  /* "-1" means the element is not touched yet. */
  for (i = 0; i < mesh[0] * mesh[1] * mesh[2]; i++) {
    map[i] = -1;
  }

#ifndef GRID_ORDER_XYZ
  for (i = 0; i < mesh[2]; i++) {
    for (j = 0; j < mesh[1]; j++) {
      for (k = 0; k < mesh[0]; k++) {
	grid_address_double[0] = k * 2 + is_shift[0];
	grid_address_double[1] = j * 2 + is_shift[1];
	grid_address_double[2] = i * 2 + is_shift[2];
#else
  for (i = 0; i < mesh[0]; i++) {
    for (j = 0; j < mesh[1]; j++) {
      for (k = 0; k < mesh[2]; k++) {
  	grid_address_double[0] = i * 2 + is_shift[0];
  	grid_address_double[1] = j * 2 + is_shift[1];
  	grid_address_double[2] = k * 2 + is_shift[2];
#endif	

	grid_point = get_grid_point_double_mesh(grid_address_double, mesh);
	get_grid_address(grid_address[grid_point], grid_address_double, mesh);

	for (l = 0; l < point_symmetry->size; l++) {
	  mat_multiply_matrix_vector_i3(grid_address_rot,
					point_symmetry->rot[l],	grid_address_double);
	  get_vector_modulo(grid_address_rot, mesh_double);
	  grid_point_rot = get_grid_point_double_mesh(grid_address_rot, mesh);

	  if (grid_point_rot > -1) { /* Invalid if even --> odd or odd --> even */
	    if (map[grid_point_rot] > -1) {
	      map[grid_point] = map[grid_point_rot];
	      break;
	    }
	  }
	}
	
	if (map[grid_point] == -1) {
	  map[grid_point] = grid_point;
	  num_ir++;
	}
      }
    }
  }

  return num_ir;
}

static int
get_ir_reciprocal_mesh_openmp(int grid_address[][3],
			      int map[],
			      const int mesh[3],
			      const int is_shift[3],
			      SPGCONST PointSymmetry * point_symmetry)
{
  int i, j, k, l, grid_point, grid_point_rot, num_ir;
  int grid_address_double[3], grid_address_rot[3], mesh_double[3];

  for (i = 0; i < 3; i++) {
    mesh_double[i] = mesh[i] * 2;
  }

#ifndef GRID_ORDER_XYZ
#pragma omp parallel for private(j, k, l, grid_point, grid_point_rot, grid_address_double, grid_address_rot)
  for (i = 0; i < mesh[2]; i++) {
    for (j = 0; j < mesh[1]; j++) {
      for (k = 0; k < mesh[0]; k++) {
	grid_address_double[0] = k * 2 + is_shift[0];
	grid_address_double[1] = j * 2 + is_shift[1];
	grid_address_double[2] = i * 2 + is_shift[2];
#else
#pragma omp parallel for private(j, k, l, grid_point, grid_point_rot, grid_address_double, grid_address_rot)
  for (i = 0; i < mesh[0]; i++) {
    for (j = 0; j < mesh[1]; j++) {
      for (k = 0; k < mesh[2]; k++) {
  	grid_address_double[0] = i * 2 + is_shift[0];
  	grid_address_double[1] = j * 2 + is_shift[1];
  	grid_address_double[2] = k * 2 + is_shift[2];
#endif	

	grid_point = get_grid_point_double_mesh(grid_address_double, mesh);
	map[grid_point] = grid_point;
	get_grid_address(grid_address[grid_point], grid_address_double, mesh);

	for (l = 0; l < point_symmetry->size; l++) {
	  mat_multiply_matrix_vector_i3(grid_address_rot,
					point_symmetry->rot[l],	grid_address_double);
	  get_vector_modulo(grid_address_rot, mesh_double);
	  grid_point_rot = get_grid_point_double_mesh(grid_address_rot, mesh);

	  if (grid_point_rot > -1) { /* Invalid if even --> odd or odd --> even */
	    if (grid_point_rot < map[grid_point]) {
	      map[grid_point] = grid_point_rot;
	    }
	  }
	}
      }
    }
  }

  num_ir = 0;

#pragma omp parallel for reduction(+:num_ir)
  for (i = 0; i < mesh[0] * mesh[1] * mesh[2]; i++) {
    if (map[i] == i) {
      num_ir++;
    }
  }
  
  return num_ir;
}

/* Relocate grid grid_pointes to first Brillouin zone */
/* bz_grid_grid_point[prod(mesh + 1)][3] */
/* bz_map[prod(mesh * 2)] */
static int relocate_BZ_grid_address(int bz_grid_address[][3],
				    int bz_map[],
				    int bz_map_to_pp[],
				    SPGCONST int grid_address[][3],
				    const int mesh[3],
				    SPGCONST double rec_lattice[3][3],
				    const int is_shift[3])
{
  double tolerance, min_distance;
  double q_vector[3], distance[NUM_DIM_SEARCH];
  int bzmesh[3], bzmesh_double[3], bz_address_double[3];
  int i, j, k, min_index, boundary_num_gp, total_num_gp, bzgp, gp;

  tolerance = get_tolerance_for_BZ_reduction(rec_lattice, mesh);
  for (i = 0; i < 3; i++) {
    bzmesh[i] = mesh[i] * 2;
    bzmesh_double[i] = bzmesh[i] * 2;
  }
  for (i = 0; i < bzmesh[0] * bzmesh[1] * bzmesh[2]; i++) {
    bz_map[i] = -1;
  }
  for (i=0; i<(mesh[0]+1)*(mesh[1]+1)*(mesh[2]+1); i++){
    bz_map_to_pp[i]=-1;
  }
  boundary_num_gp = 0;
  total_num_gp = mesh[0] * mesh[1] * mesh[2];
  for (i = 0; i < total_num_gp; i++) {
    for (j = 0; j < NUM_DIM_SEARCH; j++) {
      for (k = 0; k < 3; k++) {
	q_vector[k] = 
	  ((grid_address[i][k] + search_space[j][k] * mesh[k]) * 2 +
	   is_shift[k]) / ((double)mesh[k]) / 2;
      }
      mat_multiply_matrix_vector_d3(q_vector, rec_lattice, q_vector);
      distance[j] = mat_norm_squared_d3(q_vector);
    }
    min_distance = distance[0];
    min_index = 0;
    for (j = 1; j < NUM_DIM_SEARCH; j++) {
      if (distance[j] < min_distance) {
	min_distance = distance[j];
	min_index = j;
      }
    }

    for (j = 0; j < NUM_DIM_SEARCH; j++) {
      if (distance[j] < min_distance + tolerance) {
	if (j == min_index) {
	  gp = i;
	} else {
	  gp = boundary_num_gp + total_num_gp;
	}
	bz_map_to_pp[gp]=i;
	for (k = 0; k < 3; k++) {
	  bz_grid_address[gp][k] = 
	    grid_address[i][k] + search_space[j][k] * mesh[k];
	  bz_address_double[k] = bz_grid_address[gp][k] * 2 + is_shift[k];
	}
	get_vector_modulo(bz_address_double, bzmesh_double);
	bzgp = get_grid_point_double_mesh(bz_address_double, bzmesh);
	bz_map[bzgp] = gp;
	if (j != min_index) {
	  boundary_num_gp++;
	}
      }
    }
  }

  return boundary_num_gp + total_num_gp;
}

void kpt_get_neighboring_grid_points(int neighboring_grid_points[],
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

static int get_unique_tetrahedra(int *unique,
				 const int *bz_grid_address,
				 const int *bz_map, 
				 const int *relative_address,
				 const int *mesh, 
				 const int num_grid, 
				 const int dim0, 
				 const int dim1)
{
  int i,j,k,l, num_unique=0,exists, bz_gp, is_skip, is_equal;
  int bzmesh[3], add_temp[3], vgp[dim1];
  for (i=0;i<3;i++)
  {
    bzmesh[i]=mesh[i]*2;
  }

  for (i=0;i<num_grid;i++)
  {
    for (j=0;j<dim0;j++)
    {
      for (k=0;k<dim1;k++)
      {
	for (l=0;l<3;l++)
	{
	  
	  add_temp[l]=(bz_grid_address[i*3+l] + relative_address[j*dim1*3+k*3+l]) % bzmesh[l];
	  if (add_temp[l]<0)
	    add_temp[l] += bzmesh[l];
	}
	bz_gp=add_temp[0]+add_temp[1]*bzmesh[0]+add_temp[2]*bzmesh[0]*bzmesh[1];

	vgp[k]=bz_map[bz_gp];
      }
      
      
      is_skip=0;
      for (k=0; k<dim1; k++)
      {
	if(vgp[k] == -1)
	{
	  is_skip = 1;
	  break;
	}
      }
      if (is_skip) continue;
   
      exists=0;
      for (l=0;l<num_unique;l++)
      {
	
	is_equal=1;
	for (k=0; k<dim1; k++)
	{
	  if (vgp[k] != unique[l*dim1+k])
	  {
	    is_equal=0;
	    break;
	  }
	}
	if (is_equal)
	{
	  exists=1;
	  break;
	}
      }
      if (exists==0)
      {
	for (k=0;k<dim1;k++)
	{
	  unique[num_unique*dim1+k] = vgp[k];
	}
	num_unique++;
      }
    }
  }
  return num_unique;
}

static double get_tolerance_for_BZ_reduction(SPGCONST double rec_lattice[3][3],
					     const int mesh[3])
{
  
  int i, j;
  double tolerance;
  double length[3];
  
  for (i = 0; i < 3; i++) {
    length[i] = 0;
    for (j = 0; j < 3; j++) {
      length[i] += rec_lattice[j][i] * rec_lattice[j][i];
    }
    length[i] /= mesh[i] * mesh[i];
  }
  tolerance = length[0];
  for (i = 1; i < 3; i++) {
    if (tolerance < length[i]) {
      tolerance = length[i];
    }
  }
  tolerance *= 0.01;
  
  return tolerance;
}

static int get_ir_triplets_at_q(int weights[],
				int grid_address[][3],
				int third_q[],
				int map[],
				int rot_map_q[],
				const int grid_point,
				const int mesh[3],
				SPGCONST PointSymmetry * pointgroup)
{
  int i, j, num_grid, q_2, num_ir_q, num_ir_triplets, ir_address;
  int mesh_double[3], is_shift[3];
  int address_double0[3], address_double1[3], address_double2[3];
  int *map_q, *ir_addresses, *weight_q;
  double tolerance;
  double stabilizer_q[1][3];
  PointSymmetry pointgroup_q;

  tolerance = 0.1 / (mesh[0] + mesh[1] + mesh[2]);

  num_grid = mesh[0] * mesh[1] * mesh[2];

  for (i = 0; i < 3; i++) {
    /* Only consider the gamma-point */
    is_shift[i] = 0;
    mesh_double[i] = mesh[i] * 2;
  }

  /* Search irreducible q-points (map_q) with a stabilizer */
  grid_point_to_address_double(address_double0, grid_point, mesh, is_shift); /* q */
  for (i = 0; i < 3; i++) {
    stabilizer_q[0][i] = (double)address_double0[i] / mesh_double[i];
  }

  pointgroup_q = get_point_group_reciprocal_with_q(pointgroup,
						   tolerance,
						   1,
						   stabilizer_q);
  map_q = (int*) malloc(sizeof(int) * num_grid);

#ifdef _OPENMP
  num_ir_q = get_ir_mesh_openmp_with_rot_map(grid_address,
					     map_q,
					     rot_map_q,
					     mesh,
					     is_shift,
					     &pointgroup_q);
#else
  num_ir_q = get_ir_mesh_with_rot_map(grid_address,
				      map_q,
				      rot_map_q,
				      mesh,
				      is_shift,
				      &pointgroup_q);
#endif

  ir_addresses = (int*) malloc(sizeof(int) * num_ir_q);
  weight_q = (int*) malloc(sizeof(int) * num_grid);
  num_ir_q = 0;
  for (i = 0; i < num_grid; i++) {
    if (map_q[i] == i) {
      ir_addresses[num_ir_q] = i;
      num_ir_q++;
    }
    weight_q[i] = 0;
    third_q[i] = -1;
    weights[i] = 0;
  }

  for (i = 0; i < num_grid; i++) {
    map[i] = map_q[i]; // initialize map
    weight_q[map_q[i]]++;
  }

#pragma omp parallel for private(j, address_double1, address_double2)
  for (i = 0; i < num_ir_q; i++) {
    grid_point_to_address_double(address_double1, ir_addresses[i], mesh, is_shift); /* q' */
    for (j = 0; j < 3; j++) { /* q'' */
      address_double2[j] = - address_double0[j] - address_double1[j];
    }
    get_vector_modulo(address_double2, mesh_double);
    third_q[ir_addresses[i]] = get_grid_point_double_mesh(address_double2, mesh);
  }

  num_ir_triplets = 0;
//interchangable q' and q''
  
//   for (i = 0; i < num_ir_q; i++) {
//     ir_address = ir_addresses[i];
//     q_2 = third_q[ir_address];
//     if (weights[map_q[q_2]]) {
//       weights[map_q[q_2]] += weight_q[ir_address];
//       for (j = 0; j < num_grid; j++){
// 	if (map_q[j] == ir_address){
// 	  map[j] = map_q[q_2];
// 	  mat_multiply_matrix_i3(rot_map_q[j],rot_map_q[q_2],rot_map_q[j]); //R(1<-4) = R(1<-2) * R(3<-4) ( 2 and 3 are interchangable) 
// 	}
//       }
//     } else {
//       weights[ir_address] = weight_q[ir_address];
//       num_ir_triplets++;
//     }
//   }
  
  for (i=0;i<num_ir_q;i++){
    ir_address = ir_addresses[i];
    weights[ir_address] = weight_q[ir_address];
    num_ir_triplets++;
  }

  free(map_q);
  map_q = NULL;
  free(weight_q);
  weight_q = NULL;
  free(ir_addresses);
  ir_addresses = NULL;

  return num_ir_triplets;
}

void kpt_get_grid_points_by_rotations(int rot_grid_points[],
				      const int address_orig[3],
				      const MatINT * rot_reciprocal,
				      const int mesh[3],
				      const int is_shift[3])
{
  int i;
  int address_double_orig[3], address_double[3];

  for (i = 0; i < 3; i++) {
    address_double_orig[i] = address_orig[i] * 2 + is_shift[i];
  }
  for (i = 0; i < rot_reciprocal->size; i++) {
    mat_multiply_matrix_vector_i3(address_double,
				  rot_reciprocal->mat[i],
				  address_double_orig);
    rot_grid_points[i] = get_grid_point_double_mesh(address_double, mesh);
  }
}

void kpt_get_BZ_grid_points_by_rotations(int rot_grid_points[],
					 const int address_orig[3],
					 const MatINT * rot_reciprocal,
					 const int mesh[3],
					 const int is_shift[3],
					 const int bz_map[])
{
  int i;
  int address_double_orig[3], address_double[3], mesh_double[3], bzmesh_double[3];

  for (i = 0; i < 3; i++) {
    mesh_double[i] = mesh[i] * 2;
    bzmesh_double[i] = mesh[i] * 4;
    address_double_orig[i] = address_orig[i] * 2 + is_shift[i];
  }
  for (i = 0; i < rot_reciprocal->size; i++) {
    mat_multiply_matrix_vector_i3(address_double,
				  rot_reciprocal->mat[i],
				  address_double_orig);
    get_vector_modulo(address_double, bzmesh_double);
    rot_grid_points[i] =
      bz_map[get_grid_point_double_mesh(address_double, mesh_double)];
  }
}

int kpt_get_BZ_triplets_at_q(int triplets[][3],
			     const int grid_point,
			     SPGCONST int bz_grid_address[][3],
			     const int bz_map[],
			     const int map_triplets[],
			     const int num_map_triplets,
			     const int mesh[3])
{
  return get_BZ_triplets_at_q(triplets,
			      grid_point,
			      bz_grid_address,
			      bz_map,
			      map_triplets,
			      num_map_triplets,
			      mesh);
}

int  reduce_triplets_permute_sym(int triplet_mappings[],
				 char sequence[][3],
				 const int triplets[][3],
				 const int grid_points[],
				 const int triplet_numbers[],
				 const int mesh[3],
				 const int first_mapping[],
				 const int first_rotation[][3][3],     
				 const int second_mapping[],
				 const int num_grid, 
				 const int num_grid_all)
{
  int i, j, k, m, q0, q1, q2,i_prime;
  int is_shift[3], mesh_double[3], vtmp[3];
  int num_triplets_accum[num_grid];
  int num_irred_triplets=0;
  
  for (i=0; i<3; i++)
  {
    is_shift[i] = 0;
    mesh_double[i] = mesh[i] * 2;
  }
  num_triplets_accum[0] = 0;
  for (i=1; i<num_grid; i++)
    num_triplets_accum[i] = num_triplets_accum[i-1] + triplet_numbers[i-1];
 #pragma omp parallel for reduction(+:num_irred_triplets) private(j, k, q0, q1, q2, m, i_prime, vtmp)
  for (i = 0; i < num_grid; i++)
  {
    for (j=num_triplets_accum[i]; j<num_triplets_accum[i]+triplet_numbers[i]; j++)
    {
      q0 = triplets[j][0];
      q1 = triplets[j][1];
      q2 = triplets[j][2];
      sequence[j][0] = 0; sequence[j][1] = 1; sequence[j][2] = 2;
      i_prime = i;
      if (first_mapping[q1] < q0 || first_mapping[q2] < q0)
      {
	      if(first_mapping[q1] < first_mapping[q2])
	      {
	        m = q0; q0 = q1; q1 = m;
	        sequence[j][0] = 1;
	        sequence[j][1] = 0;
	        sequence[j][2] = 2;
	      }
	      else
	      {
	        m = q0; q0 = q2; q2 = m;
	        sequence[j][0] = 2;
	        sequence[j][1] = 1;
	        sequence[j][2] = 0;
	      }
	
	      i_prime = mat_index_from_Iarray(grid_points, num_grid, first_mapping[q0]);
	      if (i_prime != -1)
	      {
	        grid_point_to_address_double(vtmp, q1, mesh, is_shift);
	        mat_multiply_matrix_vector_i3(vtmp, first_rotation[q0], vtmp);
	        get_vector_modulo(vtmp, mesh_double);
	        q1 = get_grid_point_double_mesh(vtmp, mesh);
	
	        grid_point_to_address_double(vtmp, q2, mesh, is_shift);
	        mat_multiply_matrix_vector_i3(vtmp, first_rotation[q0], vtmp);
	        get_vector_modulo(vtmp, mesh_double);
	        q2 = get_grid_point_double_mesh(vtmp, mesh);
	      }
	      else // In case of no matches
	      {
          q0 = triplets[j][0];
          q1 = triplets[j][1];
          q2 = triplets[j][2];
          sequence[j][0] = 0; sequence[j][1] = 1; sequence[j][2] = 2;
          i_prime = i;
	      }
      }
      
      k=q1; m=q2;
      q1 = second_mapping[i_prime * num_grid_all + q1];
      q2 = second_mapping[i_prime * num_grid_all + q2];

      if(q2 < q1 || (q2 == q1 && m < k))
      {
	      q1 = q2;
	      vtmp[0] = 0; vtmp[1] = 2; vtmp[2] = 1;
	      sequence[j][0] = vtmp[sequence[j][0]];
	      sequence[j][1] = vtmp[sequence[j][1]];
	      sequence[j][2] = vtmp[sequence[j][2]];
      }
 
      if (sequence[j][0] == 0 && sequence[j][1] == 1) // triplet sequence unchaged
      {
	      triplet_mappings[j] = j;
	      num_irred_triplets++;
      }
      else
      {
	      m = -1;
	      for (k=num_triplets_accum[i_prime]; k<triplet_numbers[i_prime]+num_triplets_accum[i_prime]; k++)
	        if (triplets[k][1] == q1)
	        {
	          m = k;
	          break;
	        }
	      if (m == -1)
	      {
	        printf("Warning! index of triplet not found!\n");
	        printf("Triplet: [%d, %d, %d]\n",triplets[j][0],triplets[j][1],triplets[j][2]);
	        triplet_mappings[j] = j;
	        sequence[j][0] = 0; sequence[j][1] = 1; sequence[j][2] = 2;
	        num_irred_triplets++;
	      }
	      else
	        triplet_mappings[j] = m;
      }
    }
  }
  return num_irred_triplets;
}

int  reduce_pairs_permute_sym(int pair_mappings[],
				 char sequence[][2],
				 const int pairs[][2],
				 const int grid_points[],
				 const int pair_numbers[],
				 const int mesh[3],
				 const int first_mapping[],
				 const int first_rotation[][3][3],     
				 const int second_mapping[],
				 const int num_grid, 
				 const int num_grid_all)
{
  int i, j, k, m, q0, q1,i_prime;
  int is_shift[3], mesh_double[3], vtmp[3];
  int num_pairs_accum[num_grid];
  int num_irred_pairs=0;
  
  for (i=0; i<3; i++)
  {
    is_shift[i] = 0;
    mesh_double[i] = mesh[i] * 2;
  }
  num_pairs_accum[0] = 0;
  for (i=1; i<num_grid; i++)
    num_pairs_accum[i] = num_pairs_accum[i-1] + pair_numbers[i-1];
 #pragma omp parallel for reduction(+:num_irred_pairs) private(j, k, q0, q1, m, i_prime, vtmp)
  for (i = 0; i < num_grid; i++)
  {
    for (j=num_pairs_accum[i]; j<num_pairs_accum[i]+pair_numbers[i]; j++)
    {
      q0 = pairs[j][0];
      q1 = pairs[j][1];
      sequence[j][0] = 0; sequence[j][1] = 1;
      i_prime = i;
      if (first_mapping[q1] < q0)
      {
	      m = q0; q0 = q1; q1 = m;
	      sequence[j][0] = 1;
	      sequence[j][1] = 0;
	
	      i_prime = mat_index_from_Iarray(grid_points, num_grid, first_mapping[q0]);
	      grid_point_to_address_double(vtmp, q1, mesh, is_shift);
	      mat_multiply_matrix_vector_i3(vtmp, first_rotation[q0], vtmp);
	      get_vector_modulo(vtmp, mesh_double);
	      q1 = get_grid_point_double_mesh(vtmp, mesh);
      }
      else
	      num_irred_pairs++;
      
      k=q1;
      q1 = second_mapping[i_prime * num_grid_all + q1];
      m = -1;
      for (k=num_pairs_accum[i_prime]; k<pair_numbers[i_prime]+num_pairs_accum[i_prime]; k++)
	      if (pairs[k][1] == q1)
	      {
	        m = k;
	        break;
	      }
      if (m == -1)
      {
	      printf("Warning! index of pair not found!\n");
	      printf("pair: [%d, %d]\n",pairs[j][0],pairs[j][1]);
	      pair_mappings[j] = j;
	      sequence[j][0] = 0; sequence[j][1] = 1;
	      num_irred_pairs++;
      }
      else
	      pair_mappings[j] = m;
    }
  }
  return num_irred_pairs;
}



static void set_grid_triplets_at_q(int triplets[][3],
				   const int q_grid_point,
				   SPGCONST int grid_address[][3],
				   const int third_q[],
				   const int weights[],
				   const int mesh[3])
{
  const int is_shift[3] = {0, 0, 0};
  int i, j, k, num_edge, edge_pos, num_ir;
  int address_double[3][3], ex_mesh[3], ex_mesh_double[3];

  for (i = 0; i < 3; i++) {
    ex_mesh[i] = mesh[i] + (mesh[i] % 2 == 0);
    ex_mesh_double[i] = ex_mesh[i] * 2;
  }

  for (i = 0; i < 3; i++) {
    address_double[0][i] = grid_address[q_grid_point][i] * 2;
  }

  num_ir = 0;

  for (i = 0; i < mesh[0] * mesh[1] * mesh[2]; i++) {
    if (weights[i] < 1) {
      continue;
    }

    for (j = 0; j < 3; j++) {
      address_double[1][j] = grid_address[i][j] * 2;
      address_double[2][j] = grid_address[third_q[i]][j] * 2;
    }

    for (j = 0; j < 3; j++) {
      num_edge = 0;
      edge_pos = -1;
      for (k = 0; k < 3; k++) {
	if (abs(address_double[k][j]) == mesh[j]) {
	  num_edge++;
	  edge_pos = k;
	}
      }
      if (num_edge == 1) { // for the case that one qpoint is on the edge, directly set q(edge)=-q1-q2
	address_double[edge_pos][j] = 0;
	for (k = 0; k < 3; k++) {
	  if (k != edge_pos) {
	    address_double[edge_pos][j] -= address_double[k][j];
	  }
	}
      }
      if (num_edge == 2) {
	address_double[edge_pos][j] = -address_double[edge_pos][j]; //set q(1edge)=-q(1edge); q(2edge)=-q(2dge)
      }
    }

    for (j = 0; j < 3; j++) {
      get_vector_modulo(address_double[j], ex_mesh_double);
      triplets[num_ir][j] = get_grid_point_double_mesh(address_double[j], ex_mesh);
    }
    
    num_ir++;
  }

}

static void grid_point_to_address_double(int address_double[3],
					 const int grid_point,
					 const int mesh[3],
					 const int is_shift[3])
{
  int i;
  int address[3];

#ifndef GRID_ORDER_XYZ
  address[2] = grid_point / (mesh[0] * mesh[1]);
  address[1] = (grid_point - address[2] * mesh[0] * mesh[1]) / mesh[0];
  address[0] = grid_point % mesh[0];
#else
  address[0] = grid_point / (mesh[1] * mesh[2]);
  address[1] = (grid_point - address[0] * mesh[1] * mesh[2]) / mesh[2];
  address[2] = grid_point % mesh[2];
#endif

  for (i = 0; i < 3; i++) {
    address_double[i] = address[i] * 2 + is_shift[i];
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

static void get_grid_address(int grid[3],
			    const int address_double[3],
			    const int mesh[3])
{
  int i;

  for (i = 0; i < 3; i++) {
    if (address_double[i] % 2 == 0) {
      grid[i] = address_double[i] / 2;
    } else {
      grid[i] = (address_double[i] - 1) / 2;
    }

#ifndef GRID_BOUNDARY_AS_NEGATIVE
    grid[i] = grid[i] - mesh[i] * (grid[i] > mesh[i] / 2);
#else
    grid[i] = grid[i] - mesh[i] * (grid[i] >= mesh[i] / 2);
#endif
  }  
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

static int get_BZ_triplets_at_q(int triplets[][3],
				const int grid_point,
				SPGCONST int bz_grid_address[][3],
				const int bz_map[],
				const int map_triplets[],
				const int num_map_triplets,
				const int mesh[3])
{
  int i, j, k, num_ir;
  int bz_address[3][3], bz_address_double[3], bzmesh[3], bzmesh_double[3];
  int *ir_grid_points;

  for (i = 0; i < 3; i++) {
    bzmesh[i] = mesh[i] * 2;
    bzmesh_double[i] = bzmesh[i] * 2;
  }

  num_ir = 0;
  ir_grid_points = (int*) malloc(sizeof(int) * num_map_triplets);
  for (i = 0; i < num_map_triplets; i++) {
    if (map_triplets[i] == i) {
      ir_grid_points[num_ir] = i;
      num_ir++;
    }
  }
 
#pragma omp parallel for private(j, k, bz_address, bz_address_double)
  for (i = 0; i < num_ir; i++) {
    for (j = 0; j < 3; j++) {
      bz_address[0][j] = bz_grid_address[grid_point][j];
      bz_address[1][j] = bz_grid_address[ir_grid_points[i]][j];
      bz_address[2][j] = - bz_address[0][j] - bz_address[1][j];
    }
    for (j = 2; j > -1; j--) {
      if (get_third_q_of_triplets_at_q(bz_address,
    				       j,
    				       bz_map,
    				       mesh,
    				       bzmesh,
    				       bzmesh_double) == 0) {
    	break;
      }
    }
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	bz_address_double[k] = bz_address[j][k] * 2;
      }
      get_vector_modulo(bz_address_double, bzmesh_double);
      triplets[i][j] =
	bz_map[get_grid_point_double_mesh(bz_address_double, bzmesh)];
    }
  }

  free(ir_grid_points);
  
  return num_ir;
}

static int get_third_q_of_triplets_at_q(int bz_address[3][3],
					const int q_index,
					const int bz_map[],
					const int mesh[3],
					const int bzmesh[3],
					const int bzmesh_double[3])
{
  int i, j, smallest_g, smallest_index, sum_g, delta_g[3];
  int bzgp[NUM_DIM_SEARCH], bz_address_double[3];

  get_vector_modulo(bz_address[q_index], mesh);
  for (i = 0; i < 3; i++) {
    delta_g[i] = 0;
    for (j = 0; j < 3; j++) {
      delta_g[i] += bz_address[j][i];
    }
    delta_g[i] /= mesh[i];
  }
  
  for (i = 0; i < NUM_DIM_SEARCH; i++) {
    for (j = 0; j < 3; j++) {
      bz_address_double[j] = (bz_address[q_index][j] +
			   search_space[i][j] * mesh[j]) * 2;
    }
    for (j = 0; j < 3; j++) {
      if (bz_address_double[j] < 0) {
	bz_address_double[j] += bzmesh_double[j];
      }
    }
    get_vector_modulo(bz_address_double, bzmesh_double);
    bzgp[i] = bz_map[get_grid_point_double_mesh(bz_address_double, bzmesh)];
  }

  for (i = 0; i < NUM_DIM_SEARCH; i++) {
    if (bzgp[i] != -1) {
      goto escape;
    }
  }
  warning_print("******* Warning *******\n");
  warning_print(" No third-q was found.\n");
  warning_print("******* Warning *******\n");

 escape:

  smallest_g = 4;
  smallest_index = 0;

  for (i = 0; i < NUM_DIM_SEARCH; i++) {
    if (bzgp[i] > -1) { /* q'' is in BZ */
      sum_g = (abs(delta_g[0] + search_space[i][0]) +
	       abs(delta_g[1] + search_space[i][1]) +
	       abs(delta_g[2] + search_space[i][2]));
      if (sum_g < smallest_g) {
	smallest_index = i;
	smallest_g = sum_g;
      }
    }
  }

  for (i = 0; i < 3; i++) {
    bz_address[q_index][i] += search_space[smallest_index][i] * mesh[i];
  }

  return smallest_g;
}

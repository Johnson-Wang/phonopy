/* spglib.c */
/* Copyright (C) 2008 Atsushi Togo */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spglib.h"
#include "refinement.h"
#include "cell.h"
#include "lattice.h"
#include "mathfunc.h"
#include "pointgroup.h"
#include "primitive.h"
#include "spacegroup.h"
#include "symmetry.h"
#include "kpoint.h"
#include "spin.h"
#include "tetrahedron_method.h"

static SpglibDataset * get_dataset(SPGCONST double lattice[3][3],
				   SPGCONST double position[][3],
				   const int types[],
				   const int num_atom,
				   const double symprec);

static int get_symmetry(int rotation[][3][3],
			double translation[][3],
			const int max_size,
			SPGCONST double lattice[3][3],
			SPGCONST double position[][3],
			const int types[],
			const int num_atom,
			const double symprec);

static int get_symmetry_from_dataset(int rotation[][3][3],
				     double translation[][3],
				     const int max_size,
				     SPGCONST double lattice[3][3],
				     SPGCONST double position[][3],
				     const int types[],
				     const int num_atom,
				     const double symprec);

static int get_symmetry_with_collinear_spin(int rotation[][3][3],
					    double translation[][3],
					    const int max_size,
					    SPGCONST double lattice[3][3],
					    SPGCONST double position[][3],
					    const int types[],
					    const double spins[],
					    const int num_atom,
					    const double symprec);

static int get_multiplicity(SPGCONST double lattice[3][3],
			    SPGCONST double position[][3],
			    const int types[],
			    const int num_atom,
			    const double symprec);

static int find_primitive(double lattice[3][3],
			  double position[][3],
			  int types[],
			  const int num_atom,
			  const double symprec);

static int get_international(char symbol[11],
			     SPGCONST double lattice[3][3],
			     SPGCONST double position[][3],
			     const int types[],
			     const int num_atom,
			     const double symprec);

static int get_schoenflies(char symbol[10],
			   SPGCONST double lattice[3][3],
			   SPGCONST double position[][3],
			   const int types[], const int num_atom,
			   const double symprec);

static int refine_cell(double lattice[3][3],
		       double position[][3],
		       int types[],
		       const int num_atom,
		       const double symprec);

static int get_ir_kpoints(int map[],
			  SPGCONST double kpoints[][3],
			  const int num_kpoint,
			  SPGCONST double lattice[3][3],
			  SPGCONST double position[][3],
			  const int types[],
			  const int num_atom,
			  const int is_time_reversal,
			  const double symprec);

static int get_ir_reciprocal_mesh(int grid_point[][3],
				  int map[],
				  const int mesh[3],
				  const int is_shift[3],
				  const int is_time_reversal,
				  SPGCONST double lattice[3][3],
				  SPGCONST double position[][3],
				  const int types[],
				  const int num_atom,
				  const double symprec);

static int get_stabilized_reciprocal_mesh(int grid_point[][3],
					  int map[],
					  int rot_map[],
					  const int mesh[3],
					  const int is_shift[3],
					  const int is_time_reversal,
					  const int num_rot,
					  SPGCONST int rotations[][3][3],
					  const int num_q,
					  SPGCONST double qpoints[][3]);
static int get_triplets_reciprocal_mesh_at_q(int weights[],
					     int grid_points[][3],
					     int third_q[],
					     int map_q[],
					     int rot_map_q[],
					     const int grid_point,
					     const int mesh[3],
					     const int is_time_reversal,
					     const int num_rot,
					     SPGCONST int rotations[][3][3]);
static int get_kpoint_group_at_q(int reverse_rotations[][3][3],
			      const double kpoints[][3],
			      const int mesh[3],
			      const int is_time_reversal,
			      const int num_rot,
			      const int num_q,
			      SPGCONST int rotations[][3][3]);

SpglibDataset * spg_get_dataset(SPGCONST double lattice[3][3],
				SPGCONST double position[][3],
				const int types[],
				const int num_atom,
				const double symprec)
{
  sym_set_angle_tolerance(-1.0);

  return get_dataset(lattice,
		     position,
		     types,
		     num_atom,
		     symprec);
}

SpglibDataset * spgat_get_dataset(SPGCONST double lattice[3][3],
				  SPGCONST double position[][3],
				  const int types[],
				  const int num_atom,
				  const double symprec,
				  const double angle_tolerance)
{
  sym_set_angle_tolerance(angle_tolerance);

  return get_dataset(lattice,
		     position,
		     types,
		     num_atom,
		     symprec);
}

void spg_free_dataset(SpglibDataset *dataset)
{
  if (dataset->n_operations > 0) {
    free(dataset->rotations);
    dataset->rotations = NULL;
    free(dataset->translations);
    dataset->translations = NULL;
  }
  
  if (! (dataset->wyckoffs == NULL)) {
    free(dataset->wyckoffs);
    dataset->wyckoffs = NULL;
  }
  
  if (! (dataset->equivalent_atoms == NULL)) {
    free(dataset->equivalent_atoms);
    dataset->equivalent_atoms = NULL;
  }

  free(dataset);
  dataset = NULL;
}

int spg_get_symmetry(int rotation[][3][3],
		     double translation[][3],
		     const int max_size,
		     SPGCONST double lattice[3][3],
		     SPGCONST double position[][3],
		     const int types[],
		     const int num_atom,
		     const double symprec)
{
  sym_set_angle_tolerance(-1.0);

  return get_symmetry_from_dataset(rotation,
				   translation,
				   max_size,
				   lattice,
				   position,
				   types,
				   num_atom,
				   symprec);
}

int spgat_get_symmetry(int rotation[][3][3],
		       double translation[][3],
		       const int max_size,
		       SPGCONST double lattice[3][3],
		       SPGCONST double position[][3],
		       const int types[],
		       const int num_atom,
		       const double symprec,
		       const double angle_tolerance)
{
  sym_set_angle_tolerance(angle_tolerance);

  return get_symmetry_from_dataset(rotation,
				   translation,
				   max_size,
				   lattice,
				   position,
				   types,
				   num_atom,
				   symprec);
}

int spg_get_symmetry_with_collinear_spin(int rotation[][3][3],
					 double translation[][3],
					 const int max_size,
					 SPGCONST double lattice[3][3],
					 SPGCONST double position[][3],
					 const int types[],
					 const double spins[],
					 const int num_atom,
					 const double symprec)
{
  sym_set_angle_tolerance(-1.0);

  return get_symmetry_with_collinear_spin(rotation,
					  translation,
					  max_size,
					  lattice,
					  position,
					  types,
					  spins,
					  num_atom,
					  symprec);
}

int spgat_get_symmetry_with_collinear_spin(int rotation[][3][3],
					   double translation[][3],
					   const int max_size,
					   SPGCONST double lattice[3][3],
					   SPGCONST double position[][3],
					   const int types[],
					   const double spins[],
					   const int num_atom,
					   const double symprec,
					   const double angle_tolerance)
{
  sym_set_angle_tolerance(angle_tolerance);

  return get_symmetry_with_collinear_spin(rotation,
					  translation,
					  max_size,
					  lattice,
					  position,
					  types,
					  spins,
					  num_atom,
					  symprec);
}

int spg_get_multiplicity(SPGCONST double lattice[3][3],
			 SPGCONST double position[][3],
			 const int types[],
			 const int num_atom,
			 const double symprec)
{
  sym_set_angle_tolerance(-1.0);

  return get_multiplicity(lattice,
			  position,
			  types,
			  num_atom,
			  symprec);
}

int spgat_get_multiplicity(SPGCONST double lattice[3][3],
			   SPGCONST double position[][3],
			   const int types[],
			   const int num_atom,
			   const double symprec,
			   const double angle_tolerance)
{
  sym_set_angle_tolerance(angle_tolerance);

  return get_multiplicity(lattice,
			  position,
			  types,
			  num_atom,
			  symprec);
}

int spg_get_smallest_lattice(double smallest_lattice[3][3],
			     SPGCONST double lattice[3][3],
			     const double symprec)
{
  return lat_smallest_lattice_vector(smallest_lattice, lattice, symprec);
}

int spg_find_primitive(double lattice[3][3],
		       double position[][3],
		       int types[],
		       const int num_atom,
		       const double symprec)
{
  sym_set_angle_tolerance(-1.0);

  return find_primitive(lattice,
			position,
			types,
			num_atom,
			symprec);
}

int spgat_find_primitive(double lattice[3][3],
			 double position[][3],
			 int types[],
			 const int num_atom,
			 const double symprec,
			 const double angle_tolerance)
{
  sym_set_angle_tolerance(angle_tolerance);

  return find_primitive(lattice,
			position,
			types,
			num_atom,
			symprec);
}

int spg_get_international(char symbol[11],
			  SPGCONST double lattice[3][3],
			  SPGCONST double position[][3],
			  const int types[],
			  const int num_atom,
			  const double symprec)
{
  sym_set_angle_tolerance(-1.0);

  return get_international(symbol,
			   lattice,
			   position,
			   types,
			   num_atom,
			   symprec);
}

int spgat_get_international(char symbol[11],
			    SPGCONST double lattice[3][3],
			    SPGCONST double position[][3],
			    const int types[],
			    const int num_atom,
			    const double symprec,
			    const double angle_tolerance)
{
  sym_set_angle_tolerance(angle_tolerance);

  return get_international(symbol,
			   lattice,
			   position,
			   types,
			   num_atom,
			   symprec);
}

int spg_get_schoenflies(char symbol[10],
			SPGCONST double lattice[3][3],
			SPGCONST double position[][3],
			const int types[],
			const int num_atom,
			const double symprec)
{
  sym_set_angle_tolerance(-1.0);

  return get_schoenflies(symbol,
			 lattice,
			 position,
			 types,
			 num_atom,
			 symprec);
}

int spgat_get_schoenflies(char symbol[10],
			  SPGCONST double lattice[3][3],
			  SPGCONST double position[][3],
			  const int types[],
			  const int num_atom,
			  const double symprec,
			  const double angle_tolerance)
{
  sym_set_angle_tolerance(angle_tolerance);

  return get_schoenflies(symbol,
			 lattice,
			 position,
			 types,
			 num_atom,
			 symprec);
}

int spg_get_pointgroup(char symbol[6],
		       int trans_mat[3][3],
		       SPGCONST int rotations[][3][3],
		       const int num_rotations)
{
  int ptg_num;
  double tmp_trans_mat[3][3];
  Pointgroup ptgroup;

  ptg_num = ptg_get_pointgroup_number_by_rotations(rotations,
						   num_rotations);
  ptgroup = ptg_get_pointgroup(ptg_num);
  strcpy(symbol, ptgroup.symbol);
  ptg_get_transformation_matrix(tmp_trans_mat,
				rotations,
				num_rotations);
  mat_cast_matrix_3d_to_3i(trans_mat, tmp_trans_mat);
  return ptg_num + 1;
}

int spg_refine_cell(double lattice[3][3],
		    double position[][3],
		    int types[],
		    const int num_atom,
		    const double symprec)
{
  sym_set_angle_tolerance(-1.0);

  return refine_cell(lattice,
		     position,
		     types,
		     num_atom,
		     symprec);
}

int spgat_refine_cell(double lattice[3][3],
		      double position[][3],
		      int types[],
		      const int num_atom,
		      const double symprec,
		      const double angle_tolerance)
{
  sym_set_angle_tolerance(angle_tolerance);

  return refine_cell(lattice,
		     position,
		     types,
		     num_atom,
		     symprec);
}

int spg_get_ir_kpoints(int map[],
		       SPGCONST double kpoints[][3],
		       const int num_kpoint,
		       SPGCONST double lattice[3][3],
		       SPGCONST double position[][3],
		       const int types[],
		       const int num_atom,
		       const int is_time_reversal,
		       const double symprec)
{
  sym_set_angle_tolerance(-1.0);

  return get_ir_kpoints(map,
			kpoints,
			num_kpoint,
			lattice,
			position,
			types,
			num_atom,
			is_time_reversal,
			symprec);
}

int spg_get_ir_reciprocal_mesh(int grid_point[][3],
			       int map[],
			       const int mesh[3],
			       const int is_shift[3],
			       const int is_time_reversal,
			       SPGCONST double lattice[3][3],
			       SPGCONST double position[][3],
			       const int types[],
			       const int num_atom,
			       const double symprec)
{
  sym_set_angle_tolerance(-1.0);

  return get_ir_reciprocal_mesh(grid_point,
				map,
				mesh,
				is_shift,
				is_time_reversal,
				lattice,
				position,
				types,
				num_atom,
				symprec);
}

int spg_get_stabilized_reciprocal_mesh(int grid_point[][3],
				       int map[],
				       int rot_map[],
				       const int mesh[3],
				       const int is_shift[3],
				       const int is_time_reversal,
				       const int num_rot,
				       SPGCONST int rotations[][3][3],
				       const int num_q,
				       SPGCONST double qpoints[][3])
{
  return get_stabilized_reciprocal_mesh(grid_point,
					map,
					rot_map,
					mesh,
					is_shift,
					is_time_reversal,
					num_rot,
					rotations,
					num_q,
					qpoints);
}

int spg_get_unique_tetrahedra(int *unique_vertices,
			      const int *bz_grid_address,
			      const int *bz_map, 
			      const int *relative_address,
			      const int *mesh, 
			      const int num_grid, 
			      const int dim0, 
			      const int dim1)
{
  return kpt_get_unique_tetrahedra(unique_vertices,
				   bz_grid_address,
				   bz_map,
				   relative_address,
				   mesh,
				   num_grid, 
				   dim0,
				   dim1);
}

int spg_relocate_BZ_grid_address(int bz_grid_address[][3],
				 int bz_map[],
				 int bz_map_to_pp[],
				 SPGCONST int grid_address[][3],
				 const int mesh[3],
				 SPGCONST double rec_lattice[3][3],
				 const int is_shift[3])
{
  return kpt_relocate_BZ_grid_address(bz_grid_address,
				      bz_map,
				      bz_map_to_pp,
				      grid_address,
				      mesh,
				      rec_lattice,
				      is_shift);
}

int spg_get_triplets_reciprocal_mesh_at_q(int weights[],
					  int grid_points[][3],
					  int third_q[],
					  int map_q[],
					  int rot_map_q[],
					  const int grid_point,
					  const int mesh[3],
					  const int is_time_reversal,
					  const int num_rot,
					  SPGCONST int rotations[][3][3])
{
  return get_triplets_reciprocal_mesh_at_q(weights,
					   grid_points,
					   third_q,
					   map_q,
					   rot_map_q,
					   grid_point,
					   mesh,
					   is_time_reversal,
					   num_rot,
					   rotations);
}

int spg_get_kpoint_group_at_q(int reverse_rotations[][3][3],
			      const double kpoints[][3],
			      const int mesh[3],
			      const int is_time_reversal,
			      const int num_rot,
			      const int num_q,
			      SPGCONST int rotations[][3][3])
{
  return get_kpoint_group_at_q(reverse_rotations,
			       kpoints,
			       mesh,
			       is_time_reversal,
			       num_rot,
			       num_q,
			       rotations);
}

int  spg_reduce_triplets_permute_sym(int triplet_mappings[],
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
  return reduce_triplets_permute_sym(triplet_mappings,
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
}

int  spg_reduce_pairs_permute_sym(int pair_mappings[],
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
  return reduce_pairs_permute_sym(pair_mappings,
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
}

void spg_set_grid_triplets_at_q(int triplets[][3],
				const int q_grid_point,
				SPGCONST int grid_points[][3],
				const int third_q[],
				const int weights[],
				const int mesh[3])

{
  kpt_set_grid_triplets_at_q(triplets,
			     q_grid_point,
			     grid_points,
			     third_q,
			     weights,
			     mesh);
}

// void spg_get_rotations_at_stars(int num_rots[], 
// 				int rotations_at_stars[][3][3],
// 				SPGCONST kpoint_symmetry[][3][3],
// 				const int grid_points[],
// 				const int grid_address[][3],
// 				const int num_grid)
// {
//   
// }

static SpglibDataset * get_dataset(SPGCONST double lattice[3][3],
				   SPGCONST double position[][3],
				   const int types[],
				   const int num_atom,
				   const double symprec)
{
  int i, j;
  int *mapping_table, *wyckoffs, *equiv_atoms, *equiv_atoms_prim;
  double tolerance;
  Spacegroup spacegroup;
  SpglibDataset *dataset;
  Cell *cell, *primitive;
  double inv_mat[3][3];
  Symmetry *symmetry;

  dataset = (SpglibDataset*) malloc(sizeof(SpglibDataset));

  cell = cel_alloc_cell(num_atom);
  cel_set_cell(cell, lattice, position, types);

  mapping_table = (int*) malloc(sizeof(int) * cell->size);
  primitive = prm_get_primitive_with_mapping_table(mapping_table,
						   cell,
						   symprec);
  tolerance = prm_get_current_tolerance();
  spacegroup = spa_get_spacegroup_with_primitive(primitive, tolerance);

  if (spacegroup.number > 0) {
    /* Spacegroup type, transformation matrix, origin shift */
    dataset->spacegroup_number = spacegroup.number;
    dataset->hall_number = spacegroup.hall_number;
    strcpy(dataset->international_symbol, spacegroup.international_short);
    strcpy(dataset->hall_symbol, spacegroup.hall_symbol);
    mat_inverse_matrix_d3(inv_mat, lattice, tolerance);
    mat_multiply_matrix_d3(dataset->transformation_matrix,
			   inv_mat,
			   spacegroup.bravais_lattice);
    mat_copy_vector_d3(dataset->origin_shift, spacegroup.origin_shift);

    /* Wyckoff positions */
    wyckoffs = (int*) malloc(sizeof(int) * primitive->size);
    equiv_atoms_prim = (int*) malloc(sizeof(int) * primitive->size);
    for (i = 0; i < primitive->size; i++) {
      wyckoffs[i] = -1;
      equiv_atoms_prim[i] = -1;
    }
    ref_get_Wyckoff_positions(wyckoffs, 
			      equiv_atoms_prim,
			      primitive,
			      &spacegroup,
			      tolerance);
    dataset->n_atoms = cell->size;
    dataset->wyckoffs = (int*) malloc(sizeof(int) * cell->size); 
    for (i = 0; i < cell->size; i++) {
      dataset->wyckoffs[i] = wyckoffs[mapping_table[i]];
    }
    
    free(wyckoffs);
    wyckoffs = NULL;

    /* Equivalent atoms */
    dataset->equivalent_atoms = (int*) malloc(sizeof(int) * cell->size);
    equiv_atoms = (int*) malloc(sizeof(int) * primitive->size);
    for (i = 0; i < primitive->size; i++) {
      for (j = 0; j < cell->size; j++) {
	if (mapping_table[j] == equiv_atoms_prim[i]) {
	  equiv_atoms[i] = j;
	  break;
	}
      }
    }
    for (i = 0; i < cell->size; i++) {
      dataset->equivalent_atoms[i] = equiv_atoms[mapping_table[i]];
    }
    free(equiv_atoms);
    equiv_atoms = NULL;

    free(equiv_atoms_prim);
    equiv_atoms_prim = NULL;

    /* Symmetry operations */
    symmetry = ref_get_refined_symmetry_operations(cell,
						   primitive,
						   &spacegroup,
						   tolerance);
    dataset->rotations = (int (*)[3][3]) malloc(sizeof(int[3][3]) * symmetry->size);
    dataset->translations = (double (*)[3]) malloc(sizeof(double[3]) * symmetry->size);
    dataset->n_operations = symmetry->size;
    for (i = 0; i < symmetry->size; i++) {
      mat_copy_matrix_i3(dataset->rotations[i], symmetry->rot[i]);
      mat_copy_vector_d3(dataset->translations[i], symmetry->trans[i]);
    }
    sym_free_symmetry(symmetry);

  } else {
    dataset->spacegroup_number = 0;
  }

  free(mapping_table);
  mapping_table = NULL;

  cel_free_cell(primitive);

  if (dataset->spacegroup_number == 0) {
    strcpy(dataset->international_symbol, "");
    strcpy(dataset->hall_symbol, "");
    dataset->origin_shift[0] = 0;
    dataset->origin_shift[1] = 0;
    dataset->origin_shift[2] = 0;
    dataset->n_atoms = 0;
    dataset->wyckoffs = NULL;
    dataset->equivalent_atoms = NULL;
    dataset->n_operations = 0;
    dataset->rotations = NULL;
    dataset->translations = NULL;
  }
  
  cel_free_cell(cell);
  return dataset;
}

static int get_symmetry(int rotation[][3][3],
			double translation[][3],
			const int max_size,
			SPGCONST double lattice[3][3],
			SPGCONST double position[][3],
			const int types[],
			const int num_atom,
			const double symprec)
{
  int i, j, size;
  Symmetry *symmetry;
  Cell *cell;

  cell = cel_alloc_cell(num_atom);
  cel_set_cell(cell, lattice, position, types);
  symmetry = sym_get_operation(cell, symprec);

  if (symmetry->size > max_size) {
    fprintf(stderr, "spglib: Indicated max size(=%d) is less than number ", max_size);
    fprintf(stderr, "spglib: of symmetry operations(=%d).\n", symmetry->size);
    sym_free_symmetry(symmetry);
    return 0;
  }

  for (i = 0; i < symmetry->size; i++) {
    mat_copy_matrix_i3(rotation[i], symmetry->rot[i]);
    for (j = 0; j < 3; j++) {
      translation[i][j] = symmetry->trans[i][j];
    }
  }

  size = symmetry->size;

  cel_free_cell(cell);
  sym_free_symmetry(symmetry);

  return size;
}

static int get_symmetry_from_dataset(int rotation[][3][3],
				     double translation[][3],
				     const int max_size,
				     SPGCONST double lattice[3][3],
				     SPGCONST double position[][3],
				     const int types[],
				     const int num_atom,
				     const double symprec)
{
  int i, j, k, num_sym;
  SpglibDataset *dataset;

  dataset = get_dataset(lattice,
			position,
			types,
			num_atom,
			symprec);
  
  if (dataset->n_operations > max_size) {
    fprintf(stderr,
	    "spglib: Indicated max size(=%d) is less than number ", max_size);
    fprintf(stderr,
	    "spglib: of symmetry operations(=%d).\n", dataset->n_operations);
    num_sym = 0;
    goto ret;
  }

  num_sym = dataset->n_operations;
  for (i = 0; i < num_sym; i++) {
    for (j = 0; j < 3; j++) {
      translation[i][j] = dataset->translations[i][j];
      for (k = 0; k < 3; k++) {
	rotation[i][j][k] = dataset->rotations[i][j][k];
      }
    }	  
  }
  
 ret:
  spg_free_dataset(dataset);
  return num_sym;
}

static int get_symmetry_with_collinear_spin(int rotation[][3][3],
					    double translation[][3],
					    const int max_size,
					    SPGCONST double lattice[3][3],
					    SPGCONST double position[][3],
					    const int types[],
					    const double spins[],
					    const int num_atom,
					    const double symprec)
{
  int i, j, size;
  Symmetry *symmetry;
  Cell *cell;

  cell = cel_alloc_cell(num_atom);
  cel_set_cell(cell, lattice, position, types);
  symmetry = spn_get_collinear_operation(cell, spins, symprec);
  
  if (symmetry->size > max_size) {
    fprintf(stderr, "spglib: Indicated max size(=%d) is less than number ", max_size);
    fprintf(stderr, "spglib: of symmetry operations(=%d).\n", symmetry->size);
    sym_free_symmetry(symmetry);
    return 0;
  }

  for (i = 0; i < symmetry->size; i++) {
    mat_copy_matrix_i3(rotation[i], symmetry->rot[i]);
    for (j = 0; j < 3; j++) {
      translation[i][j] = symmetry->trans[i][j];
    }
  }

  size = symmetry->size;

  cel_free_cell(cell);
  sym_free_symmetry(symmetry);

  return size;
}

static int get_multiplicity(SPGCONST double lattice[3][3],
			    SPGCONST double position[][3],
			    const int types[],
			    const int num_atom,
			    const double symprec)
{
  Symmetry *symmetry;
  Cell *cell;
  int size;

  cell = cel_alloc_cell(num_atom);
  cel_set_cell(cell, lattice, position, types);
  symmetry = sym_get_operation(cell, symprec);

  size = symmetry->size;

  cel_free_cell(cell);
  sym_free_symmetry(symmetry);

  return size;
}

static int find_primitive(double lattice[3][3],
			  double position[][3],
			  int types[],
			  const int num_atom,
			  const double symprec)
{
  int i, j, num_prim_atom=0;
  Cell *cell, *primitive;

  cell = cel_alloc_cell(num_atom);
  cel_set_cell(cell, lattice, position, types);

  /* find primitive cell */
  primitive = prm_get_primitive(cell, symprec);
  if (primitive->size == cell->size) { /* Already primitive */
    num_prim_atom = 0;
  } else { /* Primitive cell was found. */
    num_prim_atom = primitive->size;
    if (num_prim_atom < num_atom && num_prim_atom > 0 ) {
      mat_copy_matrix_d3(lattice, primitive->lattice);
      for (i = 0; i < primitive->size; i++) {
	types[i] = primitive->types[i];
	for (j=0; j<3; j++) {
	  position[i][j] = primitive->position[i][j];
	}
      }
    }
  }

  cel_free_cell(primitive);
  cel_free_cell(cell);
    
  return num_prim_atom;
}

static int get_international(char symbol[11],
			     SPGCONST double lattice[3][3],
			     SPGCONST double position[][3],
			     const int types[],
			     const int num_atom,
			     const double symprec)
{
  Cell *cell;
  Spacegroup spacegroup;

  cell = cel_alloc_cell(num_atom);
  cel_set_cell(cell, lattice, position, types);
  spacegroup = spa_get_spacegroup(cell, symprec);
  if (spacegroup.number > 0) {
    strcpy(symbol, spacegroup.international_short);
  }

  cel_free_cell(cell);
  
  return spacegroup.number;
}

static int get_schoenflies(char symbol[10],
			   SPGCONST double lattice[3][3],
			   SPGCONST double position[][3],
			   const int types[],
			   const int num_atom,
			   const double symprec)
{
  Cell *cell;
  Spacegroup spacegroup;

  cell = cel_alloc_cell(num_atom);
  cel_set_cell(cell, lattice, position, types);

  spacegroup = spa_get_spacegroup(cell, symprec);
  if (spacegroup.number > 0) {
    strcpy(symbol, spacegroup.schoenflies);
  }

  cel_free_cell(cell);

  return spacegroup.number;
}

static int refine_cell(double lattice[3][3],
		       double position[][3],
		       int types[],
		       const int num_atom,
		       const double symprec)
{
  int i, num_atom_bravais;
  Cell *cell, *bravais;

  cell = cel_alloc_cell(num_atom);
  cel_set_cell(cell, lattice, position, types);

  bravais = ref_refine_cell(cell, symprec);
  cel_free_cell(cell);

  if (bravais->size > 0) {
    mat_copy_matrix_d3(lattice, bravais->lattice);
    num_atom_bravais = bravais->size;
    for (i = 0; i < bravais->size; i++) {
      types[i] = bravais->types[i];
      mat_copy_vector_d3(position[i], bravais->position[i]);
    }
  } else {
    num_atom_bravais = 0;
  }

  cel_free_cell(bravais);
  
  return num_atom_bravais;
}

static int get_ir_kpoints(int map[],
			  SPGCONST double kpoints[][3],
			  const int num_kpoint,
			  SPGCONST double lattice[3][3],
			  SPGCONST double position[][3],
			  const int types[],
			  const int num_atom,
			  const int is_time_reversal,
			  const double symprec)
{
  Symmetry *symmetry;
  Cell *cell;
  int num_ir_kpoint;

  cell = cel_alloc_cell(num_atom);
  cel_set_cell(cell, lattice, position, types);
  symmetry = sym_get_operation(cell, symprec);

  num_ir_kpoint = kpt_get_irreducible_kpoints(map,
					      kpoints,
					      num_kpoint,
					      symmetry,
					      is_time_reversal,
					      symprec);


  cel_free_cell(cell);
  sym_free_symmetry(symmetry);

  return num_ir_kpoint;
}

static int get_ir_reciprocal_mesh(int grid_point[][3],
				  int map[],
				  const int mesh[3],
				  const int is_shift[3],
				  const int is_time_reversal,
				  SPGCONST double lattice[3][3],
				  SPGCONST double position[][3],
				  const int types[],
				  const int num_atom,
				  const double symprec)
{
  Symmetry *symmetry;
  Cell *cell;
  int num_ir;

  cell = cel_alloc_cell(num_atom);
  cel_set_cell(cell, lattice, position, types);
  symmetry = sym_get_operation(cell, symprec);

  num_ir = kpt_get_irreducible_reciprocal_mesh(grid_point,
					       map,
					       mesh,
					       is_shift,
					       is_time_reversal,
					       symmetry);


  cel_free_cell(cell);
  sym_free_symmetry(symmetry);

  return num_ir;
}

static int get_stabilized_reciprocal_mesh(int grid_point[][3],
					  int map[],
					  int rot_map[],
					  const int mesh[3],
					  const int is_shift[3],
					  const int is_time_reversal,
					  const int num_rot,
					  SPGCONST int rotations[][3][3],
					  const int num_q,
					  SPGCONST double qpoints[][3])
{
  MatINT *rot_real;
  int i, num_ir;
  
  rot_real = mat_alloc_MatINT(num_rot);
  for (i = 0; i < num_rot; i++) {
    mat_copy_matrix_i3(rot_real->mat[i], rotations[i]);
  }

  num_ir = kpt_get_stabilized_reciprocal_mesh(grid_point,
					      map,
					      rot_map,
					      mesh,
					      is_shift,
					      is_time_reversal,
					      rot_real,
					      num_q,
					      qpoints);

  mat_free_MatINT(rot_real);

  return num_ir;
}

static int get_triplets_reciprocal_mesh_at_q(int weights[],
					     int grid_points[][3],
					     int third_q[],
					     int map_q[],
					     int rot_map_q[],
					     const int grid_point,
					     const int mesh[3],
					     const int is_time_reversal,
					     const int num_rot,
					     SPGCONST int rotations[][3][3])
{
  MatINT *rot_real;
  int i, num_ir;
  rot_real = mat_alloc_MatINT(num_rot);
  for (i = 0; i < num_rot; i++) {
    mat_copy_matrix_i3(rot_real->mat[i], rotations[i]);
  }

  num_ir = kpt_get_ir_triplets_at_q(weights,
				    grid_points,
				    third_q,
				    map_q,
				    rot_map_q,
				    grid_point,
				    mesh,
				    is_time_reversal,
				    rot_real);

  mat_free_MatINT(rot_real);
  return num_ir;
}

static int get_kpoint_group_at_q(int reverse_rotations[][3][3],
			      const double kpoints[][3],
			      const int mesh[3],
			      const int is_time_reversal,
			      const int num_rot,
			      const int num_q,
			      SPGCONST int rotations[][3][3])
{
  MatINT *rot_real;
  int i, num_reci_rot;
  rot_real = mat_alloc_MatINT(num_rot);
  for (i = 0; i < num_rot; i++) {
    mat_copy_matrix_i3(rot_real->mat[i], rotations[i]);
  }

  num_reci_rot = kpt_get_kpoint_group_at_q(reverse_rotations,
				     kpoints,
				     mesh,
				     is_time_reversal,
				     num_rot,
				     num_q,
				     rot_real);

  mat_free_MatINT(rot_real);
  return num_reci_rot;
}

int spg_get_grid_points_by_rotations(int rot_grid_points[],
				     const int address_orig[3],
				     const int num_rot,
				     SPGCONST int rot_reciprocal[][3][3],
				     const int mesh[3],
				     const int is_shift[3])
{
  int i;
  MatINT *rot;

  rot = NULL;

  if ((rot = mat_alloc_MatINT(num_rot)) == NULL) {
    return 0;
  }

  for (i = 0; i < num_rot; i++) {
    mat_copy_matrix_i3(rot->mat[i], rot_reciprocal[i]);
  }
  kpt_get_grid_points_by_rotations(rot_grid_points,
				   address_orig,
				   rot,
				   mesh,
				   is_shift);
  mat_free_MatINT(rot);
  rot = NULL;

  return 1;
}

void spg_get_BZ_grid_points_by_rotations(int rot_grid_points[],
					 const int address_orig[3],
					 const int num_rot,
					 SPGCONST int rot_reciprocal[][3][3],
					 const int mesh[3],
					 const int is_shift[3],
					 const int bz_map[])
{
  int i;
  MatINT *rot;

  rot = mat_alloc_MatINT(num_rot);
  for (i = 0; i < num_rot; i++) {
    mat_copy_matrix_i3(rot->mat[i], rot_reciprocal[i]);
  }
  kpt_get_BZ_grid_points_by_rotations(rot_grid_points,
				      address_orig,
				      rot,
				      mesh,
				      is_shift,
				      bz_map);
  mat_free_MatINT(rot);
}

int spg_get_BZ_triplets_at_q(int triplets[][3],
			     const int grid_point,
			     SPGCONST int bz_grid_address[][3],
			     const int bz_map[],
			     const int map_triplets[],
			     const int num_map_triplets,
			     const int mesh[3])
{
  return kpt_get_BZ_triplets_at_q(triplets,
				  grid_point,
				  bz_grid_address,
				  bz_map,
				  map_triplets,
				  num_map_triplets,
				  mesh);
}
void spg_get_neighboring_grid_points(int relative_grid_points[],
				     const int grid_point,
				     SPGCONST int relative_grid_address[][3],
				     const int num_relative_grid_address,
				     const int mesh[3],
				     SPGCONST int bz_grid_address[][3],
				     const int bz_map[])
{
  kpt_get_neighboring_grid_points(relative_grid_points,
				  grid_point,
				  relative_grid_address,
				  num_relative_grid_address,
				  mesh,
				  bz_grid_address,
				  bz_map);
}

/*--------------------*/
/* tetrahedron method */
/*--------------------*/
void
spg_get_tetrahedra_relative_grid_address(int relative_grid_address[24][4][3],
					 SPGCONST double rec_lattice[3][3])
{
  thm_get_relative_grid_address(relative_grid_address, rec_lattice);
}

void
spg_get_all_tetrahedra_relative_grid_address
(int relative_grid_address[4][24][4][3])
{
  thm_get_all_relative_grid_address(relative_grid_address);
}

double
spg_get_tetrahedra_integration_weight(const double omega,
				      SPGCONST double tetrahedra_omegas[24][4],
				      const char function)
{
  return thm_get_integration_weight(omega,
				    tetrahedra_omegas,
				    function);
}

void
spg_get_tetrahedra_integration_weight_at_omegas
(double integration_weights[],
 const int num_omegas,
 const double omegas[],
 SPGCONST double tetrahedra_omegas[24][4],
 const char function)
{
  thm_get_integration_weight_at_omegas(integration_weights,
				       num_omegas,
				       omegas,
				       tetrahedra_omegas,
				       function);
}

double
spg_get_tetrahedra_integration_weight_1D(const double omega,
				      SPGCONST double tetrahedra_omegas[2][2],
				      const char function)
{
  return thm_get_integration_weight_1D(omega,
				    tetrahedra_omegas,
				    function);
}

void
spg_get_tetrahedra_integration_weight_at_omegas_1D
(double integration_weights[],
 const int num_omegas,
 const double omegas[],
 SPGCONST double tetrahedra_omegas[2][2],
 const char function)
{
  thm_get_integration_weight_at_omegas_1D(integration_weights,
				       num_omegas,
				       omegas,
				       tetrahedra_omegas,
				       function);
}

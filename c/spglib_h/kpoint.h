/* kpoints.h */
/* Copyright (C) 2008 Atsushi Togo */

#ifndef __kpoints_H__
#define __kpoints_H__

#include "symmetry.h"
#include "mathfunc.h"

int kpt_get_irreducible_kpoints(int map[],
				SPGCONST double kpoints[][3], 
				const int num_kpoint,
				const Symmetry * symmetry,
				const int is_time_reversal,
				const double symprec);
int kpt_get_irreducible_reciprocal_mesh(int grid_points[][3],
					int map[],
					const int mesh[3],
					const int is_shift[3],
					const int is_time_reversal,
					const Symmetry * symmetry);
int kpt_get_stabilized_reciprocal_mesh(int grid_points[][3],
				       int map[],
				       int rot_map[],
				       const int mesh[3],
				       const int is_shift[3],
				       const int is_time_reversal,
				       const MatINT * pointgroup_real,
				       const int num_q,
				       SPGCONST double qpoints[][3]);
int kpt_get_kpoint_group_at_q(int reverse_rotations[][3][3],
			      const double kpoints[][3],
			      const int mesh[3],
			      const int is_time_reversal,
			      const int num_rot,
			      const int num_q,
			      const MatINT * rotations);
int kpt_relocate_BZ_grid_address(int bz_grid_address[][3],
				 int bz_map[],
				 int bz_map_to_pp[],
				 SPGCONST int grid_address[][3],
				 const int mesh[3],
				 SPGCONST double rec_lattice[3][3],
				 const int is_shift[3]);
int kpt_get_unique_tetrahedra(int unique_vertices[],
			      const int bz_grid_address[],
			      const int bz_map[], 
			      const int relative_address[],
			      const int mesh[], 
			      const int num_grid, 
			      const int dim0, 
			      const int dim1);
void kpt_get_neighboring_grid_points(int neighboring_grid_points[],
				     const int grid_point,
				     SPGCONST int relative_grid_address[][3],
				     const int num_relative_grid_address,
				     const int mesh[3],
				     SPGCONST int bz_grid_address[][3],
				     const int bz_map[]);
void kpt_get_grid_points_by_rotations(int rot_grid_points[],
				      const int address_orig[3],
				      const MatINT * rot_reciprocal,
				      const int mesh[3],
				      const int is_shift[3]);
void kpt_get_BZ_grid_points_by_rotations(int rot_grid_points[],
					 const int address_orig[3],
					 const MatINT * rot_reciprocal,
					 const int mesh[3],
					 const int is_shift[3],
					 const int bz_map[]);
int kpt_get_BZ_triplets_at_q(int triplets[][3],
			     const int grid_point,
			     SPGCONST int bz_grid_address[][3],
			     const int bz_map[],
			     const int map_triplets[],
			     const int num_map_triplets,
			     const int mesh[3]);
int kpt_get_ir_triplets_at_q(int weights[],
			     int grid_points[][3],
			     int third_q[],
			     int map_q[],
			     int rot_map_q[],
			     const int grid_point,
			     const int mesh[3],
			     const int is_time_reversal,
			     const MatINT * rotations);
void kpt_set_grid_triplets_at_q(int triplets[][3],
				const int q_grid_point,
				SPGCONST int grid_points[][3],
				const int third_q[],
				const int weights[],
				const int mesh[3]);
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
				 const int num_grid_all);

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
				 const int num_grid_all);
#endif

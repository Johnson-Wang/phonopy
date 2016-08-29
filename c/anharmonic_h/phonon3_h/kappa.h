#ifndef __kappa_H__
#define __kappa_H__

void  get_collision_at_all_band_permute(double *collision, // actually collision_in
					      const double *interaction,
					      const double *occupation,
					      const double *frequency,
					      const int num_triplet,
					      const int num_band,
					      const double *g,
					      const double cutoff_frequency);

void  get_collision_at_all_band(double *collision, // actually collision_in
				const double *interaction,
				const double *frequency,
				const double *g,
				const int num_triplet,
				const double temperature,
				const int num_band,	
				const double cutoff_frequency);

void  get_next_perturbation_at_all_bands(double *summation,
					 const double *F_prev,
					 const double *scatt,
					 const int *convergence,
					 const int *q1s,
					 const int *spg_mapping_index,
					 const double *inv_rot_sum,
					 const int num_grids,
					 const int num_triplet,
					 const int num_band);

void collision_degeneracy(double *collision, // shape: [ntriplet, 3, nband, nband] or [ntriplet, nband, nband]
                        const int *degeneracy, // shape: [ngrids, nband]
                        const int (*triplets)[3],
                        const int num_triplet,
                        const int num_band,
                        const int is_permute);

void get_interaction_from_reduced(double *interaction, 
				  const double *interaction_all,
				  const int *triplet_mapping,
				  const char *triplet_sequence,
				  const int num_triplet,
				  const int num_band0,
				  const int num_band);

void get_collision_from_reduced(double *scatt, 
				      const double *scattall, 
				      const int *triplet_mapping,
				      const char *sequence,
				      const int num_triplet, 
				      const int num_band);

void get_kappa_at_grid_point(double kappa[],
			     Iarray* kpt_rotations_at_q,
			     const double rec_lat[],
			     const double gv[],
			     const double heat_capacity[],
			     const double mfp[],
			     const int deg[],
			     const int num_band,
			     const int num_temp);
void  get_kappa(double kappa[],
		const double F[], //F (equivalent)
		const double heat_capacity[],
		const double gv[],
		const double rec_lat[],
		const int index_mappings[],
		const int kpt_rotations[], 
		const int degeneracies[],
		const int num_grid,
		const int num_all_grids,
		const int num_band,
		const int num_temp);
#endif
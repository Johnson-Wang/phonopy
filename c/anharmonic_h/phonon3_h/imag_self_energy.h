#ifndef __imag_self_energy_H__
#define __imag_self_energy_H__

#include "phonoc_array.h"

void get_imag_self_energy(double *gamma,
			  const Darray *fc3_normal_sqared,
			  const double fpoint,
			  const double *frequencies,
			  const int *grid_point_triplets,
			  const int *triplet_weights,
			  const double *asigma,
			  const double temperature,
			  const double unit_conversion_factor,
                          const double cutoff_delta,
			  const double cutoff_frequency,
			  const double cutoff_gamma);

void get_imag_self_energy_at_bands(double *imag_self_energy,
				   const Darray *fc3_normal_sqared,
				   const int *band_indices,
				   const double *frequencies,
				   const int *grid_point_triplets,
				   const int *triplet_weights,
				   const double *asigma,
				   const double temperature,
				   const double unit_conversion_factor,
                                   const double cutoff_delta,
				   const double cutoff_frequency,
				   const double cutoff_hfrequency,
				   const double cutoff_gamma);
int get_jointDOS(double *jdos,
		 const int num_omega,
		 const int num_triplet,
		 const int num_band,
		 const double *o,
		 const double *f,
		 const int *w,
		 const double sigma);
int get_decay_channels(double *decay,
		       const int num_omega,
		       const int num_triplet,
		       const int num_band,
		       const double *o,
		       const double *f,
		       const double *fc3_normal_sqared,
		       const double sigma,
		       const double t,
		       const double unit_conversion_factor);

void get_thm_imag_self_energy_at_bands(double *imag_self_energy,
				       const Darray *fc3_normal_sqared,
				       const double *frequencies,
				       const int *grid_point_triplets,
				       const int *triplet_weights,
				       const double *g,
				       const int *band_indices,
				       const double temperature,
				       const double unit_conversion_factor,
				       const double cutoff_frequency);
#endif

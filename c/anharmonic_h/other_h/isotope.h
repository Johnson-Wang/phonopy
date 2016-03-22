#ifndef __isotope_H__
#define __isotope_H__

void get_isotope_scattering_strength(double *collision,
				     const int grid_point,
				     const double *mass_variances,
				     const double *frequencies,
				     const lapack_complex_double *eigenvectors,
				     const int num_grid_points,
				     const int *band_indices,
				     const double *occupations,
				     const int num_band,
				     const int num_band0,
				     const int num_t,
				     const double sigma,
				     const double cutoff_frequency);
void
get_thm_isotope_scattering_strength(double *collision,
				    const int grid_point,
				    const int *ir_grid_points,
				    const double *mass_variances,
				    const double *frequencies,
				    const lapack_complex_double *eigenvectors,
				    const int num_grid_points,
				    const int *band_indices,
				    const double *occupations, //occupation
				    const int num_band,
				    const int num_band0,
				    const double *integration_weights,
				    const double cutoff_frequency);
#endif

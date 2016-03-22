#include <stdlib.h>
#include<math.h>
#include <lapacke.h>
#include "phonoc_utils.h"

void get_isotope_scattering_strength(double *collision, //collision[temp, band0]
				     const int grid_point,
				     const double *mass_variances,
				     const double *frequencies,
				     const lapack_complex_double *eigenvectors,
				     const int num_grid_points,
				     const int *band_indices,
				     const double *occupations, //occupation
				     const int num_band,
				     const int num_band0,
				     const double sigma,
				     const double cutoff_frequency)
{
  int i, j, k, l;
  double *e0_r, *e0_i, e1_r, e1_i, a, b, f,n, *f0, *n0, sum_g_local, dist;
  int nbb = num_grid_points * num_band0 * num_band, bb = num_band0 * num_band;

  e0_r = (double*)malloc(sizeof(double) * num_band * num_band0);
  e0_i = (double*)malloc(sizeof(double) * num_band * num_band0);
  f0 = (double*)malloc(sizeof(double) * num_band0);
  n0 = (double*)malloc(sizeof(double) *num_band0);

  for (i = 0; i < num_band0; i++) {
    f0[i] = frequencies[grid_point * num_band + band_indices[i]];
    n0[i] =occupations[grid_point*num_band0+band_indices[i]]; 
    for (j = 0; j < num_band; j++) {
      e0_r[i * num_band + j] = lapack_complex_double_real
	(eigenvectors[grid_point * num_band * num_band +
		      j * num_band + band_indices[i]]);
      e0_i[i * num_band + j] = lapack_complex_double_imag
	(eigenvectors[grid_point * num_band * num_band +
		      j * num_band + band_indices[i]]);
    }
  }
  
  for (i = 0; i < num_grid_points * num_band0 * num_band; i++) {
    collision[i] = 0;
  }

  for (i = 0; i < num_band0; i++) { /* band index0 */
    if (f0[i] < cutoff_frequency) {
      continue;
    }
// #pragma omp parallel for private(k, l, f,n, e1_r, e1_i, a, b, sum_g_local, dist)
    for (j = 0; j < num_grid_points; j++) {
      for (k = 0; k < num_band; k++) { /* band index */
	sum_g_local = 0;
	f = frequencies[j * num_band + k];
	if (f < cutoff_frequency) {
	  continue;
	}
	dist = gaussian(f - f0[i], sigma);
	for (l = 0; l < num_band; l++) { /* elements */
	  e1_r = lapack_complex_double_real
	    (eigenvectors[j * num_band * num_band + l * num_band + k]);
	  e1_i = lapack_complex_double_imag
	    (eigenvectors[j * num_band * num_band + l * num_band + k]);
	  a = e0_r[i * num_band + l] * e1_r + e0_i[i * num_band + l] * e1_i;
	  b = e0_i[i * num_band + l] * e1_r - e0_r[i * num_band + l] * e1_i;
	  sum_g_local += (a * a + b * b) * mass_variances[l / 3];

	}
	n=occupations[j*num_band+k];
// 	#pragma omp atomic
	collision[i * bb + j * num_band + k] += sum_g_local * dist * f0[j] * f * (n0[j] * n + (n0[j] + n) / 2);
	
      }
    }
  }

  free(n0);
  free(f0);
  free(e0_r);
  free(e0_i);
}

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
				    const double cutoff_frequency)
{
  int i, j, k, l, m, gp;
  double *e0_r, *e0_i, *f0, *n0;
  double e1_r, e1_i, a, b, f, dist, sum_g_k, n;
  int nbb = num_grid_points * num_band0 * num_band, bb = num_band0 * num_band;

  e0_r = (double*)malloc(sizeof(double) * num_band * num_band0);
  e0_i = (double*)malloc(sizeof(double) * num_band * num_band0);
  f0 = (double*)malloc(sizeof(double) * num_band0);
  n0 = (double*)malloc(sizeof(double) *num_band0);

  for (i = 0; i < num_band0; i++) {
    f0[i] = frequencies[grid_point * num_band + band_indices[i]];
    n0[i] =occupations[grid_point *num_band0+band_indices[i]]; 
    for (j = 0; j < num_band; j++) {
      e0_r[i * num_band + j] = lapack_complex_double_real
	(eigenvectors[grid_point * num_band * num_band +
		      j * num_band + band_indices[i]]);
      e0_i[i * num_band + j] = lapack_complex_double_imag
	(eigenvectors[grid_point * num_band * num_band +
		      j * num_band + band_indices[i]]);
    }
  }
  
#pragma omp parallel for
  for (i = 0; i < num_grid_points * num_band0 * num_band; i++) {
    collision[i] = 0;
  }

#pragma omp parallel for private(j, k, l, m, f, gp, e1_r, e1_i, a, b, dist, sum_g_k, n)
  for (i = 0; i < num_grid_points; i++) {
    gp = ir_grid_points[i];
    for (j = 0; j < num_band0; j++) { /* band index0 */
      if (f0[j] < cutoff_frequency) {
	continue;
      }
      for (k = 0; k < num_band; k++) { /* band index */
	sum_g_k = 0;
	n = occupations[gp  *num_band + k];
	f = frequencies[gp * num_band + k];
	if (f < cutoff_frequency) {
	  continue;
	}
	dist = integration_weights[i * bb + j * num_band + k];
	for (l = 0; l < num_band / 3; l++) { /* elements */
	  a = 0;
	  b = 0;
	  for (m = 0; m < 3; m++) {
	    e1_r = lapack_complex_double_real
	      (eigenvectors
	       [gp * num_band * num_band + (l * 3 + m) * num_band + k]);
	    e1_i = lapack_complex_double_imag
	      (eigenvectors
	       [gp * num_band * num_band + (l * 3 + m) * num_band + k]);
	    a += (e0_r[j * num_band + l * 3 + m] * e1_r +
		  e0_i[j * num_band + l * 3 + m] * e1_i);
	    b += (e0_i[j * num_band + l * 3 + m] * e1_r -
		  e0_r[j * num_band + l * 3 + m] * e1_i);
	  }
	  sum_g_k += (a * a + b * b) * mass_variances[l] * dist;
	}
	collision[i * bb + j * num_band + k] += sum_g_k *f0[j] * f * (n0[j] * n + (n0[j] + n) / 2);
      }
      
    }
  }

  free(n0);
  free(f0);
  free(e0_r);
  free(e0_i);
}


#include <lapacke.h>thm_get_integration_weight
#include <math.h>
#include "phonoc_array.h"
#include "phonoc_math.h"
#include "phonon3_h/reciprocal_to_normal.h"

static double fc3_sum_squared(const int bi0,
			      const int bi1,
			      const int bi2,
			      const lapack_complex_double *eigvecs0,
			      const lapack_complex_double *eigvecs1,
			      const lapack_complex_double *eigvecs2,
			      const lapack_complex_double *fc3_reciprocal,
                              const int *atc_rec,
			      const double *masses,
			      const int num_atom);
void reciprocal_to_normal(double *fc3_normal_squared,
			  const lapack_complex_double *fc3_reciprocal,
			  const double *freqs0,
			  const double *freqs1,
			  const double *freqs2,
			  const lapack_complex_double *eigvecs0,
			  const lapack_complex_double *eigvecs1,
			  const lapack_complex_double *eigvecs2,
			  const double *masses,
			  const int *band_indices,
			  const int num_band0,
			  const int num_band,
			  const int pos_band0,
              const int *atc_rec, //inter-atomic triplets cut in the reciprocal force constants
              const char *g_skip,
			  const double cutoff_frequency,
			  const double cutoff_hfrequency,
              const double cutoff_delta)
{
  int i, j, k, bi, bj, bk, num_atom;
  double fff;
  int nbs[3][3] = {{num_band0, num_band, num_band},
                  {num_band, num_band0, num_band},
                  {num_band, num_band, num_band0}};
  int *nb = nbs[pos_band0];
  num_atom = num_band / 3;
  for (i=0; i< num_band0 * num_band * num_band; i++)
    fc3_normal_squared[i] = 0;
  for (i = 0; i < nb[0]; i++) {
    bi = (pos_band0 == 0? band_indices[i]: i);
    if (freqs0[bi] > cutoff_frequency && freqs0[bi] < cutoff_hfrequency) { // freqs0 limited in a range 
      for (j = 0; j < nb[1]; j++) {
        bj = (pos_band0 == 1? band_indices[j]: j);
        if (freqs1[bj] > cutoff_frequency) {
          for (k = 0; k < nb[2]; k++) {
            bk = (pos_band0 == 2? band_indices[k]: k);
            if (g_skip[i * nb[1] * nb[2] + j * nb[2] + k]) continue;
            if (freqs2[bk] > cutoff_frequency) {
              if (fabs(freqs0[bi] + freqs1[bj] - freqs2[bk]) > cutoff_delta &&
                  fabs(freqs0[bi] - freqs1[bj] + freqs2[bk]) > cutoff_delta &&
                  fabs(-freqs0[bi] + freqs1[bj] + freqs2[bk]) > cutoff_delta)
                    continue;
              fff = freqs0[bi] * freqs1[bj] * freqs2[bk];
              fc3_normal_squared[i * nb[1] * nb[2] +
                     j * nb[2] +
                     k] =
                fc3_sum_squared(bi, bj, bk,
                        eigvecs0, eigvecs1, eigvecs2,
                        fc3_reciprocal,
                        atc_rec,
                        masses,
                        num_atom) / fff;
            }
          }
        }
      }
    }
  }    
}

static double fc3_sum_squared(const int bi0,
			      const int bi1,
			      const int bi2,
			      const lapack_complex_double *eigvecs0,
			      const lapack_complex_double *eigvecs1,
			      const lapack_complex_double *eigvecs2,
			      const lapack_complex_double *fc3_reciprocal,
                  const int *atc_rec,
			      const double *masses,
			      const int num_atom)
{
  int i, j, k, l, m, n;
  double sum_real, sum_imag, sum_real_cart, sum_imag_cart, mmm;
  lapack_complex_double eig_prod;

  sum_real = 0;
  sum_imag = 0;
  for (i = 0; i < num_atom; i++) {
    for (j = 0; j < num_atom; j++) {
      for (k = 0; k < num_atom; k++) {
        if (atc_rec[i*num_atom*num_atom+j*num_atom+k]){
          continue;
        }
	sum_real_cart = 0;
	sum_imag_cart = 0;
	mmm = sqrt(masses[i] * masses[j] * masses[k]);
	for (l = 0; l < 3; l++) {
	  for (m = 0; m < 3; m++) {
	    for (n = 0; n < 3; n++) {
	      eig_prod =
		phonoc_complex_prod(eigvecs0[(i * 3 + l) * num_atom * 3 + bi0],
                phonoc_complex_prod(eigvecs1[(j * 3 + m) * num_atom * 3 + bi1],
		phonoc_complex_prod(eigvecs2[(k * 3 + n) * num_atom * 3 + bi2],
                fc3_reciprocal[i * num_atom * num_atom * 27 +
			       j * num_atom * 27 +
			       k * 27 +
			       l * 9 +
			       m * 3 +
			       n])));
	      sum_real_cart += lapack_complex_double_real(eig_prod);
	      sum_imag_cart += lapack_complex_double_imag(eig_prod);
	    }
	  }
	}
	sum_real += sum_real_cart / mmm;
	sum_imag += sum_imag_cart / mmm;
      }
    }
  }
  return sum_real * sum_real + sum_imag * sum_imag;
}

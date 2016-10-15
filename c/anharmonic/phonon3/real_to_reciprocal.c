#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <lapacke.h>
#include "phonoc_array.h"
#include "phonoc_math.h"
#include "phonoc_utils.h"
#include "phonon3_h/real_to_reciprocal.h"

static void real_to_reciprocal_elements(lapack_complex_double *fc3_rec_elem,
					const double q[9],
					const Darray *fc3,
					const int *atc,//atom triplet cut. True: not considered
					const Darray *shortest_vectors,
					const Iarray *multiplicity,
					const int *p2s,
					const int *s2p,
					const int pi0,
					const int pi1,
					const int pi2);

/* fc3_reciprocal[num_patom, num_patom, num_patom, 3, 3, 3] */
void real_to_reciprocal(lapack_complex_double *fc3_reciprocal,
			const double q[9],
			const Darray *fc3,
			const int *atc,
			const Darray *shortest_vectors,
			const Iarray *multiplicity,
			const int *p2s_map,
			const int *s2p_map)
{
  int i, j, k, num_patom,l;
  double pre_phase;
  lapack_complex_double pre_phase_factor;
  num_patom = multiplicity->dims[1];

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < num_patom; j++) {
      for (k = 0; k < num_patom; k++) {
	real_to_reciprocal_elements(fc3_reciprocal +
				    i * 27 * num_patom * num_patom +
				    j * 27 * num_patom +
				    k * 27,
				    q,
				    fc3,
				    atc,
				    shortest_vectors,
				    multiplicity,
				    p2s_map,
				    s2p_map,
				    i, j, k);
	
      }
    }

    pre_phase = 0;
    for (j = 0; j < 3; j++) {
      pre_phase += shortest_vectors->data
    	[p2s_map[i] * shortest_vectors->dims[1] *
    	 shortest_vectors->dims[2] * 3 + j] * (q[j] + q[3 + j] + q[6 + j]);
    }
    pre_phase_factor = lapack_make_complex_double(cos(M_2PI * pre_phase),
    						  sin(M_2PI * pre_phase));
    for (j = 0; j < num_patom * num_patom * 27; j++) {
      fc3_reciprocal[i * num_patom * num_patom * 27 + j] =
    	phonoc_complex_prod(fc3_reciprocal[i * num_patom * num_patom * 27 + j],
    			    pre_phase_factor);
    }
  }
}		       

static void real_to_reciprocal_elements(lapack_complex_double *fc3_rec_elem,
					const double q[9],
					const Darray *fc3,
					const int *atc,
					const Darray *shortest_vectors,
					const Iarray *multiplicity,
					const int *p2s,
					const int *s2p,
					const int pi0,
					const int pi1,
					const int pi2)
{
  int i, j, k, l, num_satom;
  lapack_complex_double phase_factor, phase_factor1, phase_factor2;
  double fc3_rec_real[27], fc3_rec_imag[27];
  double *fc3_elem;

  for (i = 0; i < 27; i++) {
    fc3_rec_real[i] = 0;
    fc3_rec_imag[i] = 0;
  }
  
  num_satom = multiplicity->dims[0];

  i = p2s[pi0];

  for (j = 0; j < num_satom; j++) {
    if (s2p[j] != p2s[pi1]) {
      continue;
    }
    phase_factor1 =
      get_phase_factor(q, shortest_vectors, multiplicity, pi0, j, 1);
    for (k = 0; k < num_satom; k++) {
      if (s2p[k] != p2s[pi2]) {
	continue;
      }
      if (atc[i*num_satom*num_satom + j*num_satom + k]){
      	continue;
      	}
      phase_factor2 =
	get_phase_factor(q, shortest_vectors, multiplicity, pi0, k, 2);
      fc3_elem = fc3->data + (i * 27 * num_satom * num_satom +
			      j * 27 * num_satom +
			      k * 27);

      phase_factor = phonoc_complex_prod(phase_factor1, phase_factor2);
      for (l = 0; l < 27; l++) {
	fc3_rec_real[l] +=
	  lapack_complex_double_real(phase_factor) * fc3_elem[l];
	fc3_rec_imag[l] +=
	  lapack_complex_double_imag(phase_factor) * fc3_elem[l];
      }
    }
  }

  for (i = 0; i < 27; i++) {
    fc3_rec_elem[i] =
      lapack_make_complex_double(fc3_rec_real[i], fc3_rec_imag[i]);
  }
}


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "phonoc_array.h"
#include "phonoc_utils.h"
#include "phonon3_h/imag_self_energy.h"

static double
sum_thm_imag_self_energy_at_band(const int num_band,
				 const double *fc3_normal_sqared,
				 const double n0,
				 const double *n1,
				 const double *n2,
				 const double *g0,
				 const double *g1,
				 const double *g2);
static double
sum_thm_imag_self_energy_at_band_0K(const int num_band,
				    const double *fc3_normal_sqared,
				    const double *n1,
				    const double *n2,
				    const double *g);

static double get_imag_self_energy_at_band(double *imag_self_energy,
					   const int band_index,
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
static double sum_imag_self_energy_at_band(const int num_band,
					   const double *fc3_normal_sqared,
					   const double fpoint,
					   const double *freqs0,
					   const double *freqs1,
					   const double *asigma,
					   const double temperature,
                                           const double cutoff_delta,
					   const double cutoff_frequency, 
					   const double cutoff_gamma);
static double sum_imag_self_energy_at_band_0K(const int num_band,
					      const double *fc3_normal_sqared,
					      const double fpoint,
					      const double *freqs0,
					      const double *freqs1,
					      const double *asigma,
                                              const double cutoff_delta,
					      const double cutoff_frequency,
					      const double cutoff_gamma);
    
/* imag_self_energy[num_band0] */
/* fc3_normal_sqared[num_triplets, num_band0, num_band, num_band] */
void get_imag_self_energy(double *imag_self_energy,
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
			  const double cutoff_gamma)
{
  int i, num_band0;
  num_band0 = fc3_normal_sqared->dims[1];
  
  for (i = 0; i < num_band0; i++) {
    imag_self_energy[i] =
      get_imag_self_energy_at_band(imag_self_energy,
				   i,
				   fc3_normal_sqared,
				   fpoint,
				   frequencies,
				   grid_point_triplets,
				   triplet_weights,
				   asigma,
				   temperature,
				   unit_conversion_factor,
                   cutoff_delta,
				   cutoff_frequency,
				   cutoff_gamma);
  }
}

void get_imag_self_energy_at_bands(double *imag_self_energy,
				   const Darray *fc3_normal_sqared,
				   const int *band_indices,
				   const double *frequencies,
				   const int *grid_point_triplets,
				   const int *triplet_weights,
				   const double *asigma, // adaptive sigma
				   const double temperature,
				   const double unit_conversion_factor,
                                   const double cutoff_delta,
				   const double cutoff_frequency, 
				   const double cutoff_hfrequency,
				   const double cutoff_gamma)
{
  int i, num_band0, num_band, gp0;
  double fpoint;
  
  num_band0 = fc3_normal_sqared->dims[1];
  num_band = fc3_normal_sqared->dims[2];
  gp0 = grid_point_triplets[0];

  /* num_band0 and num_band_indices have to be same. The shape of asigma and fc3_normal_sqared must be the same*/
  for (i = 0; i < num_band0; i++) {
    fpoint = frequencies[gp0 * num_band + band_indices[i]];
    if (fpoint>cutoff_hfrequency || fpoint < cutoff_frequency) {
      imag_self_energy[i] = 0;
      continue;} // frequency higher than the up limit is ignored
    imag_self_energy[i] =
      get_imag_self_energy_at_band(imag_self_energy,
				   i,
				   fc3_normal_sqared,
				   fpoint,
				   frequencies,
				   grid_point_triplets,
				   triplet_weights,
				   asigma,
				   temperature,
				   unit_conversion_factor,
                                   cutoff_delta,
				   cutoff_frequency, 
				   cutoff_gamma);
  }
}

int get_jointDOS(double *jdos,
		 const int num_omega,
		 const int num_triplet,
		 const int num_band,
		 const double *o,
		 const double *f,
		 const int *w,
		 const double sigma)
{
  int i, j, k, l;
  double f2, f3;

#pragma omp parallel for private(j, k, l, f2, f3)
  for (i = 0; i < num_omega; i++) {
    jdos[i * 2] = 0.0; //annihilation
    jdos[i * 2 + 1] = 0.0; //creation
    for (j = 0; j < num_triplet; j++) {
      for (k = 0; k < num_band; k++) {
	for (l = 0; l < num_band; l++) {
	  f2 = f[j * 3 * num_band + num_band + k];
	  f3 = f[j * 3 * num_band + 2 * num_band + l];
	  jdos[i * 2] += gaussian(f2 + f3 - o[i], sigma) * w[j];
	  jdos[i * 2 + 1] += gaussian(o[i] - f2 + f3, sigma) * w[j];
	}
      }
    }
  }

  return 1;
}

static double get_imag_self_energy_at_band(double *imag_self_energy,
					   const int band_index,
					   const Darray *fc3_normal_sqared,
					   const double fpoint,
					   const double *frequencies,
					   const int *grid_point_triplets,
					   const int *triplet_weights,
					   const double *asigma, // adaptive sigma
					   const double temperature,
					   const double unit_conversion_factor,
                                           const double cutoff_delta,
					   const double cutoff_frequency,
					   const double cutoff_gamma)
{
  int i, num_triplets, num_band0, num_band, gp1, gp2;
  double sum_g;

  num_triplets = fc3_normal_sqared->dims[0];
  num_band0 = fc3_normal_sqared->dims[1];
  num_band = fc3_normal_sqared->dims[2];

  sum_g = 0;
#pragma omp parallel for private(gp1, gp2) reduction(+:sum_g)
  for (i = 0; i < num_triplets; i++) {
    gp1 = grid_point_triplets[i * 3 + 1];
    gp2 = grid_point_triplets[i * 3 + 2];
    if (temperature > 0) {
      sum_g +=
	sum_imag_self_energy_at_band(num_band,
				     fc3_normal_sqared->data +
				     i * num_band0 * num_band * num_band +
				     band_index * num_band * num_band,
				     fpoint,
				     frequencies + gp1 * num_band,
				     frequencies + gp2 * num_band,
				     asigma + i * num_band0 * num_band * num_band +
				     band_index * num_band * num_band,
				     temperature,
                                     cutoff_delta,
				     cutoff_frequency,
                                     cutoff_gamma) *
	triplet_weights[i] * unit_conversion_factor;
    } else {
      sum_g +=
	sum_imag_self_energy_at_band_0K(num_band,
					fc3_normal_sqared->data +
					i * num_band0 * num_band * num_band +
					band_index * num_band * num_band,
					fpoint,
					frequencies + gp1 * num_band,
					frequencies + gp2 * num_band,
					asigma + i * num_band0 * num_band * num_band +
					band_index * num_band * num_band,
                                        cutoff_delta,
					cutoff_frequency,
				        cutoff_gamma) *
	triplet_weights[i] * unit_conversion_factor;
    }
  }
  return sum_g;
}

static double sum_imag_self_energy_at_band(const int num_band,
					   const double *fc3_normal_sqared,
					   const double freq0,
					   const double *freqs1,
					   const double *freqs2,
					   const double *sigmas,
					   const double temperature,
                       const double cutoff_delta,
					   const double cutoff_frequency,
					   const double cut_gamma)
{
  int i, j;
  double n0, n1, n2, g0, g1, g2, sum_g, fsum0, fsum1, fsum2, sigma;
  sum_g = 0;
  if (freq0 < cutoff_frequency) return sum_g;
  n0 = bose_einstein(freq0, temperature);
  for (i = 0; i < num_band; i++) {
    if (freqs1[i] > cutoff_frequency) {
      n1 = bose_einstein(freqs1[i], temperature);
      for (j = 0; j < num_band; j++) {
	sigma=sigmas[i * num_band + j];
	if (freqs2[j] > cutoff_frequency && sigma > cut_gamma) {
          fsum0=-freq0 + freqs1[i] + freqs2[j];
          fsum1=freq0 - freqs1[i] + freqs2[j];
          fsum2=freq0 + freqs1[i] - freqs2[j];
          if (fabs(fsum0) > cutoff_delta &&
              fabs(fsum1) > cutoff_delta &&
              fabs(fsum2) > cutoff_delta)
	  {
            continue;
	  } 
	  n2 = bose_einstein(freqs2[j], temperature);
	  g0 = gaussian(fsum0, sigma);
	  g1 = gaussian(fsum1, sigma);
	  g2 = gaussian(fsum2, sigma);
// 	  sum_g += ((n1 + n2 + 1) * g0 + (n1 - n2) * (g2 - g1)) *
// 	    fc3_normal_sqared[i * num_band + j];
	  sum_g += fc3_normal_sqared[i * num_band + j] * (n1 * n2 * (n0+1) * g0 + n0 * n2 * (n1+1) * g1 + n0 * n1 * (n2+1) * g2);
	}
      }
    }
  }
  sum_g /= n0 * (n0+1);
  return sum_g;
}

static double sum_imag_self_energy_at_band_0K(const int num_band,
					      const double *fc3_normal_sqared,
					      const double fpoint,
					      const double *freqs0,
					      const double *freqs1,
					      const double *sigmas,
                          const double cutoff_delta,
					      const double cutoff_frequency,
					      const double cut_gamma)
{
  int i, j;
  double g1, sum_g, fsum, sigma;

  sum_g = 0;
  for (i = 0; i < num_band; i++) {
    if (freqs0[i] > cutoff_frequency) {
      for (j = 0; j < num_band; j++) {
	sigma = sigmas[i * num_band + j];
	if (freqs1[j] > cutoff_frequency && sigma > cut_gamma) {
          fsum=fpoint - freqs0[i] - freqs1[j];
          if (fabs(fsum)>cutoff_delta)
          {  continue; }
	  g1 = gaussian(fsum, sigma);
	  sum_g += g1 * fc3_normal_sqared[i * num_band + j];
	}
      }
    }
  }
  return sum_g;
}

int get_decay_channels(double *decay,
		       const int num_omega,
		       const int num_triplet,
		       const int num_band,
		       const double *o,
		       const double *f,
		       const double *fc3_normal_sqared,
		       const double sigma,
		       const double t,
		       const double cutoff_frequency)
{
  int i, j, k, l, address_a, address_d;
  double f2, f3, n2, n3;

#pragma omp parallel for private(j, k, l, address_a, address_d, f2, f3, n2, n3)
  for (i = 0; i < num_triplet; i++) {
    for (j = 0; j < num_band; j++) {
      for (k = 0; k < num_band; k++) {
	for (l = 0; l < num_omega; l++) {
	  address_a = i * num_omega * num_band * num_band + l * num_band * num_band + j * num_band + k;
	  address_d = i * num_band * num_band + j * num_band + k;
	  f2 = f[i * 3 * num_band + num_band + j];
	  f3 = f[i * 3 * num_band + 2 * num_band + k];
	  if (f2>cutoff_frequency && f3 >cutoff_frequency){
	    if (t > 0) 
	    {
	      n2 = bose_einstein(f2, t);
	      n3 = bose_einstein(f3, t);
	      #pragma omp atomic
	      decay[address_d] += ((1.0 + n2 + n3) * 
	        (gaussian(f2 + f3 - o[l], sigma) - gaussian(f2 + f3 + o[l], sigma)) +
		  (n2 - n3) * (gaussian(o[l]+ f2 - f3, sigma) -
		    gaussian(o[l]- f2 +f3 , sigma))) * fc3_normal_sqared[address_a];
	    } 
	    else 
	    {
	      decay[address_d] += (gaussian(f2 + f3 - o[l], sigma) - gaussian(f2 + f3 + o[l], sigma))  * fc3_normal_sqared[address_a];
	    }
	  }
	}
      }
    }
  }

  return 1;
}

void get_thm_imag_self_energy_at_bands(double *imag_self_energy,
				       const Darray *fc3_normal_sqared,
				       const double *frequencies,
				       const int *triplets,
				       const int *weights,
				       const double *g,
				       const int *band_indices,
				       const double temperature,
				       const double unit_conversion_factor,
				       const double cutoff_frequency)
{
  int i, j, num_triplets, num_band0, num_band, gp1, gp2;
  double f1, f2;
  double *n1, *n2, *ise, n0, f0;

  num_triplets = fc3_normal_sqared->dims[0];
  num_band0 = fc3_normal_sqared->dims[1];
  num_band = fc3_normal_sqared->dims[2];

  ise = (double*)malloc(sizeof(double) * num_triplets * num_band0);
  
#pragma omp parallel for private(j, gp1, gp2, n0, f0,  n1, n2, f1, f2)
  for (i = 0; i < num_triplets; i++) {
    gp1 = triplets[i * 3 + 1];
    gp2 = triplets[i * 3 + 2];
    n1 = (double*)malloc(sizeof(double) * num_band);
    n2 = (double*)malloc(sizeof(double) * num_band);
    for (j = 0; j < num_band; j++) {
      f1 = frequencies[gp1 * num_band + j];
      f2 = frequencies[gp2 * num_band + j];
      if (f1 > cutoff_frequency)
	    n1[j] = bose_einstein(f1, temperature);
      else
	    n1[j] = -1;

      if (f2 > cutoff_frequency)
	    n2[j] = bose_einstein(f2, temperature);
      else
	    n2[j] = -1;
    }
    
    for (j = 0; j < num_band0; j++) {
      f0 = frequencies[triplets[i * 3] * num_band + band_indices[j]];
      if (temperature > 0) {
        if (f0 > cutoff_frequency)
          n0 = bose_einstein(f0, temperature);
	    else
	      n0 = -1;
	    if (n0 < 0) {
	      ise[i * num_band0 + j] = 0.;
	      continue;
	    }
        ise[i * num_band0 + j] =
            sum_thm_imag_self_energy_at_band
          (num_band,
          	fc3_normal_sqared->data +
          	 i * num_band0 * num_band * num_band + j * num_band * num_band,
          n0,
          n1,
          n2,
          g + i * num_band0 * num_band * num_band + j * num_band * num_band, //g0
          g + (i + num_triplets) * num_band0 * num_band * num_band +
	      j * num_band * num_band,  //g1
	      g + (i + 2 * num_triplets) * num_band0 * num_band * num_band +
	      j * num_band * num_band); //g2
      } else {
      	ise[i * num_band0 + j] =
	  sum_thm_imag_self_energy_at_band_0K
	  (num_band,
	   fc3_normal_sqared->data +
	   i * num_band0 * num_band * num_band + j * num_band * num_band,
	   n1,
	   n2,
	   g + i * num_band0 * num_band * num_band + j * num_band * num_band);
      }
    }
    free(n1);
    free(n2);
  }
  for (i = 0; i < num_band0; i++) {
    imag_self_energy[i] = 0;
    for (j = 0; j < num_triplets; j++) {

      imag_self_energy[i] += ise[j * num_band0 + i] * weights[j];
    }
    imag_self_energy[i] *= unit_conversion_factor;
  }
  free(ise);
}

static double
sum_thm_imag_self_energy_at_band(const int num_band,
				 const double *fc3_normal_sqared,
				 const double n0,
				 const double *n1,
				 const double *n2,
				 const double *g0,
				 const double *g1,
				 const double *g2)
{
  int i, j;
  double sum_g, ni, nj;

  sum_g = 0;
  for (i = 0; i < num_band; i++) {
    ni = n1[i];
    if (ni < 0) {continue;}
    for (j = 0; j < num_band; j++) {
      nj = n2[j];
      if (nj < 0) {continue;}
      sum_g += ((n0+1) * ni * nj * g0[i * num_band + j] +
                (ni+1) * n0 * nj * g1[i * num_band + j] +
                (nj+1) * n0 * ni * g2[i * num_band + j]) * 
                fc3_normal_sqared[i * num_band + j];
    }
  }
  sum_g /= n0 * (n0 + 1);
  return sum_g;
}

static double
sum_thm_imag_self_energy_at_band_0K(const int num_band,
				    const double *fc3_normal_sqared,
				    const double *n1,
				    const double *n2,
				    const double *g1)
{
  int i, j;
  double sum_g;

  sum_g = 0;
  for (i = 0; i < num_band; i++) {
    if (n1[i] < 0) {continue;}
    for (j = 0; j < num_band; j++) {
      if (n2[j] < 0) {continue;}
      sum_g += g1[i * num_band + j] * fc3_normal_sqared[i * num_band + j];
    }
  }
  return sum_g;
}


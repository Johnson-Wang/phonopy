#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <math.h>
#include "phonoc_array.h"
#include "phonoc_utils.h"
#include "phonon3_h/interaction.h"
#include "phonon3_h/real_to_reciprocal.h"
#include "phonon3_h/reciprocal_to_normal.h"

static const int index_exchange[6][3] = {{0, 1, 2},
					 {2, 0, 1},
					 {1, 2, 0},
					 {2, 1, 0},
					 {0, 2, 1},
					 {1, 0, 2}};

static void real_to_normal(double *fc3_normal_squared,
			   const double *freqs0,
			   const double *freqs1,
			   const double *freqs2,		      
			   const lapack_complex_double *eigvecs0,
			   const lapack_complex_double *eigvecs1,
			   const lapack_complex_double *eigvecs2,
			   const Darray *fc3,
			   const int *atc,
			   const int *atc_rec,
               const char* g_skip,
			   const double q[9], /* q0, q1, q2 */
			   const Darray *shortest_vectors,
			   const Iarray *multiplicity,
			   const double *masses,
			   const int *p2s_map,
			   const int *s2p_map,
			   const int *band_indices,
			   const int num_band0,
			   const int num_band,
			   const int pos_band0,
			   const double cutoff_frequency,
			   const double cutoff_hfrequency,
                           const double cutoff_delta);
static void real_to_normal_sym_q(double *fc3_normal_squared,
				 double *freqs[3],
				 lapack_complex_double *eigvecs[3],
				 const Darray *fc3,
				 const int *atc,
				 const int *atc_rec,
                 const char* g_skip,
				 const double q[9], /* q0, q1, q2 */
				 const Darray *shortest_vectors,
				 const Iarray *multiplicity,
				 const double *masses,
				 const int *p2s_map,
				 const int *s2p_map,
				 const int *band_indices,
				 const int num_band0,
				 const int num_band,
				 const double cutoff_frequency,
				 const double cutoff_hfrequency,
                                 const double cutoff_delta);
static int collect_undone_grid_points(int *undone,
				      char *phonon_done,
				      const Iarray *triplets);
static void interaction_degeneracy_triplet(double *interaction,
				const int *degeneracy,
				const int triplet[3],
				const int band_indices[],
				const int num_band0,
				const int num_band);
void interaction_degeneracy_grid(double *interaction,
				const int *degeneracy,
				const int triplets_grid[][3],
				const int band_indices[],
				const int num_triplets,
				const int num_band0,
				const int num_band)
{
  int i;
  #pragma omp parallel for
  for (i = 0; i < num_triplets; i++) // i: grid2 index
    interaction_degeneracy_triplet(interaction + i * num_band0 * num_band * num_band,
                                degeneracy,
                                triplets_grid[i],
                                band_indices,
                                num_band0,
                                num_band);

}

static void interaction_degeneracy_triplet(double *interaction,
				const int *degeneracy,
				const int triplet[3],
				const int band_indices[],
				const int num_band0,
				const int num_band)
{
  //The frog jump algorithm for the degeneracy.
  int i, j, k, l, ndeg, is_done;
  double *interaction_temp = (double *) malloc(sizeof(double) * num_band * num_band);
  int *deg[3], *deg_pos = (int*) malloc(sizeof(int) * num_band);
  //i is the index for the position of the current frog
  for (j = 0; j < 3; j++)
    deg[j] = degeneracy + triplet[j] * num_band;
  // for the band indices of the first phonon
  //triplet0
  for (i = 0; i < num_band0; i++)
  {
    ndeg = 0;
    for (j = 0; j < num_band0; j++){
      if (deg[0][band_indices[j]] == deg[0][band_indices[i]])
      {
        if (j < i)
          break;
        else
          deg_pos[ndeg++] = j;
      }
    }
    if (j < i) continue; // the current branch has been done previously. for the case e.g. deg={0, 1, 2, 1...}
    if (ndeg > 1)
      for (k = 0; k < num_band * num_band; k++)
      {
        interaction_temp[k] = 0.;
        for (j = 0; j < ndeg; j++)
          interaction_temp[k] += interaction[deg_pos[j] * num_band * num_band + k] / ndeg;
        //assign the new value of scatt
        for (j = 0; j < ndeg; j++)
          interaction[deg_pos[j] * num_band * num_band + k] = interaction_temp[k];
      }
  }
  //triplet1
  for (i = 0; i < num_band; i++)
  {
    ndeg = 0;
    for (j = 0; j < num_band; j++)
      if (deg[1][j] == deg[1][i])
      {
        if (j < i)
          break;
        else
          deg_pos[ndeg++] = j;
      }
    if (j < i) continue;
    if (ndeg > 1)
    {
      for (k = 0; k < num_band0; k++) //triplet 0
        for (l = 0; l < num_band; l++) //triplet 2
        {
          interaction_temp[k * num_band + l] = 0.;
          for (j = 0; j < ndeg; j++)
            interaction_temp[k * num_band + l] += interaction[k * num_band * num_band + deg_pos[j] * num_band + l] / ndeg;
          for (j = 0; j < ndeg; j++)
            interaction[k * num_band * num_band + deg_pos[j] * num_band + l] = interaction_temp[k * num_band + l];
        }
    }
  }
  //triplet2
  for (i = 0; i < num_band; i++)
  {
    ndeg = 0;
    for (j = 0; j < num_band; j++)
      if (deg[2][j] == deg[2][i])
      {
        if (j < i)
          break;
        else
          deg_pos[ndeg++] = j;
      }
    if (j < i) continue;
    if (ndeg > 1)
      for (k = 0; k < num_band0; k++)
        for (l = 0; l < num_band; l++)
        {
          interaction_temp[k * num_band + l] = 0.;
          for (j = 0; j < ndeg; j++)
            interaction_temp[k * num_band + l] += interaction[k * num_band * num_band + l * num_band + deg_pos[j]] / ndeg;
          for (j = 0; j < ndeg; j++)
            interaction[k * num_band * num_band + l * num_band + deg_pos[j]] = interaction_temp[k * num_band + l];
        }
  }
  free(interaction_temp);
  free(deg_pos);
}



void set_phonon_triplets(Darray *frequencies,
			 Carray *eigenvectors,
			 Iarray *degeneracies,
			 char *phonon_done,
			 const Iarray *triplets,
			 const int *grid_address,
			 const int *mesh,
			 const Darray *fc2,
			 const Darray *svecs_fc2,
			 const Iarray *multi_fc2,
			 const double *masses_fc2,
			 const int *p2s_fc2,
			 const int *s2p_fc2,
			 const double unit_conversion_factor,
			 const double *born,
			 const double *dielectric,
			 const double *reciprocal_lattice,
			 const double *q_direction,
			 const double nac_factor,
			 const char uplo)
{
  int num_undone;
  int *undone;

  undone = (int*)malloc(sizeof(int) * frequencies->dims[0]);
  num_undone = collect_undone_grid_points(undone, phonon_done, triplets);

  get_undone_phonons(frequencies,
		     eigenvectors,
		     degeneracies,
		     undone,
		     num_undone,
		     grid_address,
		     mesh,
		     fc2,
		     svecs_fc2,
		     multi_fc2,
		     masses_fc2,
		     p2s_fc2,
		     s2p_fc2,
		     unit_conversion_factor,
		     born,
		     dielectric,
		     reciprocal_lattice,
		     q_direction,
		     nac_factor,
		     uplo);

  free(undone);
}

/* fc3_normal_squared[num_triplets, num_band0, num_band, num_band] */
void get_interaction(Darray *fc3_normal_squared,
		     const Darray *frequencies,
		     const Carray *eigenvectors,
		     const Iarray *triplets,
		     const int *grid_address,
		     const int *mesh,
		     const Darray *fc3,
             const int *atc,
             const int *atc_rec,
             const char* g_skip,
		     const Darray *shortest_vectors,
		     const Iarray *multiplicity,
		     const double *masses,
		     const int *p2s_map,
		     const int *s2p_map,
		     const int *band_indices,
		     const int symmetrize_fc3_q,
		     const double cutoff_frequency,
		     const double cutoff_hfrequency,
             const double cutoff_delta)
{
  int i, j, k, gp, num_band, num_band0;
  double *freqs[3];
  lapack_complex_double *eigvecs[3];
  double q[9];

  num_band = frequencies->dims[1];
  num_band0 = fc3_normal_squared->dims[1];

#pragma omp parallel for private(j, k, q, gp, freqs, eigvecs)
  for (i = 0; i < triplets->dims[0]; i++) {

    for (j = 0; j < 3; j++) {
      gp = triplets->data[i * 3 + j];
      for (k = 0; k < 3; k++) {
	q[j * 3 + k] = ((double)grid_address[gp * 3 + k]) / mesh[k];
      }
      freqs[j] = frequencies->data + gp * num_band;
      eigvecs[j] = eigenvectors->data + gp * num_band * num_band;
    }

    if (symmetrize_fc3_q) {
      real_to_normal_sym_q((fc3_normal_squared->data +
			    i * num_band0 * num_band * num_band),
			   freqs,
			   eigvecs,
			   fc3,
			   atc,
			   atc_rec,
               g_skip + i * num_band0 * num_band * num_band,
			   q, /* q0, q1, q2 */
			   shortest_vectors,
			   multiplicity,
			   masses,
			   p2s_map,
			   s2p_map,
			   band_indices,
			   num_band0,
			   num_band,
			   cutoff_frequency,
			   cutoff_hfrequency,
               cutoff_delta);
    } else {
      real_to_normal((fc3_normal_squared->data +
		      i * num_band0 * num_band * num_band),
		     freqs[0],
		     freqs[1],
		     freqs[2],
		     eigvecs[0],
		     eigvecs[1],
		     eigvecs[2],
		     fc3,
		     atc,
		     atc_rec,
             g_skip + i * num_band0 * num_band * num_band,
		     q, /* q0, q1, q2 */
		     shortest_vectors,
		     multiplicity,
		     masses,
		     p2s_map,
		     s2p_map,
		     band_indices,
		     num_band0,
		     num_band,
		     0,
		     cutoff_frequency,
		     cutoff_hfrequency,
             cutoff_delta);
    }
  }
}

static void real_to_normal(double *fc3_normal_squared,
			   const double *freqs0,
			   const double *freqs1,
			   const double *freqs2,		      
			   const lapack_complex_double *eigvecs0,
			   const lapack_complex_double *eigvecs1,
			   const lapack_complex_double *eigvecs2,
			   const Darray *fc3,
			   const int *atc, //atom triplet cut off
			   const int *atc_rec,
               const char* g_skip,
			   const double q[9], /* q0, q1, q2 */
			   const Darray *shortest_vectors,
			   const Iarray *multiplicity,
			   const double *masses,
			   const int *p2s_map,
			   const int *s2p_map,
			   const int *band_indices,
			   const int num_band0,
			   const int num_band,
			   const int pos_band0,
			   const double cutoff_frequency,
			   const double cutoff_hfrequency,
               const double cutoff_delta)
			   
{
  int num_patom;
  lapack_complex_double *fc3_reciprocal;

  num_patom = num_band / 3;

  fc3_reciprocal =
    (lapack_complex_double*)malloc(sizeof(lapack_complex_double) *
				   num_patom * num_patom * num_patom * 27);

  real_to_reciprocal(fc3_reciprocal,
		     q,
		     fc3,
		     atc,
		     shortest_vectors,
		     multiplicity,
		     p2s_map,
		     s2p_map);

  reciprocal_to_normal(fc3_normal_squared,
		       fc3_reciprocal,
		       atc_rec,
		       freqs0,
		       freqs1,
		       freqs2,
		       eigvecs0,
		       eigvecs1,
		       eigvecs2,
		       masses,
		       band_indices,
		       num_band0,
		       num_band,
		       pos_band0,
               g_skip,
		       cutoff_frequency,
		       cutoff_hfrequency,
               cutoff_delta);

  free(fc3_reciprocal);
}

static void real_to_normal_sym_q(double *fc3_normal_squared,
				 double *freqs[3],
				 lapack_complex_double *eigvecs[3],
				 const Darray *fc3,
				 const int *atc,
				 const int *atc_rec,
                 const char* g_skip,
				 const double q[9], /* q0, q1, q2 */
				 const Darray *shortest_vectors,
				 const Iarray *multiplicity,
				 const double *masses,
				 const int *p2s_map,
				 const int *s2p_map,
				 const int *band_indices,
				 const int num_band0,
				 const int num_band,
				 const double cutoff_frequency,
				 const double cutoff_hfrequency,
                 const double cutoff_delta)
{
  int i, j, k, l;
  int band_ex[3];
  int band_shape[3];
  int bb = num_band*num_band;
  int *ie;
  int pos_band0=0;
  double q_ex[9];
  double *fc3_normal_squared_ex;
  char *g_skip_new;
//  double dmax=0, dtemp;
  fc3_normal_squared_ex =
    (double*)malloc(sizeof(double) * num_band0 * num_band * num_band);
  g_skip_new = (char*) malloc(sizeof(char)*num_band0 * num_band * num_band);
  for (i = 0; i < num_band0 * num_band * num_band; i++) {
    fc3_normal_squared[i] = 0;
    g_skip_new[i] = 0;
  }

  for (i = 0; i < 6; i++) {

    ie = index_exchange[i];
    for (j = 0; j < 3; j++)
    {
      if (ie[j] == 0)
      {
        band_shape[j] = num_band0;
        pos_band0 = j;
      }
      else
        band_shape[j] = num_band;
    }
    for (j = 0; j < 3; j ++) {
      for (k = 0; k < 3; k ++) {
	q_ex[j * 3 + k] = q[ie[j] * 3 + k];
      }
    }

    for (j = 0; j < num_band0; j++)
      for (k = 0; k < num_band; k++)
        for (l = 0; l < num_band; l++)
        {
          band_ex[0] = j;
          band_ex[1] = k;
          band_ex[2] = l;
          g_skip_new[band_ex[ie[0]] * band_shape[1] * band_shape[2] + band_ex[ie[1]] * band_shape[2] + band_ex[ie[2]]] =
             g_skip[j * num_band * num_band + k * num_band + l];
        }

    real_to_normal(fc3_normal_squared_ex,
		   freqs[ie[0]],
		   freqs[ie[1]],
		   freqs[ie[2]],
		   eigvecs[ie[0]],
		   eigvecs[ie[1]],
		   eigvecs[ie[2]],
		   fc3,
		   atc,
		   atc_rec,
           g_skip_new,
		   q_ex, /* q0, q1, q2 */
		   shortest_vectors,
		   multiplicity,
		   masses,
		   p2s_map,
		   s2p_map,
		   band_indices,
		   num_band0,
		   num_band,
		   pos_band0,
		   cutoff_frequency,
		   cutoff_hfrequency,
           cutoff_delta);

    for (j = 0; j < num_band0; j++) {
      for (k = 0; k < num_band; k++) {
        for (l = 0; l < num_band; l++) {
          band_ex[0] = j;
          band_ex[1] = k;
          band_ex[2] = l;
          fc3_normal_squared[j * num_band * num_band +
                     k * num_band +
                     l] +=
            fc3_normal_squared_ex[band_ex[ie[0]] *
                      band_shape[1] * band_shape[2]+
                      band_ex[ie[1]] * band_shape[2] +
                      band_ex[ie[2]]] / 6.0;
//          dtemp = fabs(fc3_normal_squared[j * num_band * num_band +
//                     k * num_band +
//                     l] * 6 / (i + 1) - fc3_normal_squared_ex[band_ex[index_exchange[i][0]] *
//                      num_band * num_band +
//                      band_ex[index_exchange[i][1]] * num_band +
//                      band_ex[index_exchange[i][2]]]);
//          dmax = (dtemp > dmax)? dtemp: dmax;
        }
      }
    }
  }
  free(g_skip_new);
  free(fc3_normal_squared_ex);
}

static int collect_undone_grid_points(int *undone,
				      char *phonon_done,
				      const Iarray *triplets)
{
  int i, j, gp, num_undone;

  num_undone = 0;
  for (i = 0; i < triplets->dims[0]; i++) {
    for (j = 0; j < 3; j++) {
      gp = triplets->data[i * 3 + j];
      if (phonon_done[gp] == 0) {
	undone[num_undone] = gp;
	num_undone++;
	phonon_done[gp] = 1;
      }
    }
  }

  return num_undone;
}

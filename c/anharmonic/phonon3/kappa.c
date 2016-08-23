#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "phonoc_array.h"
#include "phonoc_utils.h"
#include "phonon3_h/kappa.h"
void get_kappa_at_grid_point(double kappa[],
			     Iarray* kpt_rotations_at_q, //rotations that can map the q* to other qpoints
			     const double rec_lat[], 
			     const double gv[],
			     const double heat_capacity[], 
			     const double mfp[], //dim[num_temp, num_band, 3] 
			     const int deg[],
			     const int num_band,
			     const int num_temp)
{
  int i,j,k, l;
  int m,remnant;
  double r[9], v[3], v2_tensor[6], v_temp[3];
  double mfp_temp[num_temp * 3];
  double vl_sum[num_temp * num_band*6]; // group_velocity * mfp
  for (i=0;i<num_temp * num_band * 6;i++)
    vl_sum[i]=0.0;
  for (i=0;i<kpt_rotations_at_q->dims[0];i++) //number of rotations
  {
    mat_copy_matrix_id3_flatten(r, kpt_rotations_at_q->data + i * 9); 
    mat_inverse_matrix_d3_flatten(r, r);
    mat_get_similar_matrix_d3_flatten(r, r, rec_lat);
    //Considering degenerate group velocity of different branches
    remnant = 0;

    for (j=0;j<num_band;j++)
    {
      if (!(remnant--))
      {
	for (k=0; k<3; k++)
	{
	  v[k]=0;
	  for (l=0; l<num_temp; l++)
	    mfp_temp[l*3+k]=0.;
	}
	for (m=0;m<num_band-j; m++)
	{
	  if (deg[j+m] == j){
	    mat_multiply_matrix_vector_d3_flatten(v_temp, r, gv + (j + m) * 3);
	    mat_add_vector_d3_flatten(v, v, v_temp);
	    for (l=0; l<num_temp; l++)
	    {
	      mat_multiply_matrix_vector_d3_flatten(v_temp, r, mfp + l * num_band * 3 + (j + m) * 3);
	      mat_add_vector_d3_flatten(mfp_temp + l * 3, mfp_temp + l * 3, v_temp);
	    }
	  }
	  else
	    break;
	}
	remnant=m-1;
	for (k=0; k<3; k++){
	  v[k] = v[k] / m;
	  for (l = 0; l < num_temp; l++)
	    mfp_temp[l * 3 + k] /= m;
	}
      }
      for (l=0; l<num_temp; l++)
      {
	mat_vector_outer_product_flatten(v2_tensor,v, mfp_temp + l * 3);
	for (k=0; k<6;k++)
	  vl_sum[l * num_band * 6 + j * 6 + k] += v2_tensor[k];
      }
    }
  }
  for (i=0; i<num_temp;i++)
  {
    for (j=0; j<num_band; j++)
    {
      for (k=0; k<6; k++)
	kappa[i * num_band * 6 + j * 6 + k] = vl_sum[i * num_band * 6 + j * 6 + k] * heat_capacity[i * num_band + j];
    }
  }
}

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
		const int num_temp)
{
  int i, j, weight_at_q;
  Iarray *kpt_rots_at_q;
  #pragma omp parallel for private(j, kpt_rots_at_q, weight_at_q)
  for (i=0;i<num_grid; i++)
  {
    weight_at_q = 0;
    kpt_rots_at_q = (Iarray*) malloc(sizeof(Iarray));
    kpt_rots_at_q->dims[1]=9;
    for (j=0; j<num_all_grids; j++)
    {
      if (index_mappings[j] == i)
      {
	weight_at_q++;
      }
    }
    kpt_rots_at_q->dims[0] = weight_at_q;
    kpt_rots_at_q->data = (int *)malloc(weight_at_q * 9 * sizeof(int));
    weight_at_q = 0;
    for (j=0; j<num_all_grids; j++)
    {
      if (index_mappings[j] == i)
      {
	mat_copy_matrix_i3_flatten(kpt_rots_at_q->data + weight_at_q * 9, kpt_rotations + j * 9);
	weight_at_q++;
      }
    }
    get_kappa_at_grid_point(kappa + i * num_temp * num_band * 6,
			    kpt_rots_at_q, //rotations that can map the other qpoints to q*
			    rec_lat, 
			    gv + i * num_band * 3,
			    heat_capacity + i * num_temp * num_band, 
			    F + i * num_temp * num_band * 3, 
			    degeneracies + i * num_band,
			    num_band,
			    num_temp);
  }
}

void  get_collision_at_all_band(double *collision, // actually collision_in
				      const double *interaction,
				      const double *frequency,
				      const double *g,
				      const int num_triplet,
				      const double temperature,
				      const int num_band,
				      const double cutoff_frequency)
{
  int i, j, k, l;
  int int_pos, scatt_pos;
  int bbb = num_band * num_band * num_band;
  int bb = num_band * num_band;
  double f0, f1, f2, n0, n1, n2, g0, g1, g2;
  for (i=0; i<num_triplet*num_band*num_band; i++)
    collision[i]=0;
#pragma omp parallel for private(j, k, l, scatt_pos, int_pos, f0, f1, f2, n0, n1, n2, g0, g1, g2)
  for (i=0;i<num_triplet;i++)
  {
      //when temperature is zero, the distribution function is meaningless
    if (temperature<1e-5){ 
      for (j=0;j<num_band;j++)
      {
	f0=frequency[i*3*num_band+0*num_band+j];
	for (k=0;k<num_band;k++)
	{
	  f1=frequency[i*3*num_band+1*num_band+k];
	  for (l=0;l<num_band;l++)	
	  {
	    f2=frequency[i*3*num_band+2*num_band+l];
	    scatt_pos = i*bb+j*num_band+k;
	    int_pos = i*bbb+j*bb+k*num_band+l;
	    g0 = g[int_pos];
	    g1 = g[num_triplet*bbb+int_pos];
	    g2 = g[2*num_triplet*bbb + int_pos];
	    collision[scatt_pos] += interaction[int_pos] * (g0 + g1 + g2);
	  }
	}
      }
    }
    else{
      for (j=0;j<num_band;j++)
      {
	f0=frequency[i*3*num_band+0*num_band+j];
	if (f0<cutoff_frequency)
	  continue;
	n0=bose_einstein(f0, temperature);
	for (k=0;k<num_band;k++)
	{
	  f1=frequency[i*3*num_band+1*num_band+k];
	  if (f1<cutoff_frequency)
	    continue;
	  n1=bose_einstein(f1, temperature);
	  for (l=0;l<num_band;l++)
	  {
	    f2=frequency[i*3*num_band+2*num_band+l];
	    if (f2<cutoff_frequency)
	      continue;
	    n2=bose_einstein(f2, temperature);
	    scatt_pos=i*num_band*num_band+j*num_band+k;
	    int_pos = i*bbb+j*bb+k*num_band+l;
	    g0 = g[int_pos];
	    g1 = g[num_triplet*bbb+int_pos];
	    g2 = g[2*num_triplet*bbb + int_pos];
	    collision[scatt_pos] += interaction[int_pos] * (n0 * n1 * (n2 + 1) * g2 +
								  n0 * (n1 + 1) * n2 * g1 +
								  (n0 + 1) * n1 * n2 * g0);

	  }
	}
      }
    }
  }
}


void get_collision_from_reduced(double *scatt, 
				      const double *scattall,
				      const int *triplet_mapping,
				      const char *triplet_sequence,
				      const int num_triplet, 
				      const int num_band)
{
  int triplet, i, j, iunique;
  int bb = num_band * num_band;
  char seq[3];
  #pragma parallel for private(iunique, i, j, t, seq)
  for (triplet=0; triplet<num_triplet;triplet++)
  {
    iunique = triplet_mapping[triplet];
    seq[0] =  triplet_sequence[triplet*3];
    seq[1] =  triplet_sequence[triplet*3+1];
    seq[2] =  triplet_sequence[triplet*3+2];
    if (seq[1] - seq[0] == 1 || seq[1] - seq[0] == -2)
    {
      for (i=0; i<num_band; i++)
	for (j=0; j<num_band; j++)
	  scatt[triplet*bb+i*num_band+j] = scattall[iunique*3*bb+seq[2]*bb+i*num_band+j]; 
    }
    else
    {
      for (i=0; i<num_band; i++)
	for (j=0; j<num_band; j++)
	  scatt[triplet*bb+i*num_band+j] = scattall[iunique*3*bb+seq[2]*bb+j*num_band+i]; 
    }
  }
}

void get_interaction_from_reduced(double *interaction, 
				  const double *interaction_all,
				  const int *triplet_mapping,
				  const char *triplet_sequence,
				  const int num_triplet,
				  const int num_band0,
				  const int num_band)
{
  int triplet, i, j, k, iunique, ijk[3];
  int bb = num_band * num_band;
  int bbb = num_band0 * num_band * num_band;
  char *seq;
  #pragma parallel for private(iunique, i, j, k, ijk, seq)
  for (triplet=0; triplet<num_triplet;triplet++)
  {
    iunique = triplet_mapping[triplet];
    seq =  triplet_sequence + triplet * 3;
    for (i=0; i<num_band0; i++)
    {
      ijk[0] = i;
      for (j=0; j<num_band; j++)
      {
        ijk[1] = j;
        for (k=0; k<num_band;k++)
        {
          ijk[2] = k;
          interaction[triplet*bbb+ijk[seq[0]]*bb+ijk[seq[1]]*num_band+ijk[seq[2]]] = interaction_all[iunique*bbb+i*bb+j*num_band+k];
        }
      }
    }
  }
}

void  get_collision_at_all_band_permute(double *collision, // actually collision_in
					      const double *interaction,
					      const double *occupation,
					      const double *frequency,
					      const int num_triplet,
					      const int num_band,
					      const double *g,
					      const double cutoff_frequency)
{
  int l, i, j, k;
  int tbb = num_triplet*num_band*num_band;
  int bbb = num_band*num_band*num_band;
  int bb = num_band*num_band;
  int index;
  double f0, f1, f2, n0, n1, n2, W, g0, g1, g2;
  for (l=0; l<3*tbb; l++)
    collision[l]=0;
#pragma omp parallel for private(i, j, k, f0, f1, f2, n0, n1, n2, W, g0, g1, g2, index)
  for (l=0;l<num_triplet;l++)
  {
    for (i=0;i<num_band;i++)
    {
      for (j=0;j<num_band;j++)
      {
        for (k=0;k<num_band;k++)
        {
            f0=frequency[l*3*num_band+0*num_band+i];
            f1=frequency[l*3*num_band+1*num_band+j];
            f2=frequency[l*3*num_band+2*num_band+k];
            if (f0 < cutoff_frequency || f1 < cutoff_frequency || f2 < cutoff_frequency)
              continue;
            n0=occupation[l*3*num_band+0*num_band+i];
            n1=occupation[l*3*num_band+1*num_band+j];
            n2=occupation[l*3*num_band+2*num_band+k];
            index = l*bbb + i * bb + j * num_band + k;
            g0 = g[index];
            index += num_triplet*bbb;
            g1 = g[index];
            index += num_triplet * bbb;
            g2 = g[index];
            W = interaction[l*bbb+i*bb+j*num_band+k] *
            (n0 * n1 * (n2 + 1) * g2  +
              n0 * (n1 + 1) * n2 * g1 +
              (n0 + 1) * n1 * n2 * g0);
            collision[l*3*bb+2*bb+i*num_band+j] += W;
            collision[l*3*bb+1*bb+k*num_band+i] += W ;
            collision[l*3*bb+0*bb+j*num_band+k] += W;
        }
      }
    }
  }
}

void collision_degeneracy(double *scatt, 
				const int *triplet_degeneracy, 
				const int num_triplet, 
				const int num_band)
{
  int i, j, k, l, m, ndeg, *deg;
  const int nbb = num_band * num_band;
  double scatt_temp[num_band];
  const int p[3][2] = {{1, 2}, {2, 0}, {0, 1}};

  //The frog jump algorithm for the degeneracy. 
  #pragma omp parallel for private(i, j, k, l, m, ndeg, deg, scatt_temp)
  for (i = 0; i < num_triplet; i++) // i: triplet index
  {
    for (j = 0; j < 3; j++) //j: the three permute indices
    {
      
      deg = triplet_degeneracy + i * 3 * num_band + p[j][0] * num_band; // first phonon band degeneracy
     

      
      //k is the index for the position of the current frog
      k = 0;
      while(k<num_band)
      {
	 //Initialization for scatt_temp
	for (m = 0; m < num_band; m++)
	  scatt_temp[m] = 0;
	// find the length of bands to skip and sum all the values in another vector
	for (l = k; l < num_band; l++)
	{
	  if (deg[l] == k) // the lth band and the kth band are degenerate
	    for (m = 0; m < num_band; m++)
	      scatt_temp[m] += scatt[i * 3 * nbb + j * nbb + l * num_band + m];
	  else 
	    break;
	}
	ndeg = l - k; // number of degenerate states
	
	//Take the average of the degenerate states
	for (m = 0; m < num_band; m++)
	  scatt_temp[m] /= ndeg;
	
	//assign the new value of scatt
	for (l = k; l < k + ndeg; l++)
	  for (m = 0; m < num_band; m++)
	    scatt[i * 3 * nbb + j * nbb + l * num_band + m] = scatt_temp[m];
	k += ndeg;
      }   
      
      
      //Repeating the last step but for another axis
      deg = triplet_degeneracy + i * 3 * num_band + p[j][1] * num_band;  //second phonon
      k = 0;
      while(k<num_band)
      {
	 //Initialization for scatt_temp
	for (m = 0; m < num_band; m++)
	  scatt_temp[m] = 0;
	
	for (l = k; l < num_band; l++)
	{
	  if (deg[l] == k)    
	    for (m = 0; m < num_band; m++)
	      scatt_temp[m] += scatt[i * 3 * nbb + j * nbb + m * num_band + l];
	  else 
	    break;
	}
	ndeg = l - k;
	for (m = 0; m < num_band; m++)
	  scatt_temp[m] /= ndeg;
	for (l = k; l < k + ndeg; l++)
	  for (m = 0; m < num_band; m++)
	    scatt[i * 3 * nbb + j * nbb + m * num_band + l] = scatt_temp[m];
	k += ndeg;
      }
    }
  }
}

void collision_degeneracy_grid(double *scatt,
				const int *degeneracy,
				const int grid_point,
				const int *grid_points2,
				const int num_grid_points2,
				const int num_band)
{
  int i, k, l, m, ndeg, *deg;
  const int nbb = num_band * num_band;
  double scatt_temp[num_band], deg0[num_band];
  //The frog jump algorithm for the degeneracy.

  for (i=0; i<num_band; i++)
      deg0[i] = degeneracy[grid_point * num_band + i];
//  #pragma omp parallel for private(i,k, l, m, ndeg, deg, scatt_temp)
  for (i = 0; i < num_grid_points2; i++) // i: grid2 index
  {
      //k is the index for the position of the current frog

      k = 0;
      while(k<num_band)
      {
	     //Initialization for scatt_temp
        for (m = 0; m < num_band; m++)
          scatt_temp[m] = 0;
        // find the length of bands to skip and sum all the values in another vector
        for (l = k; l < num_band; l++)
        {
          if (deg0[l] == k) // the lth band and the kth band are degenerate
            for (m = 0; m < num_band; m++)
              scatt_temp[m] += scatt[i * nbb + l * num_band + m];
          else
            break;
        }
        ndeg = l - k; // number of degenerate states

        //Take the average of the degenerate states
        for (m = 0; m < num_band; m++)
          scatt_temp[m] /= ndeg;

        //assign the new value of scatt
        for (l = k; l < k + ndeg; l++)
          for (m = 0; m < num_band; m++)
            scatt[i * nbb + l * num_band + m] = scatt_temp[m];
        k += ndeg;
      }
      //Repeate the last step but for another axis
      deg = degeneracy + grid_points2[i] * num_band;  //second phonon
      k = 0;
      while(k<num_band)
      {
	    //Initialization for scatt_temp
        for (m = 0; m < num_band; m++)
          scatt_temp[m] = 0;

        for (l = k; l < num_band; l++)
        {
          if (deg[l] == k)
            for (m = 0; m < num_band; m++)
              scatt_temp[m] += scatt[i * nbb + m * num_band + l];
          else
            break;
        }
        ndeg = l - k;
        for (m = 0; m < num_band; m++)
          scatt_temp[m] /= ndeg;
        for (l = k; l < k + ndeg; l++)
          for (m = 0; m < num_band; m++)
            scatt[i * nbb + m * num_band + l] = scatt_temp[m];
        k += ndeg;
      }
  }
}

void  get_next_perturbation_at_all_bands(double *summation,
					 const double *F_prev,
					 const double *scatt,
					 const int *convergence,
					 const int *q1s,
					 const int *spg_mapping_index,
					 const double *inv_rot_sum,
					 const int num_grids,
					 const int num_triplet,
					 const int num_band)
{
  int d,i, k, l,m,all_converge=1, ir1;
  const int nb2=num_band*num_band;
  double F1_sum[3];
  double scatt_temp;
  int num_threads, thread_num;
  double *summation_thread;
  for (i=0;i<num_band;i++)
  {
    if (!convergence[i])
    {
      all_converge=0;
      break;  
    }
  }
  
  if (!all_converge)
  {
    #pragma omp parallel
      #pragma omp master
        num_threads = omp_get_num_threads();
    //printf("Number of thread:%d\n", num_threads);
    summation_thread=(double*)malloc(sizeof(double)*num_threads*num_band*3);
    for (i=0;i<num_threads*num_band*3;i++)
      summation_thread[i] = 0;
    #pragma omp parallel for private(thread_num, l, d, m, ir1, F1_sum, scatt_temp)
    for (i=0;i<num_triplet;i++)
    {
      thread_num = omp_get_thread_num();
      ir1=spg_mapping_index[q1s[i]];
      for (l=0;l<num_band;l++)
      {
	if (convergence[l])
	  continue;
	else
	{
	  for (m=0;m<num_band;m++)
	  {
	    scatt_temp=scatt[i*nb2+l*num_band+m];
	    mat_multiply_matrix_vector_d3_flatten(F1_sum, inv_rot_sum+i*9, F_prev+ir1*num_band*3+m*3);
	    for (d=0;d<3;d++)
	    {
	      summation_thread[thread_num*num_band*3+l*3+d] += scatt_temp*F1_sum[d];
	    }
	  }
	}
      } 
    }
    #pragma omp parallel for private(i,k)
    for (i=0; i<num_threads; i++)
      for (k=0;k<num_band*3; k++)
      {
        #pragma omp atomic
	summation[k] += summation_thread[i*num_band*3+k];
      }
    free(summation_thread);
  }
}






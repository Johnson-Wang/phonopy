#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <lapacke.h>
#include "phonoc_math.h"
#include "phonoc_array.h"
lapack_complex_double
phonoc_complex_prod(const lapack_complex_double a,
		    const lapack_complex_double b)
{
  lapack_complex_double c;
  c = lapack_make_complex_double
    (lapack_complex_double_real(a) * lapack_complex_double_real(b) -
     lapack_complex_double_imag(a) * lapack_complex_double_imag(b),
     lapack_complex_double_imag(a) * lapack_complex_double_real(b) +
     lapack_complex_double_real(a) * lapack_complex_double_imag(b));
  return c;
}

void  phonon_multiply_dmatrix_gbb_dvector_gb3(double *vector0, 
					 const double *matrix,
					 const double *vector1,
					 const int *weights,
					 const double *rots, //the rots should be a matrix to be multiplied with vector1 first
					 const double *rec_lat,
					 const int num_grids,
					 const int num_band)
{
  int d,i, l,m, weight;
  const int nb2=num_band*num_band;
  double v_temp[3], r[9];
  double mat_temp;
  int num_threads, thread_num;
  double *summation_thread;
  #pragma omp parallel
    #pragma omp master
      num_threads = omp_get_num_threads();
  summation_thread=(double*)malloc(sizeof(double)*num_threads*num_band*3);
  for (i=0;i<num_threads*num_band*3;i++)
    summation_thread[i] = 0;
  #pragma omp parallel for private(thread_num, weight, r, l, d, m, v_temp, mat_temp)
  for (i=0;i<num_grids;i++)
  {
    thread_num = omp_get_thread_num();
    weight = weights[i];
    mat_get_similar_matrix_d3_flatten(r, rots+9*i, rec_lat);
    for (l=0;l<num_band;l++)
    {
      for (m=0;m<num_band;m++)
      {
	mat_temp=matrix[i*nb2+l*num_band+m];
	mat_multiply_matrix_vector_d3_flatten(v_temp, r, vector1+i*num_band*3+m*3);
	for (d=0;d<3;d++)
	{
	  summation_thread[thread_num*num_band*3+l*3+d] += weight * mat_temp*v_temp[d];
	}
      }
    } 
  }
  #pragma omp parallel for private(i,m)
  for (i=0; i<num_threads; i++)
    for (m=0;m<num_band*3; m++)
    {
      #pragma omp atomic
      vector0[m] += summation_thread[i*num_band*3+m];
    }
  free(summation_thread);
}



void phonon_3_multiply_dvector_gb3_dvector_gb3(double *vector0, //gb6
					 const double *vector1, //gb3
					 const double *vector2, //gb3
					 const int *rots, //the rots should be a matrix to be multiplied with vector1 first
					 const double *rec_lat,
					 const int *mappings, // mapping from a general grid to the index in the irreducible grid list
					 const int num_grids, // should be the total number of all grids
					 const int num_band)
{
  int i,j,l,m;
  double v_temp1[3], v_temp2[3], r[9];
  int num_threads, thread_num;
  double *sum_thread;
  #pragma omp parallel
    #pragma omp master
      num_threads = omp_get_num_threads();
  sum_thread=(double*)malloc(sizeof(double)*num_threads * 3);
  for (i=0;i<num_threads * 3;i++)
    sum_thread[i] = 0;
  
  for (i=0;i<3;i++)
    vector0[i] = 0;

  #pragma omp parallel for private(j, r, l, m, v_temp1, thread_num, v_temp2)
  for (i=0;i<num_grids;i++)
  {
    thread_num = omp_get_thread_num();
    mat_copy_matrix_id3_flatten(r, rots+9*i);
    mat_get_similar_matrix_d3_flatten(r, r, rec_lat);
    mat_inverse_matrix_d3_flatten(r,r);
    m = mappings[i];
    for (l=0;l<num_band;l++)
    {
      mat_multiply_matrix_vector_d3_flatten(v_temp1, r, vector1+m*num_band*3+l*3);
      mat_multiply_matrix_vector_d3_flatten(v_temp2, r, vector2+m*num_band*3+l*3);
      for (j = 0; j < 3; j++)
	sum_thread[thread_num * 3 + j] += v_temp1[j] * v_temp2[j];
    } 
  }
  
  for (i=0; i<num_threads; i++)
  {
    for (m=0;m<3; m++)
    {
      vector0[m] += sum_thread[i*3+m];
    }
  }
  free(sum_thread);
  
}







void phonon_gb33_multiply_dvector_gb3_dvector_gb3(double *vector0, //gb6
					 const double *vector1, //gb3
					 const double *vector2, //gb3
					 const int *rots, //the rots should be a matrix to be multiplied with vector1 first
					 const double *rec_lat,
					 const int *mappings, // mapping from a general grid to the index in the irreducible grid list
					 const int num_grids, // should be the total number of all grids
					 const int num_ir_grids,
					 const int num_band)
{
  int i, j,l,m;
  double v_temp1[3], v_temp2[3], r[9], v2_tensor[6];
  int num_threads, thread_num;
  double *sum_thread;
  const int gb6 = num_ir_grids * num_band*6;
  #pragma omp parallel
    #pragma omp master
      num_threads = omp_get_num_threads();
  sum_thread=(double*)malloc(sizeof(double)*num_threads * gb6);
  for (i=0;i<num_threads * gb6;i++)
    sum_thread[i] = 0;
  
  for (i=0;i<gb6;i++)
    vector0[i] = 0;

  #pragma omp parallel for private(j, r, l, m, v_temp1, thread_num, v_temp2, v2_tensor)
  for (i=0;i<num_grids;i++)
  {
    thread_num = omp_get_thread_num();
    mat_copy_matrix_id3_flatten(r, rots+9*i);
    mat_get_similar_matrix_d3_flatten(r, r, rec_lat);
    mat_inverse_matrix_d3_flatten(r,r);
    m = mappings[i];
    if (m > num_ir_grids)
    {
      printf("Error! m=%d\n",m);
      exit(1);
    }
    for (l=0;l<num_band;l++)
    {

      mat_multiply_matrix_vector_d3_flatten(v_temp1, r, vector1+m*num_band*3+l*3);
      mat_multiply_matrix_vector_d3_flatten(v_temp2, r, vector2+m*num_band*3+l*3);
      mat_vector_outer_product_flatten(v2_tensor, v_temp1,  v_temp2);
      for (j = 0; j < 6; j++)
	sum_thread[thread_num * gb6 + m*num_band*6+l*6+j] += v2_tensor[j];
    } 
  }
  
  #pragma omp parallel for private(i,m)
  for (i=0; i<num_threads; i++)
  {
    for (m=0;m<gb6; m++)
    {
      #pragma omp atomic
      vector0[m] += sum_thread[i*gb6+m];
    }
  }
  free(sum_thread);
  
}
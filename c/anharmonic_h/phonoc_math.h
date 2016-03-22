#ifndef __phonoc_math_H__
#define __phonoc_math_H__

#include <lapacke.h>

#define M_2PI 6.283185307179586

lapack_complex_double
phonoc_complex_prod(const lapack_complex_double a,
		    const lapack_complex_double b);

#endif
void  phonon_multiply_dmatrix_gbb_dvector_gb3(double *vector0, 
					 const double *matrix,
					 const double *vector1,
					 const int *weights,
					 const double *rots, //the rots should be a matrix to be multiplied with vector1 first
					 const double *rec_lat,
					 const int num_grids,
					 const int num_band);

void phonon_3_multiply_dvector_gb3_dvector_gb3(double *vector0, //gb6
					 const double *vector1, //gb3
					 const double *vector2, //gb3
					 const int *rots, //the rots should be a matrix to be multiplied with vector1 first
					 const double *rec_lat,
					 const int *mappings, // mapping from a general grid to the index in the irreducible grid list
					 const int num_grids, // should be the total number of all grids
					 const int num_band);

void  phonon_gb33_multiply_dvector_gb3_dvector_gb3(double *vector0, //gb9
					 const double *vector1, //gb3
					 const double *vector2, //gb3
					 const int *rots, //the rots should be a matrix to be multiplied with vector1 first
					 const double *rec_lat,
					 const int *mappings, // mapping from a general grid to the index in the irreducible grid list
					 const int num_grids, // should be the total number of all grids
					 const int num_ir_grids,
					 const int num_band);
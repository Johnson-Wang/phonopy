#include <Python.h>
#include <numpy/arrayobject.h>
#include <lapacke.h>
#include "phonoc_array.h"

Iarray* convert_to_iarray(const PyArrayObject* npyary)
{
  int i;
  Iarray *ary;

  ary = (Iarray*) malloc(sizeof(Iarray));
  for (i = 0; i < npyary->nd; i++) {
    ary->dims[i] = npyary->dimensions[i];
  }
  ary->data = (int*)npyary->data;
  return ary;
}


Darray* convert_to_darray(const PyArrayObject* npyary)
{
  int i;
  Darray *ary;

  ary = (Darray*) malloc(sizeof(Darray));
  for (i = 0; i < npyary->nd; i++) {
    ary->dims[i] = npyary->dimensions[i];
  }
  ary->data = (double*)npyary->data;
  return ary;
}

Carray* convert_to_carray(const PyArrayObject* npyary)
{
  int i;
  Carray *ary;

  ary = (Carray*) malloc(sizeof(Carray));
  for (i = 0; i < npyary->nd; i++) {
    ary->dims[i] = npyary->dimensions[i];
  }
  ary->data = (lapack_complex_double*)npyary->data;
  return ary;
}

static void mat_copy_matrix_d3(double a[9], double b[9])
{
  a[0] = b[0];
  a[1] = b[1];
  a[2] = b[2];
  a[3] = b[3];
  a[4] = b[4];
  a[5] = b[5];
  a[6] = b[6];
  a[7] = b[7];
  a[8] = b[8];
}

void mat_add_vector_d3_flatten( double m[3],
			double a[3],
			double b[3] )
{
  int i;
  for ( i = 0; i < 3; i++ ) {
      m[i] = a[i] + b[i];
  }
}

static double mat_get_determinant_d3(double a[9])
{
  return a[0] * (a[4] * a[8] - a[5] * a[7])
    + a[1] * (a[5] * a[6] - a[3] * a[8])
    + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

/* m=axb */
void mat_multiply_matrix_d3_flatten(double m[9],
			    double a[9],
			    double b[9])
{
  int i, j;                   /* a_ij */
  double c[9];
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      c[i*3+j] =
	a[i*3+0] * b[j] + a[i*3+1] * b[1*3+j] + a[i*3+2] * b[2*3+j];
    }
  }
  mat_copy_matrix_d3(m, c);
}

//The tensor is symmetric and thus only the upper triangle is considered
void mat_vector_outer_product_flatten(double v[6],
			      double a[3],
			      double b[3])
{
  v[0] = a[0] * b[0];
  v[1] = a[1] * b[1];
  v[2] = a[2] * b[2];
  v[3] = a[1] * b[2];
  v[4] = a[0] * b[2];
  v[5] = a[0] * b[1];
}

void mat_add_matrix_d3_flatten( double m[9],
			double a[9],
			double b[9] )
{
  int i, j;
  for ( i = 0; i < 3; i++ ) {
    for ( j = 0; j < 3; j++ ) {
      m[i*3+j] = a[i*3+j] + b[i*3+j];
    }
  }
}

void mat_copy_vector_i3_flatten(int a[3], int b[3])
{
  a[0] = b[0];
  a[1] = b[1];
  a[2] = b[2];
}


void mat_copy_matrix_id3_flatten(double a[9], int b[9])
{
  a[0] = b[0];
  a[1] = b[1];
  a[2] = b[2];
  a[3] = b[3];
  a[4] = b[4];
  a[5] = b[5];
  a[6] = b[6];
  a[7] = b[7];
  a[8] = b[8];
}

void mat_copy_matrix_i3_flatten(int a[9], int b[9])
{
  a[0] = b[0];
  a[1] = b[1];
  a[2] = b[2];
  a[3] = b[3];
  a[4] = b[4];
  a[5] = b[5];
  a[6] = b[6];
  a[7] = b[7];
  a[8] = b[8];
}

// void mat_cast_matrix_3i_to_3d(double a[9], int b[9])
// {
//   a[0] = b[0];
//   a[1] = b[1];
//   a[2] = b[2];
//   a[3] = b[3];
//   a[4] = b[4];
//   a[5] = b[5];
//   a[6] = b[6];
//   a[7] = b[7];
//   a[8] = b[8];
// }

void mat_multiply_matrix_vector_d3_flatten(double v[3],
				   double a[9],
				   double b[3])
{
  int i;
  double c[3];
  for (i = 0; i < 3; i++)
    c[i] = a[i*3+0] * b[0] + a[i*3+1] * b[1] + a[i*3+2] * b[2];
  for (i = 0; i < 3; i++)
    v[i] = c[i];
}

double mat_multiply_vector_vector_d3(double a[3],
				   double b[3])
{
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/* m^-1 */
/* ruby code for auto generating */
/* 3.times {|i| 3.times {|j| */
/*       puts "m[#{j}*3+#{i}]=(a[#{(i+1)%3}*3+#{(j+1)%3}]*a[#{(i+2)%3}*3+#{(j+2)%3}] */
/*	 -a[#{(i+1)%3}*3+#{(j+2)%3}]*a[#{(i+2)%3}*3+#{(j+1)%3}])/det;" */
/* }} */
void mat_inverse_matrix_d3_flatten(double m[9],
			  double a[9])
{
  double det;
  double c[9];
  det = mat_get_determinant_d3(a);

  c[0] = (a[4] * a[8] - a[5] * a[7]) / det;
  c[3] = (a[5] * a[6] - a[3] * a[8]) / det;
  c[6] = (a[3] * a[7] - a[4] * a[6]) / det;
  c[1] = (a[7] * a[2] - a[8] * a[1]) / det;
  c[4] = (a[8] * a[0] - a[6] * a[2]) / det;
  c[7] = (a[6] * a[1] - a[7] * a[0]) / det;
  c[2] = (a[1] * a[5] - a[2] * a[4]) / det;
  c[5] = (a[2] * a[3] - a[0] * a[5]) / det;
  c[8] = (a[0] * a[4] - a[1] * a[3]) / det;
  mat_copy_matrix_d3(m, c);
}

/* m = b^-1 a b */
void mat_get_similar_matrix_d3_flatten(double m[9],
			      double a[9],
			      double b[9])
{
  double c[9];
  mat_inverse_matrix_d3_flatten(c, b);
//   mat_multiply_matrix_d3_flatten(m, a, b);
//   mat_multiply_matrix_d3_flatten(m, c, m);
  mat_multiply_matrix_d3_flatten(m, a, c);
  mat_multiply_matrix_d3_flatten(m, b, m);
}
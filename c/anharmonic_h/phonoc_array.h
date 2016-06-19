#ifndef __phonoc_array_H__
#define __phonoc_array_H__

#include <Python.h>
#include <numpy/arrayobject.h>
#include <lapacke.h>
#define MAX_NUM_DIM 20

/* It is assumed that number of dimensions is known for each array. */
typedef struct {
  int dims[MAX_NUM_DIM];
  int *data;
} Iarray;

typedef struct {
  int dims[MAX_NUM_DIM];
  double *data;
} Darray;

typedef struct {
  int dims[MAX_NUM_DIM];
  lapack_complex_double *data;
} Carray;

Iarray* convert_to_iarray(const PyArrayObject* npyary);
Darray* convert_to_darray(const PyArrayObject* npyary);
Carray* convert_to_carray(const PyArrayObject* npyary);
void mat_add_matrix_d3_flatten(double m[9],double a[9], double b[9]);
void mat_add_vector_d3_flatten(double m[3], double a[3], double b[3]);
double mat_multiply_vector_vector_d3(double a[3],double b[3]);
double mat_multiply_vector_vector_dn(double a[],double b[], int dim);
void mat_copy_vector_dn(double a[],double b[],int dim);
void mat_copy_vector_i3_flatten(int a[3], int b[3]);
void mat_copy_matrix_id3_flatten(double a[9], int b[9]);
void mat_copy_matrix_i3_flatten(int a[9], int b[9]);
void mat_multiply_matrix_d3_flatten(double m[9], double a[9], double b[9]);
void mat_multiply_matrix_vector_d3_flatten(double v[3],double a[9],double b[3]);
void mat_vector_outer_product_flatten(double v[6], double a[3], double b[3]);
void mat_inverse_matrix_d3_flatten(double m[9], double a[9]);
void mat_get_similar_matrix_d3_flatten(double m[9],double a[9],double b[9]);
#endif

/* tetrahedron_method.h */
/* Copyright (C) 2014 Atsushi Togo */

#ifndef __tetrahedron_method_H__
#define __tetrahedron_method_H__
#include "mathfunc.h"

void thm_get_relative_grid_address(int relative_grid_address[24][4][3],
				   SPGCONST double rec_lattice[3][3]);
void thm_get_all_relative_grid_address(int relative_grid_address[4][24][4][3]);
void thm_get_integration_weight(double iw[120][4],
                  const double omega,
				  SPGCONST double tetrahedra_omegas[120][4],
				  const int is_linear,
				  const char function);
void
thm_get_integration_weight_at_omegas(double integration_weights[][120][4],
				     const int num_omegas,
				     const double *omegas,
				     SPGCONST double tetrahedra_omegas[120][4],
				     const int is_linear,
				     const char function);
#endif

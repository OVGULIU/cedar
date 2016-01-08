#ifndef BOXMG_2D_UTIL_MPI_H
#define BOXMG_2D_UTIL_MPI_H

#include <mpi.h>
#include "boxmg/types.h"
#include "boxmg/2d/types.h"
#include "boxmg/mpi/grid_topo.h"


namespace boxmg { namespace bmg2d { namespace util { namespace mpi {

bool has_boundary(grid_topo & grid, bmg2d::dir dir);

}}}}
#endif

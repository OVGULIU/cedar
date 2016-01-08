#ifndef BOXMG_2D_UTIL_TOPO_H
#define BOXMG_2D_UTIL_TOPO_H

#include <mpi.h>
#include "boxmg/types.h"
#include "boxmg/mpi/grid_topo.h"

namespace boxmg { namespace bmg2d { namespace util {

			topo_ptr create_topo(MPI_Comm comm, len_t nx, len_t ny);

}}}
#endif

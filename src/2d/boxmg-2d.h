#ifndef boxmg_2d_h
#define boxmg_2d_h

#include "boxmg-2d/core/array.h"
#include "boxmg-2d/core/boundary_iterator.h"
#include "boxmg-2d/core/discrete_op.h"
#include "boxmg-2d/core/grid_func.h"
#include "boxmg-2d/core/grid_stencil.h"
#include "boxmg-2d/core/mpi/grid_func.h"
#include "boxmg-2d/core/mpi/grid_stencil.h"
#include "boxmg-2d/core/mpi/grid_topo.h"
#include "boxmg-2d/core/mpi/stencil_op.h"
#include "boxmg-2d/core/relax_stencil.h"
#include "boxmg-2d/core/stencil.h"
#include "boxmg-2d/core/stencil_op.h"
#include "boxmg-2d/core/types.h"
#include "boxmg-2d/inter/mpi/prolong_op.h"
#include "boxmg-2d/inter/mpi/restrict_op.h"
#include "boxmg-2d/inter/prolong_op.h"
#include "boxmg-2d/inter/restrict_op.h"
#include "boxmg-2d/inter/types.h"
#include "boxmg-2d/kernel/factory.h"
#include "boxmg-2d/kernel/galerkin_prod.h"
#include "boxmg-2d/kernel/halo.h"
#include "boxmg-2d/kernel/interp.h"
#include "boxmg-2d/kernel/mpi/factory.h"
#include "boxmg-2d/kernel/name.h"
#include "boxmg-2d/kernel/registry.h"
#include "boxmg-2d/kernel/relax.h"
#include "boxmg-2d/kernel/residual.h"
#include "boxmg-2d/kernel/restrict.h"
#include "boxmg-2d/kernel/setup_cg_boxmg.h"
#include "boxmg-2d/kernel/setup_cg_lu.h"
#include "boxmg-2d/kernel/setup_interp.h"
#include "boxmg-2d/kernel/setup_nog.h"
#include "boxmg-2d/kernel/setup_relax.h"
#include "boxmg-2d/kernel/solve_cg.h"
#include "boxmg-2d/solver/boxmg.h"
#include "boxmg-2d/solver/cg.h"
#include "boxmg-2d/solver/level.h"
#include "boxmg-2d/solver/mpi/boxmg.h"
#include "boxmg-2d/solver/multilevel.h"
#include "boxmg-2d/util/mpi_grid.h"
#include "boxmg-2d/util/topo.h"

#endif // boxmg_2d_h

#ifndef CEDAR_2D_MPI_KERNEL_MANAGER_H
#define CEDAR_2D_MPI_KERNEL_MANAGER_H

#include <cedar/2d/mpi/types.h>
#include <cedar/kernel_manager.h>
#include <cedar/kernel_registry.h>

namespace cedar { namespace cdr2 { namespace mpi {

using kman_ptr = std::shared_ptr<kernel_manager<klist<stypes, exec_mode::mpi>>>;

using point_relax = kernels::point_relax<stypes>;
template<relax_dir rdir>
using line_relax = kernels::line_relax<stypes, rdir>;
using coarsen_op = kernels::coarsen_op<stypes>;
using interp_add = kernels::interp_add<stypes>;
using restriction = kernels::restriction<stypes>;
using matvec = kernels::matvec<stypes>;
using residual = kernels::residual<stypes>;
using setup_interp = kernels::setup_interp<stypes>;
using halo_exchange = kernels::halo_exchange<stypes>;
using setup_nog = kernels::setup_nog<stypes>;


kman_ptr build_kernel_manager(config & conf);
kman_ptr build_kernel_manager(std::shared_ptr<kernel_params> params);


}}}
#endif

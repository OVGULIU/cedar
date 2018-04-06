#ifndef CEDAR_3D_REDIST_SETUP_REDIST_H
#define CEDAR_3D_REDIST_SETUP_REDIST_H

#include <cedar/kernel_params.h>
#include <cedar/3d/mpi/redist_solver.h>

namespace cedar { namespace cdr3 {

template<class inner_solver, class T>
std::function<void(mpi::grid_func &, const mpi::grid_func &)>
	create_redist_solver(std::shared_ptr<T> kernels,
                     config::reader & conf,
                     mpi::stencil_op<xxvii_pt> & cop,
                     std::shared_ptr<config::reader> cg_conf,
                     std::array<int, 3> & choice)
{
	auto params = build_kernel_params(conf);
	using rsolver = mpi::redist_solver<inner_solver>;
	auto cg_bmg = std::make_shared<rsolver>(cop,
	                                        kernels->get_halo_exchanger().get(),
	                                        cg_conf,
	                                        choice);

	auto coarse_solver = [=](mpi::grid_func & x, const mpi::grid_func & b)
	{
		cg_bmg->solve(x, b);
		if (params->per_mask())
			kernels->halo_exchange(x);
	};

	return coarse_solver;
}

}}

#endif

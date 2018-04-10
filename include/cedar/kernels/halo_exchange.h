#ifndef CEDAR_HALO_EXCHANGE_H
#define CEDAR_HALO_EXCHANGE_H

#include <cedar/halo_exchanger_base.h>
#include <cedar/mpi/grid_topo.h>
#include <cedar/kernel.h>

namespace cedar { namespace kernels {

template<class solver_types>
class halo_exchange : public kernel<solver_types>, public halo_exchanger_base
{
public:
	template<class sten>
		using stencil_op = typename kernel<solver_types>::template stencil_op<sten>;
	using comp_sten = typename kernel<solver_types>::comp_sten;
	using full_sten = typename kernel<solver_types>::full_sten;
	using grid_func = typename kernel<solver_types>::grid_func;

	const std::string name = "halo exchange";

	virtual void setup(std::vector<topo_ptr> topos) = 0;
	virtual void run(stencil_op<comp_sten> & so) = 0;
	virtual void run(stencil_op<full_sten> & so) = 0;
	virtual void run(grid_func & gf) = 0;

	virtual void exchange_func(int k, real_t *gf) = 0;
	virtual void exchange_sten(int k, real_t *so) = 0;
	virtual aarray<int, len_t, 2> & leveldims(int k) = 0;
	virtual len_t * datadist(int k, int grid) = 0;
	virtual MPI_Comm linecomm(int k) = 0;
};

}}

#endif
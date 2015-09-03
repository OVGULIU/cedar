#include "kernel/fortran/mpi/BMG_workspace_c.h"
#include "boxmg-common.h"
#include "core/mpi/grid_topo.h"

#include "topo.h"

extern "C"
{
	bmg2_topo bmg2_topo_create(MPI_Comm comm,
	                           unsigned int ngx,
	                           unsigned int ngy,
	                           unsigned int nlx[],
	                           unsigned int nly[],
	                           int nprocx,
	                           int nprocy)
	{
		using namespace boxmg;
		using namespace boxmg::bmg2d;
		int rank;

		MPI_Comm_rank(comm, &rank);

		std::shared_ptr<core::mpi::GridTopo> *grid_ptr;
		auto igrd = std::make_shared<std::vector<len_t>>(NBMG_pIGRD);
		auto grid = std::make_shared<core::mpi::GridTopo>(igrd, 0, 1);

		grid->comm = comm;

		grid->nproc(0) = nprocx;
		grid->nproc(1) = nprocy;

		grid->coord(0) = rank % nprocx;
		grid->coord(1) = rank / nprocx;

		grid->is(0) = 1; grid->is(1) = 1;
		for (auto i : range(grid->coord(0))) {
			grid->is(0) += nlx[i];
		}

		for (auto j : range(grid->coord(1))) {
			grid->is(1) += nly[j];
		}

		grid->nglobal(0) = ngx + 2;
		grid->nglobal(1) = ngy + 2;

		grid->nlocal(0) = nlx[grid->coord(0)] + 2;
		grid->nlocal(1) = nly[grid->coord(1)] + 2;

		grid_ptr = new std::shared_ptr<core::mpi::GridTopo>(std::move(grid));
		return reinterpret_cast<bmg2_topo>(grid_ptr);
	}
}

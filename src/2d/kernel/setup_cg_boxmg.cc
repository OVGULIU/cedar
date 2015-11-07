#include <mpi.h>
#include "halo.h"
#include "fortran/mpi/BMG_workspace_c.h"
#include "core/mpi/stencil_op.h"
#include "boxmg-common.h"

#include "setup_cg_boxmg.h"


extern "C" {
	using namespace boxmg;
	void BMG2_SymStd_SETUP_cg_boxmg(len_t ii, len_t jj, len_t ngx, len_t ngy,
	                                len_t igs, len_t jgs,
	                                real_t *so, int nstncl, int nog,
	                                real_t *ws, len_t nmsgr, real_t *so_ser,
	                                int nproci, int nprocj, int nproc, int myproc,
	                                int *proc_grid, int *proc_coord, len_t *loc_array_size,
	                                MPI_Fint MPICOMM);
}


namespace boxmg { namespace bmg2d { namespace kernel {

namespace impls
{
	void setup_cg_boxmg(const StencilOp & so,
	                    std::shared_ptr<solver> *bmg)
	{
		int nstencil, nog, rank;
		mpi::StencilOp & sod = const_cast<mpi::StencilOp&>(dynamic_cast<const mpi::StencilOp&>(so));
		mpi::GridTopo & topo = sod.grid();
		MsgCtx *ctx = (MsgCtx*) sod.halo_ctx;
		auto & sten = sod.stencil();

		StencilOp so_ser(topo.nglobal(0)-2, topo.nglobal(1)-2);

		if (sten.five_pt()) nstencil = 3;
		else nstencil = 5;

		nog = topo.nlevel();

		MPI_Comm_rank(topo.comm, &rank);
		rank++; // 1 based indexing

		MPI_Fint fcomm = MPI_Comm_c2f(topo.comm);

		len_t local_arr_ptr = ctx->pMSG(ipL_MSG_LocalArraySize,0) - 1;  // 1 vs 0 based indexing

		BMG2_SymStd_SETUP_cg_boxmg(topo.nlocal(0), topo.nlocal(1),
		                           topo.nglobal(0), topo.nglobal(1),
		                           topo.is(0), topo.is(1),
		                           sod.data(), nstencil, nog,
		                           ctx->msg_buffer.data(),
		                           ctx->msg_buffer.size(),
		                           so_ser.data(), topo.nproc(0),
		                           topo.nproc(1), topo.nproc(), rank,
		                           ctx->proc_grid.data(),
		                           ctx->proc_coord.data(),
		                           &ctx->msg_geom.data()[local_arr_ptr],
		                           fcomm);

		*bmg = std::make_shared<solver>(std::move(so_ser));
	}
}

}}}

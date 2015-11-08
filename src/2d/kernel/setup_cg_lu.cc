#include "setup_cg_lu.h"
#include "halo.h"
#include "fortran/mpi/BMG_workspace_c.h"

extern "C" {
	using namespace boxmg;
	void BMG2_SymStd_SETUP_cg_LU(real_t*, len_t*, len_t*, int*, real_t*, len_t*,len_t*,int*);
	void MPI_BMG2_SymStd_SETUP_cg_LU(real_t* SO, len_t II, len_t JJ, int nstncl,
	                                 len_t iGs, len_t jGs, len_t nGx, len_t nGy,
	                                 real_t *ABD, len_t NABD1, len_t NABD2,
	                                 real_t *ws, len_t NMSGr, int nproci, int nprocj, int nproc,
	                                 int myproc, int *proc_grid, int *proc_coord, len_t *locarrsize,
	                                 int MPICOMM);
}

namespace boxmg { namespace bmg2d { namespace kernel {

namespace impls
{
	using namespace boxmg::bmg2d::core;
	void setup_cg_lu(const StencilOp & so,
	                 grid_func & ABD)
	{
		len_t nx, ny;
		int nstencil;
		len_t nabd1, nabd2;
		int ibc = 0;

		const GridStencil & so_sten = so.stencil();
		StencilOp & sod = const_cast<StencilOp&>(so);

		nx = so_sten.len(0);
		ny = so_sten.len(1);

		if (so_sten.five_pt()) nstencil = 3;
		else nstencil = 5;

		nabd1 = ABD.len(0);
		nabd2 = ABD.len(1);

		BMG2_SymStd_SETUP_cg_LU(sod.data(), &nx, &ny, &nstencil,
		                        ABD.data(), &nabd1, &nabd2, &ibc);
	}


	void mpi_setup_cg_lu(const StencilOp & so,
	                     grid_func & ABD)
	{
		mpi::StencilOp & copd = const_cast<mpi::StencilOp&>(dynamic_cast<const mpi::StencilOp&>(so));

		GridStencil & csten = copd.stencil();
		mpi::GridTopo & topo = copd.grid();
		MsgCtx *ctx = (MsgCtx*) copd.halo_ctx;
		int nstencil;

		if (csten.five_pt()) {
			nstencil = 3;
		} else {
			nstencil = 5;
		}

		int rank;
		MPI_Comm_rank(topo.comm, &rank);
		rank++; // 1 based indexing

		len_t local_arr_ptr = ctx->pMSG(ipL_MSG_LocalArraySize,0) - 1;  // 1 vs 0 based indexing

		MPI_Fint fcomm = MPI_Comm_c2f(topo.comm);

		MPI_BMG2_SymStd_SETUP_cg_LU(csten.data(), csten.len(0), csten.len(1),
		                            nstencil, topo.is(0), topo.is(1),
		                            topo.nglobal(0), topo.nglobal(1),
		                            ABD.data(), ABD.len(0), ABD.len(1),
		                            ctx->msg_buffer.data(),
		                            ctx->msg_buffer.size(),
		                            topo.nproc(0), topo.nproc(1), topo.nproc(),
		                            rank,
		                            ctx->proc_grid.data(),
		                            ctx->proc_coord.data(),
		                            &ctx->msg_geom.data()[local_arr_ptr],
		                            fcomm);

	}
}

}}}

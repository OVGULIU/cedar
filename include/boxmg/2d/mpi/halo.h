#ifndef BOXMG_2D_KERNEL_HALO_H
#define BOXMG_2D_KERNEL_HALO_H

#include <boxmg/array.h>
#include <boxmg/2d/mpi/grid_topo.h>
#include <boxmg/2d/mpi/stencil_op.h>
#include <boxmg/2d/mpi/grid_func.h>

namespace boxmg { namespace bmg2d { namespace kernel {
namespace impls
{
	namespace mpi = boxmg::bmg2d::mpi;
	struct MsgCtx
	{
		MsgCtx(mpi::grid_topo & topo);
		array<int,int,2> pMSG; // these should be int, len_t
		array<int,int,2> pLS;
		array<int,int,2> pMSGSO;
		std::vector<len_t> msg_geom;
		array<int,int,2> proc_grid;
		std::vector<int> proc_coord;
		std::vector<int> dimxfine;
		std::vector<int> dimyfine;
		array<int,int,2> dimx;
		array<int,int,2> dimy;
		std::vector<real_t> msg_buffer;
		int pSI_MSG;
		int p_NLx_kg, p_NLy_kg;
		MPI_Comm xlinecomm;
		MPI_Comm ylinecomm;
		/* std::vector<len_t> iworkmsg; */
		/* int *iworkmsg[nmsgi]; */
		/* int nmsgi; */
		/* int pmsg[nbmg_pmsg,nog]; */
		/* int msgbuffer[nmsgr]; */
		/* int nmsgr; */
	};

	void setup_msg(mpi::grid_topo &topo, void **msg_ctx);
	void msg_exchange(mpi::grid_func & f);
	void msg_stencil_exchange(mpi::stencil_op & so);
}
}}}

#endif

#ifndef CEDAR_3D_INTER_MPI_SETUP_INTERP_H
#define CEDAR_3D_INTER_MPI_SETUP_INTERP_H

#include <type_traits>

#include <cedar/kernel_params.h>
#include <cedar/halo_exchanger.h>
#include <cedar/3d/mpi/stencil_op.h>
#include <cedar/3d/inter/mpi/prolong_op.h>
#include <cedar/2d/ftn/BMG_parameters_c.h>
#include <cedar/halo_exchanger.h>

extern "C" {
	using namespace cedar;
	void MPI_BMG3_SymStd_SETUP_interp_OI(int kgf, int kgc, real_t *so, real_t *soc,
	                                     real_t *ci, len_t iif, len_t jjf, len_t kkf,
	                                     len_t iic, len_t jjc, len_t kkc,
	                                     int nog, int ifd, int nstencil, int irelax, real_t *yo,
	                                     int nogm, len_t *IGRD, int jpn, void *halof);
	void BMG_get_bc(int, int*);
}

namespace cedar { namespace cdr3 { namespace kernel {

namespace impls
{
	namespace mpi = cedar::cdr3::mpi;

	template<class sten>
		void store_fine_op(mpi::stencil_op<sten> & fop,
		                   inter::mpi::prolong_op & P);

	template<>
		inline void store_fine_op(mpi::stencil_op<seven_pt> & fop,
		                          inter::mpi::prolong_op & P)
	{
		P.fine_op_seven = &fop;
		P.fine_is_seven = true;
	}

	template<>
		inline void store_fine_op(mpi::stencil_op<xxvii_pt> & fop,
		                          inter::mpi::prolong_op & P)
	{
		P.fine_op_xxvii = &fop;
		P.fine_is_seven = false;
	}

	template<class sten>
	void mpi_setup_interp(const kernel_params & params,
	                      halo_exchanger_base * halof,
	                      const mpi::stencil_op<sten> & fop,
	                      const mpi::stencil_op<xxvii_pt> & cop,
	                      inter::mpi::prolong_op & P)
	{
		int ifd, nstencil;
		int kc, nog, kf;

		auto & fopd = const_cast<mpi::stencil_op<sten>&>(fop);
		auto & copd = const_cast<mpi::stencil_op<xxvii_pt>&>(cop);
		grid_topo & topo = fopd.grid();

		store_fine_op(fopd, P);

		nstencil = stencil_ndirs<sten>::value;

		if (std::is_same<sten, seven_pt>::value)
			ifd = 1;
		else
			ifd = 0;

		kc = topo.level();
		nog = topo.nlevel();
		kf = kc + 1;

		// TODO: preallocate this
		array<real_t, 4> yo(fop.len(0), fop.len(1), 2, 14);
		int jpn;
		BMG_get_bc(params.per_mask(), &jpn);

		MPI_BMG3_SymStd_SETUP_interp_OI(kf, kc, fopd.data(), copd.data(),
		                                P.data(), fop.len(0), fop.len(1), fop.len(2),
		                                cop.len(0), cop.len(1), cop.len(2),
		                                nog, ifd, nstencil, BMG_RELAX_SYM,
		                                yo.data(), nog, topo.IGRD(),
		                                jpn, halof);
	}
}

}}}

#endif

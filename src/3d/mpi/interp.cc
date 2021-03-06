#include <cedar/2d/ftn/mpi/BMG_parameters_c.h>

#include <cedar/3d/mpi/interp.h>


extern "C" {
	using namespace cedar;
	void MPI_BMG3_SymStd_interp_add(int kcg, int kfg,
	                                real_t *q, real_t *qc, real_t *res, real_t *so,
	                                int nstncl, real_t *ci,
	                                len_t iic, len_t jjc, len_t kkc,
	                                len_t iif, len_t jjf, len_t kkf,
	                                len_t igs, len_t jgs, len_t kgs, void *halof);
}


namespace cedar { namespace cdr3 { namespace mpi {

void interp_f90::run(const prolong_op & P,
                     const grid_func & coarse,
                     const grid_func & residual,
                     grid_func & fine)
{
		int nstencil, kf, kc;

		prolong_op & Pd = const_cast<prolong_op&>(P);
		grid_func & coarsed = const_cast<grid_func&>(coarse);
		grid_func & res = const_cast<grid_func&>(residual);
		grid_topo & topo = Pd.grid();

		real_t * fop_data;
		topo_ptr topof;
		if (Pd.fine_is_seven) {
			nstencil = 4;
			fop_data = Pd.fine_op_seven->data();
			topof = Pd.fine_op_seven->grid_ptr();
		} else {
			nstencil = 14;
			fop_data = Pd.fine_op_xxvii->data();
			topof = Pd.fine_op_xxvii->grid_ptr();
		}

		kc = topo.level() + 1;
		kf = kc + 1;

		MPI_BMG3_SymStd_interp_add(kc, kf,
		                           fine.data(), coarsed.data(), res.data(),
		                           fop_data, nstencil, Pd.data(),
		                           coarsed.len(0), coarsed.len(1), coarsed.len(2),
		                           fine.len(0), fine.len(1), fine.len(2),
		                           topof->is(0), topof->is(1), topof->is(2),
		                           halof);
}

}}}

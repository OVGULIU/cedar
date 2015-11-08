#ifndef BOXMG_2D_CORE_SOLVER_H
#define BOXMG_2D_CORE_SOLVER_H

#include <array>

#include "multilevel.h"
#include "level.h"
#include "core/stencil_op.h"
#include "core/relax_stencil.h"
#include "inter/prolong_op.h"
#include "inter/restrict_op.h"
#include "kernel/registry.h"

namespace boxmg { namespace bmg2d {

struct BoxMGLevel : Level
{
BoxMGLevel(StencilOp&& A, inter::prolong_op&& P) : /*Level(A,P),*/
	A(std::move(A)), P(std::move(P)), SOR({{RelaxStencil(), RelaxStencil()}}) { R.associate(&P); }
	StencilOp    A;
	inter::prolong_op   P;
	inter::restrict_op  R;
	grid_func     x;
	grid_func     res;
	grid_func     b;
	std::array<RelaxStencil, 2> SOR;
};

class solver: public MultiLevel<BoxMGLevel>
{
public:
	solver(StencilOp&& fop);
	~solver() {delete[] bbd;};
	int compute_num_levels(StencilOp & fop);
	void add_level(StencilOp& fop, int num_levels);
	std::shared_ptr<kernel::Registry> kernel_registry();

private:
	grid_func ABD;
	real_t *bbd;
};

}}

#endif

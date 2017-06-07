#ifndef CEDAR_3D_SOLVER_H
#define CEDAR_3D_SOLVER_H

#include <array>

#include <cedar/multilevel.h>
#include <cedar/level.h>
#include <cedar/3d/level_container.h>
#include <cedar/2d/solver.h>
#include <cedar/3d/types.h>
#include <cedar/3d/grid_func.h>
#include <cedar/3d/stencil_op.h>
#include <cedar/3d/relax_stencil.h>
#include <cedar/3d/inter/prolong_op.h>
#include <cedar/3d/inter/restrict_op.h>
#include <cedar/3d/kernel/registry.h>


namespace cedar { namespace cdr3 {


template<class sten>
	struct level3 : public level<sten, stypes>
{
	using parent = level<sten, stypes>;
level3(len_t nx, len_t ny, len_t nz) : parent::level(nx, ny, nz)
	{
		this->SOR = {{relax_stencil(nx, ny, nz),
		              relax_stencil(nx, ny, nz)}};
		this->R.associate(&this->P);
	}
level3(stencil_op<sten> & A) : parent::level(A)
	{
		this->res = grid_func(A.shape(0), A.shape(1), A.shape(2));
		this->SOR = {{relax_stencil(A.shape(0), A.shape(1), A.shape(2)),
		              relax_stencil(A.shape(0), A.shape(1), A.shape(2)),}};
	}
	/* std::vector<std::unique_ptr<::cedar::cdr2::solver>> planes_xy; */
	/* std::vector<std::unique_ptr<::cedar::cdr2::solver>> planes_xz; */
	/* std::vector<std::unique_ptr<::cedar::cdr2::solver>> planes_yz; */
};

template<class fsten>
	class solver : public multilevel<level_container<level3, fsten>,
	typename kernel::registry::parent, fsten, solver<fsten>>
{
public:
	using parent = multilevel<level_container<level3, fsten>,
		typename kernel::registry::parent, fsten, solver<fsten>>;
solver(stencil_op<fsten> & fop) : parent::multilevel(fop)
	{
		this->kreg = std::make_shared<kernel::registry>(*(this->conf));
		parent::setup(fop);
	}
solver(stencil_op<fsten> & fop,
       std::shared_ptr<config::reader> conf) :
	parent::multilevel(fop, conf)
	{
		this->kreg = std::make_shared<kernel::registry>(*(this->conf));
		parent::setup(fop);
	}
	~solver() { delete[] this->bbd; }
	std::size_t compute_num_levels(stencil_op<fsten> & fop)
	{
		float nxc, nyc, nzc;
		int ng = 0;
		int min_coarse = this->conf->template get<int>("solver.min-coarse", 3);

		auto nx = fop.shape(0);
		auto ny = fop.shape(1);
		auto nz = fop.shape(2);

		do {
			ng++;
			nxc = (nx-1)/(1<<ng) + 1;
			nyc = (ny-1)/(1<<ng) + 1;
			nzc = (nz-1)/(1<<ng) + 1;
		} while(std::min({nxc,nyc,nzc}) >= min_coarse);

		return ng;
	}


	void setup_space(std::size_t nlevels)
	{
		this->levels.init(nlevels);
		auto params = build_kernel_params(*this->conf);

		for (auto i : range<std::size_t>(nlevels - 1)) {
			// TODO: remove copy-paste coding
			if (i == 0) {
				auto & fop = this->levels.template get<fsten>(i).A;

				auto nx = fop.shape(0);
				auto ny = fop.shape(1);
				auto nz = fop.shape(2);
				auto nxc = (nx-1)/2. + 1;
				auto nyc = (ny-1)/2. + 1;
				auto nzc = (nz-1)/2. + 1;

				this->levels.add(nxc, nyc, nzc);

				log::debug << "Created coarse grid with dimensions: " << nxc << ", " <<
					nyc << ", " << nzc << std::endl;
			} else {
				auto & fop = this->levels.template get<fsten>(i).A;

				auto nx = fop.shape(0);
				auto ny = fop.shape(1);
				auto nz = fop.shape(2);
				auto nxc = (nx-1)/2. + 1;
				auto nyc = (ny-1)/2. + 1;
				auto nzc = (nz-1)/2. + 1;

				this->levels.add(nxc, nyc, nzc);

				log::debug << "Created coarse grid with dimensions: " << nxc << ", " <<
					nyc << ", " << nzc << std::endl;
			}
		}


		auto & cop = this->levels.get(this->levels.size()-1).A;
		auto nxc = cop.shape(0);
		auto nyc = cop.shape(1);
		auto nzc = cop.shape(2);
		len_t abd_len_0 = nxc*(nyc+1)+2;
		if (params->periodic[0] or params->periodic[1] or params->periodic[2])
			abd_len_0 = nxc*nyc*nzc;
		this->ABD = grid_func(abd_len_0, nxc*nyc*nzc, 0);
		this->bbd = new real_t[this->ABD.len(1)];
	}
	/* virtual void setup_relax_plane(stencil_op & sop, bmg_level & level) override; */
	/* virtual void relax_plane(const stencil_op & so, grid_func & x, */
	/*                          const grid_func & b, cycle::Dir cdir, */
	/*                          bmg_level & level) override; */
	void give_op(std::unique_ptr<stencil_op<fsten>> fop) {fop_ref = std::move(fop);}

protected:
	std::unique_ptr<stencil_op<fsten>> fop_ref;
};


}}

#endif


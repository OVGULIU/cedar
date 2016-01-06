#include <boxmg/kernel.h>
#include <boxmg/kernel_name.h>
#include <boxmg/3d/residual.h>
#include <boxmg/3d/relax/setup_relax.h>
#include <boxmg/3d/inter/setup_interp.h>
#include <boxmg/3d/inter/galerkin_prod.h>
#include <boxmg/3d/relax/relax.h>

#include <boxmg/3d/kernel/factory.h>

namespace boxmg { namespace bmg3 { namespace kernel {

namespace factory
{
	namespace name = boxmg::kernel_name;

	std::shared_ptr<registry> from_config(config::Reader & conf)
	{
		auto kreg = std::make_shared<registry>();

		kreg->add(name::residual, "fortran",
		          boxmg::kernel<const stencil_op &,
		          const grid_func &,
		          const grid_func &,
		          grid_func&>(impls::residual));

		kreg->add(name::setup_interp, "fortran",
		          boxmg::kernel<int, int, int,
		          const stencil_op &,
		          const stencil_op &,
		          inter::prolong_op &>(impls::setup_interp));

		kreg->add(name::setup_relax, "fotran-rbgs-point",
		          boxmg::kernel<const stencil_op&,
		          relax_stencil&>(impls::setup_rbgs_point));

		kreg->add(name::galerkin_prod, "fortran-ex",
		         boxmg::kernel<int,int,int,
		         const inter::prolong_op&,
		         const stencil_op&,
		         stencil_op&>(impls::galerkin_prod));

		kreg->add(name::relax, "fortran-rbgs-point",
		          boxmg::kernel<const stencil_op&,
		          grid_func&,
		          const grid_func&,
		          const relax_stencil&,
		          cycle::Dir>(impls::relax_rbgs_point));


		std::vector<std::tuple<std::string, std::string, std::string>> defaults = {
			std::make_tuple(name::residual, "kernels.residual", "fortran"),
			std::make_tuple(name::galerkin_prod, "kernels.galerkin-prod", "fortran-ex"),
			std::make_tuple(name::setup_interp, "kernels.setup-interp", "fortran"),
			std::make_tuple(name::setup_relax, "kernels.setup-relax", "fortran-rbgs-point"),
			std::make_tuple(name::relax, "kernels.relax", "fortran-rbgs-point")
		};

		for (auto&& v : defaults) {
			std::string kname = conf.get<std::string>(std::get<1>(v), std::get<2>(v));
			log::info << "Using '" + kname + " ' for " <<  std::get<0>(v) << "." << std::endl;
			kreg->set(std::get<0>(v), kname);
		}

		return kreg;
	}
}

}}}

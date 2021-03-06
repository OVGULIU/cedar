#include <mpi.h>
#include <iostream>
#include <memory>
#include <math.h>

#include <cedar/types.h>
#include <cedar/2d/mpi/grid_func.h>
#include <cedar/2d/mpi/stencil_op.h>
#include <cedar/2d/util/topo.h>
#include <cedar/kernel_params.h>
#include <cedar/2d/mpi/kernel_manager.h>

#include <cedar/util/time_log.h>

#include "per_halo.h"


int main(int argc, char *argv[])
{
	using namespace cedar;
	using namespace cedar::cdr2;

	int provided;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

	config conf;
	log::init(conf);
	log::status << "Beginning test" << std::endl;
	auto grid = util::create_topo(conf);

	mpi::grid_func b(grid);
	mpi::stencil_op<five_pt> so(grid);

	fill_gfunc(b);
	fill_stencil(so);

	// auto exchanger_str = conf.get<std::string>("halo-exchanger", "msg");

	mpi::solver<five_pt> slv(so);
	auto kman = slv.get_kernels();

	draw(b, "before-0");
	draw_so(so, "before-0");
	kman->run<mpi::halo_exchange>(b);
	kman->run<mpi::halo_exchange>(so);
	draw(b, "after-0");
	draw_so(so, "after-0");


	for (std::size_t lvl = 1; lvl < slv.nlevels(); lvl++) {
		auto & level = slv.levels.get(lvl);
		fill_gfunc(level.b);
		fill_stencil(level.A);
		draw(level.b, "before-" + std::to_string(lvl));
		draw_so(level.A, "before-" + std::to_string(lvl));
		kman->run<mpi::halo_exchange>(level.b);
		kman->run<mpi::halo_exchange>(level.A);
		draw(level.b, "after-" + std::to_string(lvl));
		draw_so(level.A, "after-" + std::to_string(lvl));
	}

	log::status << "Finished Test" << std::endl;

	MPI_Finalize();
	return 0;
}

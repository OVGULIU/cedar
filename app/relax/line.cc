#include <iostream>
#include <chrono>
#include <thread>

#include <boxmg/2d/mpi/gallery.h>

#include <boxmg/types.h>
#include <boxmg/util/time_log.h>
#include <boxmg/2d/util/topo.h>
#include <boxmg/2d/util/mpi_grid.h>


static void run_relax(boxmg::topo_ptr grid, int nrelax)
{
	using namespace boxmg;
	using namespace boxmg::bmg2d;

	timer_init(MPI_COMM_WORLD);

	log::status << "Running line-xy relaxation " << nrelax << " times" << std::endl;

	auto so = mpi::gallery::poisson(grid);
	so.stencil().five_pt() = false;
	mpi::grid_func b(grid);
	mpi::grid_func x(grid);
	mpi::grid_func res(grid);

	b.set(0);
	x.set(1);

	// setup halo
	auto kreg = so.get_registry();
	void *halo_ctx;
	kreg->halo_setup(so.grid(), &halo_ctx);
	so.halo_ctx = halo_ctx;
	kreg->halo_stencil_exchange(so);
	b.halo_ctx = halo_ctx;
	x.halo_ctx = halo_ctx;
	res.halo_ctx = halo_ctx;

	std::array<bmg2d::relax_stencil,3> SOR{{
     			bmg2d::relax_stencil(so.stencil().shape(0), so.stencil().shape(1)),
				bmg2d::relax_stencil(so.stencil().shape(0), so.stencil().shape(1)),
				bmg2d::relax_stencil(so.stencil().shape(0), so.stencil().shape(1))}};


	kreg->setup_relax(so, SOR[2]);

	MPI_Barrier(MPI_COMM_WORLD);

	timer_begin("setup-lines");
	kreg->setup_relax_x(so, SOR[0]);
	kreg->setup_relax_y(so, SOR[1]);
	timer_end("setup-lines");

	timer_begin("relax-lines");
	for (auto i = 0; i < nrelax; i++) {
		kreg->relax_lines_x(so, x, b, SOR[0], res, cycle::Dir::DOWN);
		kreg->relax_lines_y(so, x, b, SOR[1], res, cycle::Dir::DOWN);

		kreg->relax_lines_y(so, x, b, SOR[1], res, cycle::Dir::UP);
		kreg->relax_lines_x(so, x, b, SOR[0], res, cycle::Dir::UP);
	}
	timer_end("relax-lines");

	timer_begin("relax-point");
	for (auto i = 0; i < nrelax; i++) {
		kreg->relax(so, x, b, SOR[2], cycle::Dir::DOWN);
		kreg->relax(so, x, b, SOR[2], cycle::Dir::UP);
	}
	timer_end("relax-point");

	timer_save("timings.json");
}

int main(int argc, char *argv[])
{
	using namespace boxmg;
	using namespace boxmg::bmg2d;


	int provided;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

	config::reader conf;

	auto nrelax = conf.get<int>("nrelax");
	auto ndofs = conf.getvec<len_t>("grid.n");
	auto nx = ndofs[0];
	auto ny = ndofs[1];
	topo_ptr grid;

	auto nprocs = conf.getvec<int>("grid.np");
	int npx = 0;
	int npy = 0;
	if (nprocs.size() >= 2) {
		npx = nprocs[0];
		npy = nprocs[1];
	}

	if (npx == 0 or npy == 0) {
		log::error << "processor grid not specified!" << std::endl;
	} else {
		int size;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		assert(size == npx*npy);
		grid = bmg2d::util::create_topo(MPI_COMM_WORLD, npx, npy, nx, ny);
	}


	if (grid != nullptr) {
		run_relax(grid, nrelax);
	}

	MPI_Finalize();
	return 0;
}

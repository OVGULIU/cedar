#include <mpi.h>
#include <iostream>
#include <memory>
#include <math.h>

#include <cedar/types.h>
#include <cedar/3d/mpi/grid_func.h>
#include <cedar/3d/mpi/stencil_op.h>
#include <cedar/3d/kernel/mpi/factory.h>
#include <cedar/3d/kernel/mpi/registry.h>
#include <cedar/3d/util/topo.h>


static void draw(const cedar::cdr3::mpi::grid_func & b, std::string prefix)
{
	using namespace cedar;
	auto & topo = b.grid();
	std::ofstream os("output/" + prefix + "-gfunc-" + std::to_string(topo.coord(0)) + "."
	                 + std::to_string(topo.coord(1)));

	for (auto j : b.grange(1)) {
		for (auto i : b.grange(0)) {
			len_t k = 2;
			if (b(i,j,k) < 0)
				os << '*';
			else
				os << b(i,j,k);
			os << ' ';
		}
		os << '\n';
	}

	os.close();
}


static void draw_so(const cedar::cdr3::mpi::stencil_op & so, std::string prefix)
{

}


static void fill_gfunc(cedar::cdr3::mpi::grid_func & b)
{
	auto & topo = b.grid();
	b.set(-1);
	for (auto k : b.range(2)) {
		for (auto j : b.range(1)) {
			for (auto i : b.range(0)) {
				b(i,j,k) = topo.coord(2)*topo.nproc(0)*topo.nproc(1)
					+ topo.coord(1)*topo.nproc(0) + topo.coord(0);
			}
		}
	}
}


static void fill_stencil(cedar::cdr3::mpi::stencil_op & so)
{
}


int main(int argc, char *argv[])
{
	using namespace cedar;
	using namespace cedar::cdr3;

	int provided;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

	config::reader conf;
	log::init(conf);
	log::status << "Beginning test" << std::endl;
	auto grid = util::create_topo(conf);


	mpi::grid_func b(grid);
	mpi::stencil_op so(grid);
	auto kreg = std::make_shared<cedar::cdr3::kernel::mpi::registry>();
	cedar::cdr3::kernel::mpi::factory::init(kreg, conf);

	void *halo_ctx;
	kreg->halo_setup(*grid, &halo_ctx);
	b.halo_ctx = halo_ctx;
	so.halo_ctx = halo_ctx;

	fill_gfunc(b);
	draw(b, "before");
	kreg->halo_exchange(b);
	draw(b, "after");

	fill_stencil(so);
	draw_so(so, "before");
	kreg->halo_stencil_exchange(so);
	draw_so(so, "after");

	log::status << "Finished Test" << std::endl;

	MPI_Finalize();
	return 0;
}

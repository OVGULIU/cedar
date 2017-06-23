#include <mpi.h>
#include <iostream>
#include <memory>
#include <math.h>

#include <cedar/types.h>
#include <cedar/2d/mpi/grid_func.h>
#include <cedar/2d/mpi/stencil_op.h>
#include <cedar/2d/kernel/mpi/registry.h>
#include <cedar/2d/util/topo.h>

#include <cedar/util/time_log.h>


static void draw(const cedar::cdr2::mpi::grid_func & b, std::string prefix)
{
	auto & topo = b.grid();
	std::ofstream os("output/" + prefix + "-gfunc-" + std::to_string(topo.coord(0)) +
	                 "." + std::to_string(topo.coord(1)));
	for (auto j : b.grange(1)) {
		for (auto i : b.grange(0)) {
			if (b(i,j) < 0)
				os << '*';
			else
				os << b(i,j);
			os << " ";
		}
		os << '\n';
	}

	os.close();
}


static void draw_so(const cedar::cdr2::mpi::stencil_op<cedar::cdr2::five_pt> & so, std::string prefix)
{
	using namespace cedar;
	using namespace cedar::cdr2;

	std::vector<std::shared_ptr<std::ofstream>> osv;

	auto format = [](real_t v) -> std::string {
		if (v < 0.0)
			return "***";
		else {
			auto iv = static_cast<int>(v);
			std::string ret("");
			if (iv < 100)
				ret += "0";
			if (iv < 10)
				ret += "0";
			ret += std::to_string(static_cast<int>(v));
			return ret;
		}
	};

	auto & topo = so.grid();

	for (int i = 0; i < 3; ++i) {
		osv.emplace_back(std::make_shared<std::ofstream>("output/" + prefix + "-stencil-" +
		                                                 std::to_string(topo.coord(0)) +
		                                                 "." + std::to_string(topo.coord(1)) +
		                                                 "-" + std::to_string(i)));
	}

	for (auto j : so.grange(1)) {
		for (auto i : so.grange(0)) {
			*osv[0] << format(so(i,j,five_pt::c)) << " ";
			*osv[1] << format(so(i,j,five_pt::w)) << " ";
			*osv[2] << format(so(i,j,five_pt::s)) << " ";
		}
		for (int i = 0; i < 3; ++i) *(osv[i]) << '\n';
	}

	for (int i = 0; i < 3; ++i) osv[i]->close();
}


static void fill_gfunc(cedar::cdr2::mpi::grid_func & b)
{
	auto & topo = b.grid();
	b.set(-1);
	for (auto j : b.range(1)) {
		for (auto i : b.range(0)) {
			b(i,j) = topo.coord(1)*topo.nproc(0) + topo.coord(0);
		}
	}
}


static void fill_stencil(cedar::cdr2::mpi::stencil_op<cedar::cdr2::five_pt> & so)
{
	using namespace cedar;
	using namespace cedar::cdr2;

	so.set(-1);

	auto & topo = so.grid();

	for (auto j : so.range(1)) {
		for (auto i : so.range(0)) {
			so(i,j,five_pt::c) = 100*topo.coord(0) + 10*topo.coord(1);
			so(i,j,five_pt::w) = 100*topo.coord(0) + 10*topo.coord(1) + 1;
			so(i,j,five_pt::s) = 100*topo.coord(0) + 10*topo.coord(1) + 2;
		}
	}
}


int main(int argc, char *argv[])
{
	using namespace cedar;
	using namespace cedar::cdr2;

	int provided;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

	config::reader conf;
	log::init(conf);
	log::status << "Beginning test" << std::endl;
	auto grid = util::create_topo(conf);


	mpi::grid_func b(grid);
	mpi::stencil_op<five_pt> so(grid);
	auto kreg = kernel::mpi::registry(conf);

	kreg.halo_setup(*grid);

	fill_gfunc(b);
	draw(b, "before");
	kreg.halo_exchange(b);
	draw(b, "after");

	fill_stencil(so);
	draw_so(so, "before");
	kreg.halo_stencil_exchange(so);
	draw_so(so, "after");

	log::status << "Finished Test" << std::endl;

	MPI_Finalize();
	return 0;
}

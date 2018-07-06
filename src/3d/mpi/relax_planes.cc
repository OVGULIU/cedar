#include <cedar/3d/mpi/plane_mempool.h>
#include <cedar/3d/mpi/plane_mpi.h>
#include <cedar/3d/mpi/relax_planes.h>


namespace cedar { namespace cdr3 { namespace mpi {

int log_begin(bool log_planes, int ipl, const std::string & suff)
{
	auto tmp = log::lvl();
	if (log_planes)
		log::set_header_msg(" (plane-" + suff + " " + std::to_string(ipl) + ")");
	else
		log::lvl() = 0;
	return tmp;
}


void log_end(bool log_planes, int ipl, int lvl)
{
	if (log_planes)
		log::set_header_msg("");
	else
		log::lvl() = lvl;
}


cdr2::mpi::kman_ptr master_kman(config & conf, int nplanes)
{
	auto kman = cdr2::mpi::build_kernel_manager(conf);
	auto & serv = kman->services();

	auto regis = [nplanes](service_manager<cdr2::mpi::stypes> & sman) {
		sman.add<services::message_passing, plane_setup_mpi>("plane-setup");
		sman.add<services::mempool, plane_mempool>("plane", nplanes);
		sman.set<services::message_passing>("plane-setup");
		sman.set<services::mempool>("plane");
	};

	serv.set_user_reg(regis);
	return kman;
}


cdr2::mpi::kman_ptr worker_kman(int worker_id, config & conf, cdr2::mpi::kman_ptr master)
{
	auto kman = cdr2::mpi::build_kernel_manager(conf);
	auto & serv = kman->services();

	auto mp_service = master->services().get_ptr<services::message_passing>();
	auto mempool_service = master->services().get_ptr<services::mempool>();
	auto *mpi_keys = static_cast<plane_setup_mpi*>(mp_service.get())->get_keys();
	auto *addrs = static_cast<plane_mempool*>(mempool_service.get())->get_addrs();
	auto regis = [worker_id, addrs, mpi_keys](service_manager<cdr2::mpi::stypes> & sman) {
		sman.add<services::message_passing, plane_setup_mpi>("plane-setup", mpi_keys);
		sman.add<services::mempool, plane_mempool>("plane", worker_id, addrs);
		sman.set<services::message_passing>("plane-setup");
		sman.set<services::mempool>("plane");
	};

	serv.set_user_reg(regis);
	return kman;
}


template<>
void copy_rhs<relax_dir::xy>(const stencil_op<seven_pt> & so,
                             const grid_func & x,
                             const grid_func & b,
                             cdr2::mpi::grid_func & b2,
                             int ipl)
{
	for (auto j : b.range(1)) {
		for (auto i : b.range(0)) {
			b2(i,j) = b(i,j,ipl)
				+ so(i,j,ipl,seven_pt::b) * x(i,j,ipl-1)
				+ so(i,j,ipl+1,seven_pt::b) * x(i,j,ipl+1);
		}
	}
}


template<>
void copy_rhs<relax_dir::xy>(const stencil_op<xxvii_pt> & so,
                             const grid_func & x,
                             const grid_func & b,
                             cdr2::mpi::grid_func & b2,
                             int ipl)
{
	for (auto j : b.range(1)) {
		for (auto i : b.range(0)) {
			b2(i,j) = b(i,j,ipl)
				+ so(i,j,ipl,xxvii_pt::b)*x(i,j,ipl-1)
				+ so(i,j,ipl,xxvii_pt::bw)*x(i-1,j,ipl-1)
				+ so(i,j+1,ipl,xxvii_pt::bnw)*x(i-1,j+1,ipl-1)
				+ so(i,j+1,ipl,xxvii_pt::bn)*x(i,j+1,ipl-1)
				+ so(i+1,j+1,ipl,xxvii_pt::bne)*x(i+1,j+1,ipl-1)
				+ so(i+1,j,ipl,xxvii_pt::be)*x(i+1,j,ipl-1)
				+ so(i+1,j,ipl,xxvii_pt::bse)*x(i+1,j-1,ipl-1)
				+ so(i,j,ipl,xxvii_pt::bs)*x(i,j-1,ipl-1)
				+ so(i,j,ipl,xxvii_pt::bsw)*x(i-1,j-1,ipl-1)
				+ so(i,j,ipl+1,xxvii_pt::be)*x(i-1,j,ipl+1)
				+ so(i,j+1,ipl+1,xxvii_pt::bse)*x(i-1,j+1,ipl+1)
				+ so(i,j+1,ipl+1,xxvii_pt::bs)*x(i,j+1,ipl+1)
				+ so(i+1,j+1,ipl+1,xxvii_pt::bsw)*x(i+1,j+1,ipl+1)
				+ so(i+1,j,ipl+1,xxvii_pt::bw)*x(i+1,j,ipl+1)
				+ so(i,j,ipl+1,xxvii_pt::b)*x(i,j,ipl+1)
				+ so(i+1,j,ipl+1,xxvii_pt::bnw)*x(i+1,j-1,ipl+1)
				+ so(i,j,ipl+1,xxvii_pt::bn)*x(i,j-1,ipl+1)
				+ so(i,j,ipl+1,xxvii_pt::bne)*x(i-1,j-1,ipl+1);
		}
	}
}


template<>
void copy_rhs<relax_dir::xz>(const stencil_op<seven_pt> & so,
                             const grid_func & x,
                             const grid_func & b,
                             cdr2::mpi::grid_func & b2,
                             int ipl)
{
	for (auto k : b.range(2)) {
		for (auto i : b.range(0)) {
			b2(i,k) = b(i,ipl,k)
				+ so(i,ipl,k,seven_pt::ps) * x(i,ipl-1,k)
				+ so(i,ipl+1,k,seven_pt::ps) * x(i,ipl+1,k);
		}
	}
}


template<>
void copy_rhs<relax_dir::xz>(const stencil_op<xxvii_pt> & so,
                             const grid_func & x,
                             const grid_func & b,
                             cdr2::mpi::grid_func & b2,
                             int ipl)
{
	for (auto k : b.range(2)) {
		for (auto i : b.range(0)) {
			b2(i,k) = b(i,ipl,k)
				+ so(i,ipl+1,k,xxvii_pt::pnw) * x(i-1,ipl+1,k)
				+ so(i,ipl+1,k,xxvii_pt::ps) * x(i,ipl+1,k)
				+ so(i+1,ipl+1,k,xxvii_pt::psw) * x(i+1,ipl+1,k)
				+ so(i,ipl+1,k,xxvii_pt::bnw) * x(i-1,ipl+1,k-1)
				+ so(i,ipl+1,k,xxvii_pt::bn) * x(i,ipl+1,k-1)
				+ so(i+1,ipl+1,k,xxvii_pt::bne) * x(i+1,ipl+1,k-1)
				+ so(i,ipl+1,k+1,xxvii_pt::bse) * x(i-1,ipl+1,k+1)
				+ so(i,ipl+1,k+1,xxvii_pt::bs) * x(i,ipl+1,k+1)
				+ so(i+1,ipl+1,k+1,xxvii_pt::bsw) * x(i+1,ipl+1,k+1)
				+ so(i,ipl,k,xxvii_pt::psw) * x(i-1,ipl-1,k)
				+ so(i,ipl,k,xxvii_pt::ps) * x(i,ipl-1,k)
				+ so(i+1,ipl,k,xxvii_pt::pnw) * x(i+1,ipl-1,k)
				+ so(i,ipl,k,xxvii_pt::bsw) * x(i-1,ipl-1,k-1)
				+ so(i,ipl,k,xxvii_pt::bs) * x(i,ipl-1,k-1)
				+ so(i+1,ipl,k,xxvii_pt::bse) * x(i+1,ipl-1,k-1)
				+ so(i,ipl,k+1,xxvii_pt::bne) * x(i-1,ipl-1,k+1)
				+ so(i,ipl,k+1,xxvii_pt::bn) * x(i,ipl-1,k+1)
				+ so(i+1,ipl,k+1,xxvii_pt::bnw) * x(i+1,ipl-1,k+1);
		}
	}
}


template<>
void copy_rhs<relax_dir::yz>(const stencil_op<seven_pt> & so,
                             const grid_func & x,
                             const grid_func & b,
                             cdr2::mpi::grid_func & b2,
                             int ipl)
{
	for (auto k : b.range(2)) {
		for (auto j : b.range(1)) {
			b2(j,k) = b(ipl,j,k)
				+ so(ipl,j,k,seven_pt::pw) * x(ipl-1,j,k)
				+ so(ipl+1,j,k,seven_pt::pw) * x(ipl+1,j,k);
		}
	}
}


template<>
void copy_rhs<relax_dir::yz>(const stencil_op<xxvii_pt> & so,
                             const grid_func & x,
                             const grid_func & b,
                             cdr2::mpi::grid_func & b2,
                             int ipl)
{
	for (auto k : b.range(2)) {
		for (auto j : b.range(1)) {
			b2(j,k) = b(ipl,j,k)
				+ so(ipl,j+1,k,xxvii_pt::pnw) * x(ipl-1,j+1,k)
				+ so(ipl,j,k,xxvii_pt::pw) * x(ipl-1,j,k)
				+ so(ipl,j,k,xxvii_pt::psw) * x(ipl-1,j-1,k)
				+ so(ipl,j+1,k,xxvii_pt::bnw) * x(ipl-1,j+1,k-1)
				+ so(ipl,j,k,xxvii_pt::bw) * x(ipl-1,j,k-1)
				+ so(ipl,j,k,xxvii_pt::bsw) * x(ipl-1,j-1,k-1)
				+ so(ipl,j+1,k+1,xxvii_pt::bse) * x(ipl-1,j+1,k+1)
				+ so(ipl,j,k+1,xxvii_pt::be) * x(ipl-1,j,k+1)
				+ so(ipl,j,k+1,xxvii_pt::bne) * x(ipl-1,j-1,k+1)
				+ so(ipl+1,j+1,k,xxvii_pt::psw) * x(ipl+1,j+1,k)
				+ so(ipl+1,j,k,xxvii_pt::pw) * x(ipl+1,j,k)
				+ so(ipl+1,j,k,xxvii_pt::pnw) * x(ipl+1,j-1,k)
				+ so(ipl+1,j+1,k,xxvii_pt::bne) * x(ipl+1,j+1,k-1)
				+ so(ipl+1,j,k,xxvii_pt::be) * x(ipl+1,j,k-1)
				+ so(ipl+1,j,k,xxvii_pt::bse) * x(ipl+1,j-1,k-1)
				+ so(ipl+1,j+1,k+1,xxvii_pt::bsw) * x(ipl+1,j+1,k+1)
				+ so(ipl+1,j,k+1,xxvii_pt::bw) * x(ipl+1,j,k+1)
				+ so(ipl+1,j,k+1,xxvii_pt::bnw) * x(ipl+1,j-1,k+1);
		}
	}
}


template<>
void copy32<relax_dir::xy>(const grid_func & x, cdr2::mpi::grid_func & x2, int ipl)
{
	for (auto j : x.grange(1)) {
		for (auto i : x.grange(0)) {
			x2(i,j) = x(i,j,ipl);
		}
	}
}


template<>
void copy23<relax_dir::xy>(const cdr2::mpi::grid_func & x2, grid_func &x, int ipl)
{
	for (auto j : x.grange(1)) {
		for (auto i : x.grange(0)) {
			x(i,j,ipl) = x2(i,j);
		}
	}
}


template<>
void copy32<relax_dir::xz>(const grid_func & x, cdr2::mpi::grid_func & x2, int ipl)
{
	for (auto k : x.grange(2)) {
		for (auto i : x.grange(0)) {
			x2(i,k) = x(i,ipl,k);
		}
	}
}


template<>
void copy23<relax_dir::xz>(const cdr2::mpi::grid_func & x2, grid_func &x, int ipl)
{
	for (auto k : x.grange(2)) {
		for (auto i : x.grange(0)) {
			x(i,ipl,k) = x2(i,k);
		}
	}
}


template<>
void copy32<relax_dir::yz>(const grid_func & x, cdr2::mpi::grid_func & x2, int ipl)
{
	for (auto k : x.grange(2)) {
		for (auto j : x.grange(1)) {
			x2(j,k) = x(ipl,j,k);
		}
	}
}


template<>
void copy23<relax_dir::yz>(const cdr2::mpi::grid_func & x2, grid_func &x, int ipl)
{
	for (auto k : x.grange(2)) {
		for (auto j : x.grange(1)) {
			x(ipl,j,k) = x2(j,k);
		}
	}
}

}}}

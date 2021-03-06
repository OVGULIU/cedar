#include <cedar/3d/mpi/gallery.h>

namespace cedar { namespace cdr3 { namespace mpi { namespace gallery {

using namespace cedar;

stencil_op<seven_pt> poisson(topo_ptr grid)
{
	stencil_op<seven_pt> so(grid);

	auto & topo = so.grid();

	so.set(0);

	real_t nlx = topo.nlocal(0) - 2;
	real_t nly = topo.nlocal(1) - 2;
	real_t nlz = topo.nlocal(2) - 2;
	real_t ngx = topo.nglobal(0) - 2;
	real_t ngy = topo.nglobal(1) - 2;
	real_t ngz = topo.nglobal(2) - 2;

	real_t igs = topo.is(0);
	real_t jgs = topo.is(1);
	real_t kgs = topo.is(2);

	real_t hx = 1.0 / (topo.nglobal(0) - 1);
	real_t hy = 1.0 / (topo.nglobal(1) - 1);
	real_t hz = 1.0 / (topo.nglobal(2) - 1);

	real_t nlx_g = nlx + 2;
	real_t nly_g = nly + 2;
	real_t nlz_g = nlz + 2;

	// real_t i1 = nlx + 1;
	// real_t j1 = nly + 1;
	// real_t k1 = nlz + 1;
	real_t i2 = nlx;
	real_t j2 = nly;
	real_t k2 = nlz;

	real_t igf = igs + i2 - 1;
	real_t jgf = jgs + j2 - 1;
	real_t kgf = kgs + k2 - 1;

	real_t xh=hy*hz/hx;
	real_t yh=hx*hz/hy;
	real_t zh=hx*hy/hz;

	real_t ibeg = 1;
	real_t jbeg = 1;
	real_t kbeg = 1;

	if (igs == 1)
		ibeg++;
	if (jgs == 1)
		jbeg++;
	if (kgs == 1)
		kbeg++;

	real_t iend = nlx_g;
	real_t jend = nly_g;
	real_t kend = nlz_g;

	if (igf == ngx)
		iend--;
	if (jgf == ngy)
		jend--;
	if (kgf == ngz)
		kend--;

	for (auto k : range<len_t>(1, kend)) {
		for (auto j : range<len_t>(jbeg, jend)) {
			for (auto i : range<len_t>(1, iend)) {
				so(i,j,k,seven_pt::ps) = yh;
			}
		}
	}


	for (auto k : range<len_t>(1, kend)) {
		for (auto j : range<len_t>(1, jend)) {
			for (auto i : range<len_t>(ibeg, iend)) {
				so(i,j,k,seven_pt::pw) = xh;
			}
		}
	}


	for (auto k : range<len_t>(kbeg, kend)) {
		for (auto j : range<len_t>(1, jend)) {
			for (auto i : range<len_t>(1, iend)) {
				so(i,j,k,seven_pt::b) = zh;
			}
		}
	}


	for (auto k : range<len_t>(1, kend)) {
		for (auto j : range<len_t>(1, jend)) {
			for (auto i : range<len_t>(1, iend)) {
				so(i,j,k,seven_pt::p) = 2*xh + 2*yh + 2*zh;
			}
		}
	}

	return so;
}


stencil_op<xxvii_pt> fe(topo_ptr grid)
{
	stencil_op<xxvii_pt> so(grid);

	auto & topo = so.grid();

	so.set(0);

	real_t nlx = topo.nlocal(0) - 2;
	real_t nly = topo.nlocal(1) - 2;
	real_t nlz = topo.nlocal(2) - 2;
	real_t ngx = topo.nglobal(0) - 2;
	real_t ngy = topo.nglobal(1) - 2;
	real_t ngz = topo.nglobal(2) - 2;

	real_t igs = topo.is(0);
	real_t jgs = topo.is(1);
	real_t kgs = topo.is(2);

	real_t nlx_g = nlx + 2;
	real_t nly_g = nly + 2;
	real_t nlz_g = nlz + 2;

	// real_t i1 = nlx + 1;
	// real_t j1 = nly + 1;
	// real_t k1 = nlz + 1;
	real_t i2 = nlx;
	real_t j2 = nly;
	real_t k2 = nlz;

	real_t igf = igs + i2 - 1;
	real_t jgf = jgs + j2 - 1;
	real_t kgf = kgs + k2 - 1;

	real_t ibeg = 1;
	real_t jbeg = 1;
	real_t kbeg = 1;

	if (igs == 1)
		ibeg++;
	if (jgs == 1)
		jbeg++;
	if (kgs == 1)
		kbeg++;

	real_t iend = nlx_g;
	real_t jend = nly_g;
	real_t kend = nlz_g;

	if (igf == ngx)
		iend--;
	if (jgf == ngy)
		jend--;
	if (kgf == ngz)
		kend--;


	for (auto k : range<len_t>(1, kend)) {
		for (auto j : range<len_t>(1, jend)) {
			for (auto i : range<len_t>(ibeg, iend)) {
				so(i,j,k,xxvii_pt::pw) = 1;
			}
		}
	}

	for (auto k : range<len_t>(1, kend)) {
		for (auto j : range<len_t>(jbeg, jend)) {
			for (auto i : range<len_t>(1, iend)) {
				so(i,j,k,xxvii_pt::ps) = 1;
			}
		}
	}


	for (auto k : range<len_t>(kbeg, kend)) {
		for (auto j : range<len_t>(1, jend)) {
			for (auto i : range<len_t>(1, iend)) {
				so(i,j,k,xxvii_pt::b) = 1;
			}
		}
	}

	for (auto k : range<len_t>(1, kend)){
		for (auto j : range<len_t>(jbeg, jend)) {
			for (auto i : range<len_t>(ibeg, iend)) {
				so(i,j,k,xxvii_pt::pnw) = 1.0;
				so(i,j,k,xxvii_pt::psw) = 1.0;
			}
		}
	}

	for (auto k : range<len_t>(kbeg, kend)){
		for (auto j : range<len_t>(1, jend)) {
			for (auto i : range<len_t>(ibeg, iend)) {
				so(i,j,k,xxvii_pt::bw) = 1.0;
				so(i,j,k,xxvii_pt::be) = 1.0;
			}
		}
	}

	for (auto k : range<len_t>(kbeg, kend)){
		for (auto j : range<len_t>(jbeg, jend)) {
			for (auto i : range<len_t>(1, iend)) {
				so(i,j,k,xxvii_pt::bn) = 1.0;
				so(i,j,k,xxvii_pt::bs) = 1.0;
			}
		}
	}

	for (auto k : range<len_t>(kbeg, kend)){
		for (auto j : range<len_t>(jbeg, jend)) {
			for (auto i : range<len_t>(ibeg, iend)) {
				so(i,j,k,xxvii_pt::bnw) = 1.0;
				so(i,j,k,xxvii_pt::bne) = 1.0;
				so(i,j,k,xxvii_pt::bse) = 1.0;
				so(i,j,k,xxvii_pt::bsw) = 1.0;
			}
		}
	}

	for (auto k : range<len_t>(1, kend)) {
		for (auto j : range<len_t>(1, jend)) {
			for (auto i : range<len_t>(1, iend)) {
				so(i,j,k,xxvii_pt::p) = 26;
			}
		}
	}

	return so;
}

}}}}

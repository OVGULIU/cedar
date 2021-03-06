#include <iomanip>

#include <cedar/2d/prolong_op.h>


using namespace cedar::cdr2;

prolong_op::prolong_op(len_t nx, len_t ny) : stencil_op<inter_dir>(nx, ny)
{
}

namespace cedar { namespace cdr2 {
std::ostream & operator<< (std::ostream &os, const prolong_op &P)
{
	// unsigned int width = 8;

	// for (auto j: sten.range(1)) {
	// 	for (auto i: sten.range(0)) {
	// 		os << i << ", " <<  j
	// 		   << std::setw(width) << sten(i,j+1,Dir::SE) << " "
	// 		   << std::setw(width) << sten(i,j+1,Dir::B)  << " "
	// 		   << std::setw(width) << sten(i+1,j+1,Dir::SW) << " "
	// 		   << std::setw(width) << sten(i,j,Dir::R) << " "
	// 		   << std::setw(width) << 1 << " "
	// 		   << std::setw(width) << sten(i+1,j,Dir::L) << " "
	// 		   << std::setw(width) << sten(i,j,Dir::NE) << " "
	// 		   << std::setw(width) << sten(i,j,Dir::A) << " "
	// 		   << std::setw(width) << sten(i+1,j,Dir::NW) << '\n';
	// 	}
	// }

	os << std::setprecision(7);
	for (auto j: P.range(1)) {
		for (auto i: P.range(0)) {
			os << i+1 << ", " <<  j+1 << ", "
			   << std::scientific << P(i,j+1,inter_dir::SE) << ", "
			   << std::scientific << P(i,j+1,inter_dir::B)  << ", "
			   << std::scientific << P(i+1,j+1,inter_dir::SW) << ", "
			   << std::scientific << P(i,j,inter_dir::R) << ", "
			   << std::scientific << 1.0 << ", "
			   << std::scientific << P(i+1,j,inter_dir::L) << ", "
			   << std::scientific << P(i,j,inter_dir::NE) << ", "
			   << std::scientific << P(i,j,inter_dir::A) << ", "
			   << std::scientific << P(i+1,j,inter_dir::NW) << '\n';
		}
	}

	return os;
}


}}

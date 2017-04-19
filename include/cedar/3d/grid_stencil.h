#ifndef CEDAR_3D_GRIDSTENCIL_H
#define CEDAR_3D_GRIDSTENCIL_H

#include <cedar/3d/types.h>
#include <cedar/3d/inter/types.h>
#include <cedar/array.h>
#include <cedar/grid_quantity.h>

namespace cedar { namespace cdr3 {

class grid_stencil : public array<len_t, real_t, 4>, public grid_quantity<len_t, 3>
{
public:
	grid_stencil() {}
	grid_stencil & operator=(grid_stencil&& gs);
	grid_stencil(const grid_stencil &gs) = default;
	grid_stencil & operator=(const grid_stencil & gs) = default;
	grid_stencil(len_t nx, len_t ny, len_t nz, unsigned int nghosts=1, bool intergrid=false, bool symmetric=true, bool five_pt=false);

	len_t index(len_t i, len_t j, len_t k, dir direction) const
	{
		return array<len_t,real_t,4>::index(i,j,k,static_cast<len_t>(direction));
	}

	real_t & operator()(len_t i, len_t j, len_t k, dir direction)
	{
		return array<len_t,real_t,4>::operator()(i,j,k,static_cast<len_t>(direction));
	}

	real_t operator()(len_t i, len_t j, len_t k, dir direction) const
	{
		return array<len_t,real_t,4>::operator()(i,j,k,static_cast<len_t>(direction));
	}

	real_t & operator()(len_t i, len_t j, len_t k, inter::dir direction)
	{
		return array<len_t,real_t,4>::operator()(i,j,k,static_cast<len_t>(direction));
	}

	real_t operator()(len_t i, len_t j, len_t k, inter::dir direction) const
	{
		return array<len_t,real_t,4>::operator()(i,j,k,static_cast<len_t>(direction));
	}

	bool five_pt() const
	{
		return five_pt_;
	}

	bool & five_pt()
	{
		return five_pt_;
	}


private:
	bool symmetric;
	bool five_pt_;
};

}}
#endif
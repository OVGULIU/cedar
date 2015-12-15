#include <boxmg/2d/relax_stencil.h>

using namespace boxmg::bmg2d;

relax_stencil::relax_stencil(len_t nx, len_t ny, unsigned int nghosts) : array(nx+2*nghosts,ny+nghosts*2,2), num_ghosts(nghosts)
{
	range_[0] = boxmg::range(static_cast<len_t>(nghosts), static_cast<len_t>(nx + nghosts));
	range_[1] = boxmg::range(static_cast<len_t>(nghosts), static_cast<len_t>(ny + nghosts));

	grange_[0] = boxmg::range(static_cast<len_t>(0), nx + 2*nghosts);
	grange_[1] = boxmg::range(static_cast<len_t>(0), ny + 2*nghosts);
}


const boxmg::range_t<boxmg::len_t> & relax_stencil::range(int i) const
{
		#ifdef DEBUG
		return range_.at(i);
		#else
		return range_[i];
		#endif
}


const boxmg::range_t<boxmg::len_t> & relax_stencil::grange(int i) const
{
	#ifdef DEBUG
	return grange_.at(i);
	#else
	return grange_[i];
	#endif
}


boxmg::len_t relax_stencil::shape(int i) const
{
	return this->len(i) - 2*num_ghosts;
}

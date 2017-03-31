#ifndef CEDAR_3D_MPI_GALLERY_H
#define CEDAR_3D_MPI_GALLERY_H

#include <cedar/types.h>
#include <cedar/3d/mpi/stencil_op.h>

namespace cedar { namespace cdr3 { namespace mpi { namespace gallery {

stencil_op poisson(topo_ptr grid);

}}}}


#endif

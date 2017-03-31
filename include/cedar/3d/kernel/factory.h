#ifndef CEDAR_3D_KERNEL_FACTORY_H
#define CEDAR_3D_KERNEL_FACTORY_H

#include <memory>

#include <cedar/3d/kernel/registry.h>

namespace cedar { namespace cdr3 { namespace kernel {

namespace factory
{
	std::shared_ptr<registry> from_config(config::reader &conf);
}

}}}

#endif

#ifndef CEDAR_PERF_UTIL_H
#define CEDAR_PERF_UTIL_H

#include <limits>

#include <cedar/mpi/grid_topo.h>

namespace cedar {
template <int ND>
int compute_nlevels(grid_topo & topo, len_t min_coarse)
{
	int minlvl = std::numeric_limits<int>::max();

	/*
	  This assumes nlocal is constant over proc grid
	  Could easily remove this constraint by setting
	  dimxfine, dimyfine,... in  cedar::nd::model_topo
	*/

	for (auto dim = 0; dim < ND; dim++) {
		for (auto i = 0; i < topo.nproc(dim); i++) {
			int kg = 0;
			len_t gs_c = (topo.nlocal(dim)-2)*i + 1;
			len_t nl_c = topo.nlocal(dim) - 2;

			do {
				kg += 1;

				if (gs_c % 2 == 1) {
					gs_c = (gs_c + 1) / 2;
					nl_c = (nl_c + 1) / 2;
				} else {
					gs_c = gs_c / 2 + 1;
					if (nl_c % 2 == 1) {
						nl_c = (nl_c - 1) / 2;
					} else {
						nl_c = (nl_c + 1) / 2;
					}
				}
			} while (nl_c >= min_coarse);

			if (kg < minlvl)
				minlvl = kg;
		}
	}

	return minlvl;
}
}
#endif

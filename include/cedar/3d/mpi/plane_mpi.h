#ifndef CEDAR_3D_MPI_PLANE_MPI_H
#define CEDAR_3D_MPI_PLANE_MPI_H

#include <vector>
#include <memory>
#include <cedar/mpi/mpi_wrapper.h>

namespace cedar { namespace cdr3 { namespace mpi {

class plane_setup_mpi : public mpi_wrapper
{
public:
	plane_setup_mpi();
	plane_setup_mpi(std::vector<int> *keys);
	int comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) override;
	std::vector<int> *get_keys() { return key_data.get(); }

protected:
	bool ismaster;
	int currind;
	std::unique_ptr<std::vector<int>> key_data;
	std::vector<int> *keys;
};


class plane_mpi : public mpi_wrapper
{
public:
	plane_mpi(int nplanes, bool ismaster) : nplanes(nplanes), ismaster(ismaster) {}

protected:
	int nplanes;
	bool ismaster;
};

}}}

#endif
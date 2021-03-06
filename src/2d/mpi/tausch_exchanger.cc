#include <cedar/2d/mpi/tausch_exchanger.h>

extern "C" {
	using namespace cedar;
	void BMG2_SymStd_SETUP_Tausch(int nog, len_t *dimx, len_t *dimy,
	                              len_t *dimxfine, len_t *dimyfine, int nproci, int nprocj);
}

namespace cedar { namespace cdr2 { namespace mpi {


line_pkg::line_pkg(grid_topo & topo):
	datadist{{array<len_t, 2>(2, topo.nproc(0)),
			array<len_t, 2>(2, topo.nproc(1))}}
{
	MPI_Comm_split(topo.comm, topo.coord(1), topo.coord(0),
	               &linecomm[0]);
	MPI_Comm_split(topo.comm, topo.coord(0), topo.coord(1),
	               &linecomm[1]);
}


void tausch_exchanger::setup(std::vector<topo_ptr> topos)
{
	this->nlevels = topos.size();
	dims[0] = aarray<int, len_t, 2>(topos[0]->nproc(0), nlevels);
	dims[1] = aarray<int, len_t, 2>(topos[0]->nproc(1), nlevels);
	line_data = std::make_unique<line_pkg>(*topos[0]);
	coord[0] = topos[0]->coord(0);
	coord[1] = topos[0]->coord(1);


	init_gfunc(topos);
	init_so(topos);

	for (std::size_t lvl = 0; lvl < nlevels; lvl++) {
		for (int dir = 0; dir < halo_dir::count; dir++)  {
			send_active[index(lvl,dir)] = true;
			recv_active[index(lvl,dir)] = true;
		}
		if (not params->periodic[0]) {
			if (topos[0]->coord(0) == 0) {
				send_active[index(lvl, halo_dir::left)] = false;
				recv_active[index(lvl, halo_dir::right)] = false;
			}
			if (topos[0]->coord(0) == topos[0]->nproc(0) - 1) {
				send_active[index(lvl, halo_dir::right)] = false;
				recv_active[index(lvl, halo_dir::left)] = false;
			}
		}
		if (not params->periodic[1]) {
			if (topos[0]->coord(1) == 0) {
				send_active[index(lvl, halo_dir::down)] = false;
				recv_active[index(lvl, halo_dir::up)] = false;
			}
			if (topos[0]->coord(1) == topos[0]->nproc(1) - 1) {
				send_active[index(lvl, halo_dir::up)] = false;
				recv_active[index(lvl, halo_dir::down)] = false;
			}
		}
	}

	init_dims(*topos[0]);
}


void tausch_exchanger::init_dims(grid_topo & topo)
{
	auto & dimxfine = dimfine[0];
	auto & dimyfine = dimfine[1];

	if (topo.dimyfine.size() > 0) {
		dimyfine = topo.dimyfine;
	} else {
		dimyfine.reserve(topo.nproc(1));
		for (auto j : range(topo.nproc(1))) {
			dimyfine[j] = topo.nlocal(1) - 2;
		}
	}

	if (topo.dimxfine.size() > 0) {
		dimxfine = topo.dimxfine;
	} else {
		dimxfine.reserve(topo.nproc(0));
		for (auto i : range(topo.nproc(0))) {
			dimxfine[i] = topo.nlocal(0) - 2;
		}
	}

	BMG2_SymStd_SETUP_Tausch(topo.nlevel(),
	                         dims[0].data(), dims[1].data(),
	                         dimxfine.data(), dimyfine.data(),
	                         topo.nproc(0), topo.nproc(1));
	init_datadist();
}


void tausch_exchanger::init_datadist()
{
	auto & dimxfine = dimfine[0];
	auto & dimyfine = dimfine[1];
	auto & xdatadist = line_data->datadist[0];
	auto & ydatadist = line_data->datadist[1];

	xdatadist(0, 0) = 2;
	xdatadist(1, 0) = xdatadist(0, 0) + dimxfine[0] - 1;
	for (auto i : range<len_t>(1, xdatadist.len(1))) {
		xdatadist(0, i) = xdatadist(1, i-1) + 1;
		xdatadist(1, i) = xdatadist(0, i) + dimxfine[i] - 1;
	}

	ydatadist(0, 0) = 2;
	ydatadist(1, 0) = ydatadist(0, 0) + dimyfine[0] - 1;
	for (auto j : range<len_t>(1, ydatadist.len(1))) {
		ydatadist(0, j) = ydatadist(1, j-1) + 1;
		ydatadist(1, j) = ydatadist(0, j) + dimyfine[j] - 1;
	}

	line_data->linebuf.reserve(std::max(dimxfine[0], dimyfine[0])*8*std::max(ydatadist.len(1), xdatadist.len(1)));
}


void tausch_exchanger::init_gfunc(std::vector<topo_ptr> & topos)
{
	std::vector<TauschHaloSpec> remote_spec;
	std::vector<TauschHaloSpec> local_spec;

	remote_spec.reserve(halo_dir::count * nlevels);
	local_spec.reserve(halo_dir::count * nlevels);
	send_active.reserve(halo_dir::count * nlevels);
	recv_active.reserve(halo_dir::count * nlevels);

	int rank;
	MPI_Comm_rank(topos[0]->comm, &rank);

	for (std::size_t lvl = 0; lvl < nlevels; lvl++) {
		set_level_spec(lvl, rank,
		               *topos[lvl],
		               remote_spec, local_spec);
	}

	int nbuf = 1;

	tausch = std::make_unique<Tausch2D<real_t>>(MPI_DOUBLE, nbuf, nullptr, topos[0]->comm);

	tausch->setLocalHaloInfo(TAUSCH_CwC, halo_dir::count * nlevels, local_spec.data());
	tausch->setRemoteHaloInfo(TAUSCH_CwC, halo_dir::count * nlevels, remote_spec.data());
}


void tausch_exchanger::init_so(std::vector<topo_ptr> & topos)
{
	std::vector<TauschHaloSpec> remote_spec;
	std::vector<TauschHaloSpec> local_spec;

	remote_spec.reserve(halo_dir::count * nlevels);
	local_spec.reserve(halo_dir::count * nlevels);
	send_active.reserve(halo_dir::count * nlevels);
	recv_active.reserve(halo_dir::count * nlevels);

	int rank;
	MPI_Comm_rank(topos[0]->comm, &rank);

	for (std::size_t lvl = 0; lvl < nlevels; lvl++) {
		set_level_spec_so(lvl, rank,
		                  *topos[lvl],
		                  remote_spec, local_spec);
	}

	int nbuf = 9;

	tausch_so = std::make_unique<Tausch2D<real_t>>(MPI_DOUBLE, nbuf, nullptr, topos[0]->comm);

	tausch_so->setLocalHaloInfo(TAUSCH_CwC, halo_dir::count * nlevels, local_spec.data());
	tausch_so->setRemoteHaloInfo(TAUSCH_CwC, halo_dir::count * nlevels, remote_spec.data());
}


void tausch_exchanger::set_level_spec(int lvl, int rank,
                                      grid_topo & topo,
                                      std::vector<TauschHaloSpec> & remote_spec,
                                      std::vector<TauschHaloSpec> & local_spec)
{
	for (int i = 0; i < halo_dir::count; i++) {
		remote_spec[index(lvl,i)].bufferWidth = topo.nlocal(0);
		remote_spec[index(lvl,i)].bufferHeight = topo.nlocal(1);
		local_spec[index(lvl,i)].bufferWidth = topo.nlocal(0);
		local_spec[index(lvl,i)].bufferHeight = topo.nlocal(1);
	}

	// right
	remote_spec[index(lvl,halo_dir::right)].haloX = 0;
	remote_spec[index(lvl,halo_dir::right)].haloY = 0;
	remote_spec[index(lvl,halo_dir::right)].haloWidth = 1;
	remote_spec[index(lvl,halo_dir::right)].haloHeight = topo.nlocal(1);
	if (topo.coord(0) == 0)
		remote_spec[index(lvl,halo_dir::right)].remoteMpiRank = rank + topo.nproc(0) - 1;
	else
		remote_spec[index(lvl,halo_dir::right)].remoteMpiRank = rank - 1;

	local_spec[index(lvl,halo_dir::right)].haloX = topo.nlocal(0)-2;
	local_spec[index(lvl,halo_dir::right)].haloY = 0;
	local_spec[index(lvl,halo_dir::right)].haloWidth = 1;
	local_spec[index(lvl,halo_dir::right)].haloHeight = topo.nlocal(1);
	if ((rank + 1) % topo.nproc(0) == 0)
		local_spec[index(lvl,halo_dir::right)].remoteMpiRank = rank - topo.nproc(0) + 1;
	else
		local_spec[index(lvl,halo_dir::right)].remoteMpiRank = rank + 1;


	// left
	remote_spec[index(lvl,halo_dir::left)].haloX = topo.nlocal(0) - 1;
	remote_spec[index(lvl,halo_dir::left)].haloY = 0;
	remote_spec[index(lvl,halo_dir::left)].haloWidth = 1;
	remote_spec[index(lvl,halo_dir::left)].haloHeight = topo.nlocal(1);
	remote_spec[index(lvl,halo_dir::left)].remoteMpiRank = local_spec[index(lvl,halo_dir::right)].remoteMpiRank;

	local_spec[index(lvl,halo_dir::left)].haloX = 1;
	local_spec[index(lvl,halo_dir::left)].haloY = 0;
	local_spec[index(lvl,halo_dir::left)].haloWidth = 1;
	local_spec[index(lvl,halo_dir::left)].haloHeight = topo.nlocal(1);
	local_spec[index(lvl,halo_dir::left)].remoteMpiRank = remote_spec[index(lvl,halo_dir::right)].remoteMpiRank;


	// up
	remote_spec[index(lvl,halo_dir::up)].haloX = 0;
	remote_spec[index(lvl,halo_dir::up)].haloY = 0;
	remote_spec[index(lvl,halo_dir::up)].haloWidth = topo.nlocal(0);
	remote_spec[index(lvl,halo_dir::up)].haloHeight = 1;
	if (topo.coord(1) == 0)
		remote_spec[index(lvl,halo_dir::up)].remoteMpiRank = rank + topo.nproc(0)*topo.nproc(1) - topo.nproc(0);
	else
		remote_spec[index(lvl,halo_dir::up)].remoteMpiRank = rank - topo.nproc(0);

	local_spec[index(lvl,halo_dir::up)].haloX = 0;
	local_spec[index(lvl,halo_dir::up)].haloY = topo.nlocal(1) - 2;
	local_spec[index(lvl,halo_dir::up)].haloWidth = topo.nlocal(0);
	local_spec[index(lvl,halo_dir::up)].haloHeight = 1;
	if (topo.coord(1) == (topo.nproc(1) - 1))
		local_spec[index(lvl,halo_dir::up)].remoteMpiRank = rank - topo.nproc(0)*topo.nproc(1) + topo.nproc(0);
	else
		local_spec[index(lvl,halo_dir::up)].remoteMpiRank = rank + topo.nproc(0);
	// down
	remote_spec[index(lvl,halo_dir::down)].haloX = 0;
	remote_spec[index(lvl,halo_dir::down)].haloY = topo.nlocal(1) - 1;
	remote_spec[index(lvl,halo_dir::down)].haloWidth = topo.nlocal(0);
	remote_spec[index(lvl,halo_dir::down)].haloHeight = 1;
	remote_spec[index(lvl,halo_dir::down)].remoteMpiRank = local_spec[index(lvl,halo_dir::up)].remoteMpiRank;

	local_spec[index(lvl,halo_dir::down)].haloX = 0;
	local_spec[index(lvl,halo_dir::down)].haloY = 1;
	local_spec[index(lvl,halo_dir::down)].haloWidth = topo.nlocal(0);
	local_spec[index(lvl,halo_dir::down)].haloHeight = 1;
	local_spec[index(lvl,halo_dir::down)].remoteMpiRank = remote_spec[index(lvl,halo_dir::up)].remoteMpiRank;
}


void tausch_exchanger::set_level_spec_so(int lvl, int rank,
                                         grid_topo & topo,
                                         std::vector<TauschHaloSpec> & remote_spec,
                                         std::vector<TauschHaloSpec> & local_spec)
{
	for (int i = 0; i < halo_dir::count; i++) {
		remote_spec[index(lvl,i)].bufferWidth = topo.nlocal(0) + 1;
		remote_spec[index(lvl,i)].bufferHeight = topo.nlocal(1) + 1;
		local_spec[index(lvl,i)].bufferWidth = topo.nlocal(0) + 1;
		local_spec[index(lvl,i)].bufferHeight = topo.nlocal(1) + 1;
	}

	// right
	remote_spec[index(lvl,halo_dir::right)].haloX = 0;
	remote_spec[index(lvl,halo_dir::right)].haloY = 0;
	remote_spec[index(lvl,halo_dir::right)].haloWidth = 1;
	remote_spec[index(lvl,halo_dir::right)].haloHeight = topo.nlocal(1);
	if (topo.coord(0) == 0)
		remote_spec[index(lvl,halo_dir::right)].remoteMpiRank = rank + topo.nproc(0) - 1;
	else
		remote_spec[index(lvl,halo_dir::right)].remoteMpiRank = rank - 1;

	local_spec[index(lvl,halo_dir::right)].haloX = topo.nlocal(0)-2;
	local_spec[index(lvl,halo_dir::right)].haloY = 0;
	local_spec[index(lvl,halo_dir::right)].haloWidth = 1;
	local_spec[index(lvl,halo_dir::right)].haloHeight = topo.nlocal(1);
	if ((rank + 1) % topo.nproc(0) == 0)
		local_spec[index(lvl,halo_dir::right)].remoteMpiRank = rank - topo.nproc(0) + 1;
	else
		local_spec[index(lvl,halo_dir::right)].remoteMpiRank = rank + 1;


	// left
	remote_spec[index(lvl,halo_dir::left)].haloX = topo.nlocal(0) - 1;
	remote_spec[index(lvl,halo_dir::left)].haloY = 0;
	remote_spec[index(lvl,halo_dir::left)].haloWidth = 2;
	remote_spec[index(lvl,halo_dir::left)].haloHeight = topo.nlocal(1);
	remote_spec[index(lvl,halo_dir::left)].remoteMpiRank = local_spec[index(lvl,halo_dir::right)].remoteMpiRank;

	local_spec[index(lvl,halo_dir::left)].haloX = 1;
	local_spec[index(lvl,halo_dir::left)].haloY = 0;
	local_spec[index(lvl,halo_dir::left)].haloWidth = 2;
	local_spec[index(lvl,halo_dir::left)].haloHeight = topo.nlocal(1);
	local_spec[index(lvl,halo_dir::left)].remoteMpiRank = remote_spec[index(lvl,halo_dir::right)].remoteMpiRank;


	// up
	remote_spec[index(lvl,halo_dir::up)].haloX = 0;
	remote_spec[index(lvl,halo_dir::up)].haloY = 0;
	remote_spec[index(lvl,halo_dir::up)].haloWidth = topo.nlocal(0);
	remote_spec[index(lvl,halo_dir::up)].haloHeight = 1;
	if (topo.coord(1) == 0)
		remote_spec[index(lvl,halo_dir::up)].remoteMpiRank = rank + topo.nproc(0)*topo.nproc(1) - topo.nproc(0);
	else
		remote_spec[index(lvl,halo_dir::up)].remoteMpiRank = rank - topo.nproc(0);

	local_spec[index(lvl,halo_dir::up)].haloX = 0;
	local_spec[index(lvl,halo_dir::up)].haloY = topo.nlocal(1) - 2;
	local_spec[index(lvl,halo_dir::up)].haloWidth = topo.nlocal(0);
	local_spec[index(lvl,halo_dir::up)].haloHeight = 1;
	if (topo.coord(1) == (topo.nproc(1) - 1))
		local_spec[index(lvl,halo_dir::up)].remoteMpiRank = rank - topo.nproc(0)*topo.nproc(1) + topo.nproc(0);
	else
		local_spec[index(lvl,halo_dir::up)].remoteMpiRank = rank + topo.nproc(0);
	// down
	remote_spec[index(lvl,halo_dir::down)].haloX = 0;
	remote_spec[index(lvl,halo_dir::down)].haloY = topo.nlocal(1) - 1;
	remote_spec[index(lvl,halo_dir::down)].haloWidth = topo.nlocal(0);
	remote_spec[index(lvl,halo_dir::down)].haloHeight = 2;
	remote_spec[index(lvl,halo_dir::down)].remoteMpiRank = local_spec[index(lvl,halo_dir::up)].remoteMpiRank;

	local_spec[index(lvl,halo_dir::down)].haloX = 0;
	local_spec[index(lvl,halo_dir::down)].haloY = 1;
	local_spec[index(lvl,halo_dir::down)].haloWidth = topo.nlocal(0);
	local_spec[index(lvl,halo_dir::down)].haloHeight = 2;
	local_spec[index(lvl,halo_dir::down)].remoteMpiRank = remote_spec[index(lvl,halo_dir::up)].remoteMpiRank;
}


void tausch_exchanger::exchange_func(int k, real_t * gf)
{
	int lvl = nlevels - k;
	for (int dir = 0; dir < halo_dir::count; dir++) {
		if (recv_active[index(lvl, dir)])
			tausch->postReceive(TAUSCH_CwC, index(lvl, dir), index(lvl, dir));
	}

	for (int dir = 0; dir < halo_dir::count; dir++) {
		if (send_active[index(lvl, dir)]) {
			tausch->packSendBuffer(TAUSCH_CwC, index(lvl,dir), 0, gf);
			tausch->send(TAUSCH_CwC, index(lvl,dir), index(lvl,dir));

		}
		if (recv_active[index(lvl, dir)]) {
			tausch->recv(TAUSCH_CwC, index(lvl,dir));
			tausch->unpackRecvBuffer(TAUSCH_CwC, index(lvl,dir), 0, gf);
		}
	}
}

void tausch_exchanger::exchange_sten(int k, real_t * so)
{
	int lvl = nlevels - k;

	auto & dimx = leveldims(0);
	auto & dimy = leveldims(1);

	len_t II = dimx(coord[0], k-1) + 2;
	len_t JJ = dimy(coord[1], k-1) + 2;

	for (int dir = 0; dir < halo_dir::count; dir++) {
		if (recv_active[index(lvl, dir)])
			tausch_so->postReceive(TAUSCH_CwC, index(lvl, dir), index(lvl, dir));
	}

	for (int dir = 0; dir < halo_dir::count; dir++) {
		if (send_active[index(lvl, dir)]) {
			for (int sdir = 0; sdir < 5; sdir++) {
				len_t offset = (JJ+1)*(II+1)*sdir;
				tausch_so->packSendBuffer(TAUSCH_CwC, index(lvl,dir), sdir, so + offset);
			}
			tausch_so->send(TAUSCH_CwC, index(lvl,dir), index(lvl,dir));

		}
		if (recv_active[index(lvl, dir)]) {
			tausch_so->recv(TAUSCH_CwC, index(lvl,dir));
			for (int sdir = 0; sdir < 5; sdir++) {
				len_t offset = (JJ+1)*(II+1)*sdir;
				tausch_so->unpackRecvBuffer(TAUSCH_CwC, index(lvl,dir), sdir, so + offset);
			}
		}
	}
}

void tausch_exchanger::run(mpi::grid_func & f)
{
	auto lvl = f.grid().level();
	lvl = nlevels - lvl - 1;

	for (int dir = 0; dir < halo_dir::count; dir++) {
		if (recv_active[index(lvl, dir)])
			tausch->postReceive(TAUSCH_CwC, index(lvl, dir), index(lvl, dir));
	}

	for (int dir = 0; dir < halo_dir::count; dir++) {
		if (send_active[index(lvl, dir)]) {
			tausch->packSendBuffer(TAUSCH_CwC, index(lvl,dir), 0, f.data());
			tausch->send(TAUSCH_CwC, index(lvl,dir), index(lvl,dir));

		}
		if (recv_active[index(lvl, dir)]) {
			tausch->recv(TAUSCH_CwC, index(lvl,dir));
			tausch->unpackRecvBuffer(TAUSCH_CwC, index(lvl,dir), 0, f.data());
		}
	}
}


}}}

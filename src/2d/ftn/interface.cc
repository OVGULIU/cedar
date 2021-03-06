#include "cedar/util/log.h"
#include <cedar/types.h>
#include <cedar/util/time_log.h>
#include <cedar/halo_exchanger_base.h>

extern "C" {

	void print_error(char *string)
	{
		cedar::log::error << string << std::endl;
	}

	void ftimer_begin(char *string)
	{
		cedar::timer_begin(string);
	}

	void ftimer_end(char *string)
	{
		cedar::timer_end(string);
	}

	void halo_exchange(int k, cedar::real_t *q, cedar::halo_exchanger_base *halof)
	{
		halof->exchange_func(k, q);
	}


	void halo_stencil_exchange(int k, cedar::real_t *so, cedar::halo_exchanger_base *halof)
	{
		halof->exchange_sten(k, so);
	}
}

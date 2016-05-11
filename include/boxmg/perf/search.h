#ifndef BOXMG_PERF_SEARCH_H
#define BOXMG_PERF_SEARCH_H

#include <array>
#include <vector>

#include <boxmg/ss/node.h>
#include <boxmg/ss/problem.h>
#include <boxmg/ss/solution.h>
#include <boxmg/perf/vcycle_model.h>
#include <boxmg/perf/cholesky_model.h>

namespace boxmg {

struct perf_state
{
	std::shared_ptr<vcycle_model> model;
	friend bool operator<(const perf_state & s1, const perf_state & s2);
};


using perf_node = ss::node<perf_state, std::array<int, 2>, float>;

struct perf_problem : ss::problem<perf_state, std::array<int,2>, float>
{
	bool goal_test(perf_state & model);
	perf_state result(perf_state & model, std::array<int,2> action);
	float step_cost(perf_state & state, std::array<int,2> act);
	std::vector<std::array<int,2>> actions(perf_state & state);
};


class perf_solution : public ss::solution<perf_state, std::array<int,2>, float>
{
public:
	using node_ptr = std::shared_ptr<perf_node>;
perf_solution(node_ptr sol_node) :
	ss::solution<perf_state, std::array<int,2>,float>(sol_node)
	{
	}

	std::shared_ptr<vcycle_model> model() {
		config::reader conf("config.json");
		auto pnode = snode();
		float tc = conf.get<float>("machine.fp_perf");
		auto model = pnode->state.model;
		auto & topoc = model->grid(0);
		auto cg_model = std::make_shared<cholesky_model>(topoc.nglobal(0)*topoc.nglobal(1));
		cg_model->set_comp_param(tc);
		model->set_cgperf(cg_model);

		while (pnode->parent != nullptr) {
			pnode->parent->state.model->set_cgperf(pnode->state.model);
			pnode = pnode->parent;
		}

		return pnode->state.model;
	}
};

}

#endif
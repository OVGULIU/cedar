#ifndef CEDAR_KERNEL_MANAGER_H
#define CEDAR_KERNEL_MANAGER_H

#include <cedar/type_list.h>
#include <cedar/kernel_params.h>
#include <cedar/config.h>
#include <cedar/halo_exchanger_base.h>

namespace cedar {

template<class tlist>
class kernel_manager : public type_map<tlist>
{
public:
	using parent = type_map<tlist>;
	using mtype = typename parent::mtype;
	using parent::kerns;

    kernel_manager(std::shared_ptr<kernel_params> params) : params(params) {}
	kernel_manager(config & conf) { params = build_kernel_params(conf); }

	template<class T, class rclass>
	void add(const std::string & name)
	{
		parent::template add<T, rclass>(name);
		this->init<T>(name);
	}


	template<class T, class rclass, class... Args>
	void add(const std::string & name, Args&&... args)
	{
		parent::template add<T, rclass>(name, std::forward<Args>(args)...);
		this->init<T>(name);
	}


	template<class T, class... Args>
	void setup(Args&&... args)
	{
		auto kern = parent::template get_ptr<T>();
		log::debug << "setup kernel <" << T::name() << ">" << std::endl;
		if (!kern)
			log::error << "kernel not found: " << T::name() << std::endl;
		kern->setup(std::forward<Args>(args)...);
	}

	template<class T, class... Args>
	void run(Args&&... args)
	{
		auto kern = parent::template get_ptr<T>();
		log::debug << "running kernel <" << T::name() << ">" << std::endl;
		if (!kern)
			log::error << "kernel not found: " << T::name() << std::endl;
		kern->run(std::forward<Args>(args)...);
	}


	template<class T>
	void init(const std::string & name)
	{
		parent::template get<T>(name).add_params(params);
	}

	void add_halo(halo_exchanger_base * halof)
	{
		static const std::size_t n = std::tuple_size<mtype>::value;
		add_halo_impl<n, mtype>::call(kerns, halof);
	}

protected:
	std::shared_ptr<kernel_params> params;

	template<std::size_t I, class mtype>
	struct add_halo_impl
	{
		static void call(mtype & kerns, halo_exchanger_base *halof)
			{
				auto & impls = std::get<I-1>(kerns);
				for (auto & impl : impls)
					impl.second->add_halo(halof);

				add_halo_impl<I-1, mtype>::call(kerns, halof);
			}
	};


	template<class mtype>
	struct add_halo_impl<0, mtype>
	{
		static void call(mtype & kerns, halo_exchanger_base *halof){}
	};
};

}

#endif

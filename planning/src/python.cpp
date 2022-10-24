#include <string>

#include "Eigen/SparseCore"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"
#include "nanobind/tensor.h"
#include "planning/loadbalance.hpp"
#include "planning/queuing.hpp"
#include "planning/reward.hpp"

namespace nb = nanobind;

using namespace nb::literals;

template <typename data_type>
static constexpr auto make_return_tensor(const data_type *data,
                                         const size_t *dims,
                                         const size_t n_dim) {
  return nb::tensor<nb::numpy, data_type>(const_cast<data_type *>(data), n_dim,
                                          dims);
}

template <typename data_type, std::convertible_to<size_t>... size_types>
requires std::is_scalar_v<data_type>
static constexpr auto make_return_tensor(const data_type *data,
                                         size_types... sizes) {
  std::array dims{static_cast<size_t>(sizes)...};
  return make_return_tensor(data, dims.data(), sizeof...(size_types));
}

static constexpr auto make_return_tensor(const auto &v) {
  return make_return_tensor(v.data(), static_cast<size_t>(v.size()));
}

namespace nanobind {

struct bytes : std::string {
  using std::string::string;

  bytes(const std::string &str) : std::string(str) {}
  bytes(std::string &&str) : std::string(str) {}
};

namespace detail {

template <typename Type>
requires std::derived_from<Type, Eigen::SparseCompressedBase<Type>>
struct type_caster<Type> {
  using Scalar = typename Type::Scalar;
  using StorageIndex = typename Type::StorageIndex;
  static constexpr bool IsRowMajor = Type::IsRowMajor;

  NB_TYPE_CASTER(Type,
                 (const_name<Type::IsRowMajor>("scipy.sparse.csr_array[",
                                               "scipy.sparse.csc_array[") +
                  tensor_arg<Scalar>::name + const_name("]")));

  static handle from_cpp(const Type &value, rv_policy,
                         cleanup_list *) noexcept {
    if (!value.isCompressed()) return {};

    auto sp_type = module_::import_("scipy.sparse")
                       .attr(IsRowMajor ? "csr_array" : "csc_array");

    return sp_type(
               nb::make_tuple(
                   make_return_tensor(value.valuePtr(), value.nonZeros()),
                   make_return_tensor(value.innerIndexPtr(), value.nonZeros()),
                   make_return_tensor(value.outerIndexPtr(),
                                      value.outerSize() + 1)),
               nb::make_tuple(value.rows(), value.cols()))
        .release();
  }
};

template <>
struct type_caster<bytes> {
  NB_TYPE_CASTER(bytes, (const_name("bytes")));

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    Py_ssize_t size;
    char *str;
    const int status = PyBytes_AsStringAndSize(src.ptr(), &str, &size);
    if (status == -1) {
      PyErr_Clear();
      return false;
    }
    value = bytes(str, (size_t)size);
    return true;
  }

  static handle from_cpp(const bytes &value, rv_policy,
                         cleanup_list *) noexcept {
    return PyBytes_FromStringAndSize(value.c_str(), value.size());
  }
};

}  // namespace detail
}  // namespace nanobind

void make_state(nb::module_ &m) {
  nb::class_<CartesianProduct>(m, "CartesianProduct")
      .def_property_readonly(
          "n", [](const CartesianProduct &self) { return self.n; })
      .def_property_readonly(
          "d", [](const CartesianProduct &self) { return self.d; })
      .def_property_readonly("a", [](const CartesianProduct &self) {
        return make_return_tensor(self.lens.data(), self.n, self.d);
      });

  nb::class_<State>(m, "State")
      .def_property_readonly("n_env",
                             [](const State &self) { return self.n_env; })
      .def_property_readonly("n_cls",
                             [](const State &self) { return self.n_cls; })
      .def_property_readonly("cls",
                             [](const State &self) -> const CartesianProduct & {
                               return self.cls;
                             })
      .def_property_readonly("env",
                             [](const State &self) -> const CartesianProduct & {
                               return self.env;
                             })
      .def_property_readonly("sys",
                             [](const State &self) -> const CartesianProduct & {
                               return self.sys;
                             });
}

using param_type = nb::tensor<nb::numpy, float_type>;

void make_system(nb::module_ &m) {
  auto cls = nb::class_<System>(m, "System");

  cls.def_property_readonly("n_env",
                            [](const System &self) { return self.n_env; })
      .def_property_readonly(
          "limits",
          [](const System &self) { return make_return_tensor(self.limits); })
      .def_property_readonly(
          "states",
          [](const System &self) -> const State & { return self.states; })
      .def_property_readonly("n_cls",
                             [](const System &self) { return self.n_cls; });

  cls.def_property_readonly("costs",
                            [](const System &self) {
                              return make_return_tensor(self.costs.data(),
                                                        self.n_cls, self.n_env);
                            })
      .def_property_readonly("arrivals",
                             [](const System &self) {
                               return make_return_tensor(self.arrivals.data(),
                                                         self.n_cls,
                                                         self.n_env);
                             })
      .def_property_readonly("departures",
                             [](const System &self) {
                               return make_return_tensor(self.departures.data(),
                                                         self.n_cls,
                                                         self.n_env);
                             })
      .def_property_readonly("env_trans_mats",
                             [](const System &self) {
                               return make_return_tensor(
                                   self.env_trans_mats.data(), self.n_cls,
                                   self.n_env, self.n_env);
                             })
      .def_property_readonly(
          "normalized_c", [](const System &self) { return self.normalized_c; });

  cls.def_property_readonly("rewards", [](const System &self) {
    return make_return_tensor(self.rewards);
  });

  cls.def_property_readonly(
      "trans_probs",
      [](const System &self) -> const SpMats & { return self.trans_probs; });

  cls.def_property_readonly(
      "env_trans_probs",
      [](const System &self) -> const SpMat & { return self.env_trans_probs; });

  cls.def_property_readonly(
         "cls_dims",
         [](const System &self) { return make_return_tensor(self.cls_dims); })
      .def_property_readonly("cls_action_dims", [](const System &self) {
        return make_return_tensor(self.cls_action_dims);
      });

  cls.def_property_readonly("n_cls_visit", [](const System &self) {
    return make_return_tensor(self.n_cls_visit().data(),
                              self.cls_action_dims.data(),
                              self.cls_action_dims.size());
  });

  cls.def("train_q", &System::train_q)
      .def("train_q_i", &System::train_q_i)
      .def("train_qs", &System::train_qs)
      .def("train_q_full", &System::train_q_full)
      .def("train_q_off", &System::train_q_off)
      .def_property_readonly("q",
                             [](const System &self) {
                               return make_return_tensor(
                                   self.q().data(), self.cls_action_dims.data(),
                                   self.cls_action_dims.size());
                             })
      .def_property_readonly("q_policy",
                             [](const System &self) {
                               return make_return_tensor(self.q_policy().data(),
                                                         self.cls_dims.data(),
                                                         self.cls_dims.size());
                             })
      .def_property_readonly(
          "qs",
          [](const System &self) {
            const auto qs_dims = (VectorAS(self.cls_action_dims.size() + 1)
                                      << self.qs().dimension(0),
                                  self.cls_action_dims)
                                     .finished();
            return make_return_tensor(self.qs().data(), qs_dims.data(),
                                      qs_dims.size());
          })
      .def_property_readonly("i_cls_trans_probs",
                             [](const System &self) -> const SpMats & {
                               return self.i_cls_trans_probs();
                             })
      .def_property_readonly("i_cls_rewards", [](const System &self) {
        return make_return_tensor(self.i_cls_rewards());
      });

  cls.def("train_v", &System::train_v)
      .def_property_readonly("v",
                             [](const System &self) {
                               return make_return_tensor(self.v().data(),
                                                         self.cls_dims.data(),
                                                         self.cls_dims.size());
                             })
      .def_property_readonly("v_policy", [](const System &self) {
        return make_return_tensor(self.v_policy().data(), self.cls_dims.data(),
                                  self.cls_dims.size());
      });

  cls.def("train_t", &System::train_t)
      .def_property_readonly("t_env_probs",
                             [](const System &self) {
                               return make_return_tensor(self.t_env_probs());
                             })
      .def_property_readonly("t_cls_trans_probs",
                             [](const System &self) -> const SpMats & {
                               return self.t_cls_trans_probs();
                             })
      .def_property_readonly("t_cls_rewards", [](const System &self) {
        return make_return_tensor(self.t_cls_rewards());
      });

  cls.def("to_file", &System::to_file)
      .def_static("from_file", &System::from_file)
      .def("to_str",
           [](const System &self) { return nb::bytes(self.to_str()); })
      .def_static("from_str",
                  [](const nb::bytes &str) { return System::from_str(str); })
      .def("__eq__", [](const System &self, const System &other) {
        return self == other;
      });
}

void make_queuing(nb::module_ &m) {
  auto cls = nb::class_<Queuing, System>(m, "Queuing");
  static constexpr auto init =
      [](Queuing *self, index_type n_env, index_type n_cls,
         nb::tensor<nb::numpy, index_type> limits, param_type costs,
         param_type arrivals, param_type departures, param_type env_trans_probs,
         Reward reward_type, const nb::kwargs &kwargs) {
        std::optional<float_type> normalized_c =
            PyDict_GetItemString(kwargs.ptr(), "normalized_c")
                ? std::make_optional(
                      nb::cast<float_type>(kwargs["normalized_c"]))
                : std::nullopt;

        auto reward_func = reward_func_type{};
        switch (reward_type) {
          case Reward::linear_2:
            reward_func = linear_reward_2;
            break;
          case Reward::convex_2:
            reward_func = [cost_eps = nb::cast<float_type>(kwargs["cost_eps"])](
                              const auto &costs, const auto &state,
                              index_type offset) {
              return convex_reward_2(costs, state, offset, cost_eps);
            };
            break;
        }
        new (self) Queuing(
            n_env, n_cls,
            Eigen::Map<VectorAI>(static_cast<index_type *>(limits.data()),
                                 limits.shape(0)),
            static_cast<float_type *>(costs.data()),
            static_cast<float_type *>(arrivals.data()),
            static_cast<float_type *>(departures.data()),
            static_cast<float_type *>(env_trans_probs.data()), reward_func,
            normalized_c);
      };
  cls.def("__init__", init);

  static constexpr auto copy_init = [](Queuing *self, const System &other) {
    new (self) Queuing(other);
  };
  cls.def("__init__", copy_init);
}

void make_loadbalance(nb::module_ &m) {
  auto cls = nb::class_<LoadBalance, System>(m, "LoadBalance");
  static constexpr auto init =
      [](LoadBalance *self, index_type n_env, index_type n_cls,
         nb::tensor<nb::numpy, index_type> limits, param_type costs,
         float_type arrival, param_type departures, param_type env_trans_probs,
         Reward reward_type, const nb::kwargs &kwargs) {
        std::optional<float_type> normalized_c =
            PyDict_GetItemString(kwargs.ptr(), "normalized_c")
                ? std::make_optional(
                      nb::cast<float_type>(kwargs["normalized_c"]))
                : std::nullopt;

        auto reward_func = reward_func_type{};
        switch (reward_type) {
          case Reward::linear_2:
            reward_func = linear_reward_2;
            break;
          case Reward::convex_2:
            reward_func = [cost_eps = nb::cast<float_type>(kwargs["cost_eps"])](
                              const auto &costs, const auto &state,
                              index_type offset) {
              return convex_reward_2(costs, state, offset, cost_eps);
            };
            break;
        }
        new (self) LoadBalance(
            n_env, n_cls,
            Eigen::Map<VectorAI>(static_cast<index_type *>(limits.data()),
                                 limits.shape(0)),
            static_cast<float_type *>(costs.data()), arrival,
            static_cast<float_type *>(departures.data()),
            static_cast<float_type *>(env_trans_probs.data()), reward_func,
            normalized_c);
      };
  cls.def("__init__", init);

  static constexpr auto copy_init = [](LoadBalance *self, const System &other) {
    new (self) LoadBalance(other);
  };
  cls.def("__init__", copy_init);
}

NB_MODULE(planning_ext, m) {
  m.def("index_type", []() {
     return make_return_tensor<index_type>(nullptr, 0);
   }).def("float_type", []() {
    return make_return_tensor<float_type>(nullptr, 0);
  });

  nb::enum_<Reward>(m, "Reward")
      .value("linear_2", Reward::linear_2)
      .value("convex_2", Reward::convex_2);

  make_state(m);
  make_system(m);
  make_queuing(m);
  make_loadbalance(m);
}

#include "planning/system.h"

#include <array>
#include <optional>
#include <string>
#include <utility>

#include "Eigen/SparseCore"
#include "nanobind/nanobind.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/vector.h"
#include "nanobind/tensor.h"

namespace nb = nanobind;

using namespace nb::literals;

template <typename data_type, typename... size_types>
static constexpr auto make_return_tensor(const data_type *data,
                                         size_types... sizes) {
  constexpr auto n_dim = sizeof...(sizes);
  const size_t dims[] = {static_cast<size_t>(sizes)...};
  return nb::tensor<nb::numpy, data_type>(const_cast<data_type *>(data), n_dim,
                                          dims);
}

template <typename data_type>
static constexpr auto make_return_tensor(const auto &v) {
  return make_return_tensor<data_type>(v.data(), static_cast<size_t>(v.size()));
}

namespace nanobind {
namespace detail {

template <typename Type, size_t N>
struct type_caster<std::array<Type, N>>
    : list_caster<std::array<Type, N>, Type> {};

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

    return sp_type(nb::make_tuple(
                       make_return_tensor<Scalar>(value.valuePtr(),
                                                  value.nonZeros()),
                       make_return_tensor<StorageIndex>(value.innerIndexPtr(),
                                                        value.nonZeros()),
                       make_return_tensor<StorageIndex>(value.outerIndexPtr(),
                                                        value.outerSize() + 1)),
                   nb::make_tuple(value.rows(), value.cols()))
        .release();
  }
};

}  // namespace detail
}  // namespace nanobind

template <typename system_type>
auto make_system_name(const std::string &prefix) {
  auto name = prefix + std::to_string(system_type::n_env);
  for (size_t i = 0; i < system_type::n_cls; ++i) {
    name += "_" + std::to_string(system_type::cls_dims[i] - 1);
  }
  return name;
}

template <typename system_type>
void make_system(nb::module_ &m) {
  using param_type = nb::tensor<nb::numpy, float_type>;

  auto cls = nb::class_<system_type>(
      m, make_system_name<system_type>("system_").c_str());
  using py_type = decltype(cls);

  /* --------------------------------- init --------------------------------- */

  static constexpr auto init = [](system_type *s, param_type costs,
                                  param_type arrivals, param_type departures,
                                  param_type env_trans_probs,
                                  Reward reward_type,
                                  const nb::kwargs &kwargs) {
    static constexpr auto offset = system_type::n_cls;

    std::optional<float_type> normalized_c =
        PyDict_GetItemString(kwargs.ptr(), "normalized_c")
            ? std::make_optional(nb::cast<float_type>(kwargs["normalized_c"]))
            : std::nullopt;

    auto reward_func = typename system_type::reward_func_type{};
    switch (reward_type) {
      case Reward::linear_2:
        reward_func = [](const auto &costs, const auto &state) {
          return linear_reward_2(costs, state, offset);
        };
        break;
      case Reward::convex_2:
        reward_func = [cost_eps = nb::cast<float_type>(kwargs["cost_eps"])](
                          const auto &costs, const auto &state) {
          return convex_reward_2(costs, state, offset, cost_eps);
        };
        break;
    }
    new (s) system_type(static_cast<float_type *>(costs.data()),
                        static_cast<float_type *>(arrivals.data()),
                        static_cast<float_type *>(departures.data()),
                        static_cast<float_type *>(env_trans_probs.data()),
                        reward_func, normalized_c);
  };

  cls.def("__init__", init);

  /* ------------------------- constexpr state types ------------------------ */

  auto state = nb::class_<typename system_type::state>(cls, "state");
  state
      .def_property_readonly_static("cls",
                                    [](const py_type &) {
                                      return make_return_tensor<index_type>(
                                          system_type::state::cls::f.data(),
                                          system_type::state::cls::n,
                                          system_type::state::cls::d);
                                    })
      .def_property_readonly_static("env",
                                    [](const py_type &) {
                                      return make_return_tensor<index_type>(
                                          system_type::state::env::f.data(),
                                          system_type::state::env::n,
                                          system_type::state::env::d);
                                    })
      .def_property_readonly_static("sys", [](const py_type &) {
        return make_return_tensor<index_type>(system_type::state::sys::f.data(),
                                              system_type::state::sys::n,
                                              system_type::state::sys::d);
      });

  /* -------------------- variables - system transitions -------------------- */

  cls.def_property_readonly(
      "trans_probs",
      [](const system_type &s) -> const system_type::trans_probs_type & {
        return s.trans_probs;
      });

  /* -------------------------- variables - rewards ------------------------- */

  cls.def_property_readonly("rewards", [](const system_type &s) {
    return make_return_tensor<float_type>(s.rewards);
  });

  /* ----------- variables - additional precomputed probabilities ----------- */

  cls.def_property_readonly(
      "env_trans_probs",
      [](const system_type &s) -> const system_type::env_trans_probs_type & {
        return s.env_trans_probs;
      });

  /* ---------------------- class states - interactive ---------------------- */

  cls.def_property_readonly("n_cls_visit", [](const system_type &s) {
    return make_return_tensor<uint64_t>(s.n_cls_visit());
  });

  /* ------------------------------ q learning ------------------------------ */

  cls.def("train_q",
          [](system_type &s, float_type gamma, float_type greedy_eps,
             uint64_t ls, uint64_t seed) {
            s.template train_q<false, false>(gamma, greedy_eps, ls, seed);
          })
      .def("train_q_i",
           [](system_type &s, float_type gamma, float_type greedy_eps,
              uint64_t ls, uint64_t seed) {
             s.template train_q<true, false>(gamma, greedy_eps, ls, seed);
           })
      .def("train_q_qs",
           [](system_type &s, float_type gamma, float_type greedy_eps,
              uint64_t ls, uint64_t seed) {
             s.template train_q<false, true>(gamma, greedy_eps, ls, seed);
           })
      .def("train_q_full",
           [](system_type &s, float_type gamma, float_type greedy_eps,
              uint64_t ls, uint64_t seed) {
             s.template train_q<true, true>(gamma, greedy_eps, ls, seed);
           })
      .def_property_readonly("q",
                             [](const system_type &s) {
                               return make_return_tensor<float_type>(s.q());
                             })
      .def_property_readonly(
          "q_policy",
          [](const system_type &s) {
            return make_return_tensor<index_type>(s.q_policy());
          })
      .def_property_readonly("qs",
                             [](const system_type &s) {
                               return make_return_tensor<float_type>(s.qs());
                             })
      .def_property_readonly("i_cls_trans_probs",
                             [](const system_type &s)
                                 -> const system_type::cls_trans_probs_type & {
                               return s.i_cls_trans_probs();
                             })
      .def_property_readonly("i_cls_rewards", [](const system_type &s) {
        return make_return_tensor<float_type>(s.i_cls_rewards());
      });

  /* ---------------------------- value iteration --------------------------- */

  cls.def("train_v", &system_type::train_v)
      .def_property_readonly("v",
                             [](const system_type &s) {
                               return make_return_tensor<float_type>(s.v());
                             })
      .def_property_readonly("v_policy", [](const system_type &s) {
        return make_return_tensor<index_type>(s.v_policy());
      });

  /* --------------------------------- tilde -------------------------------- */

  cls.def("train_t", &system_type::train_t)
      .def_property_readonly(
          "t_env_probs",
          [](const system_type &s) {
            return make_return_tensor<float_type>(s.t_env_probs());
          })
      .def_property_readonly("t_cls_trans_probs",
                             [](const system_type &s)
                                 -> const system_type::cls_trans_probs_type & {
                               return s.t_cls_trans_probs();
                             })
      .def_property_readonly("t_cls_rewards", [](const system_type &s) {
        return make_return_tensor<float_type>(s.t_cls_rewards());
      });
}

template <auto system_limits, size_t i_t, index_type... n_env_t>
void make_system_2_len_env(nb::module_ &m,
                           std::integer_sequence<index_type, n_env_t...>) {
  ((make_system<
       System<n_env_t + 1, system_limits[i_t][0], system_limits[i_t][1]>>(m)),
   ...);
}

template <auto system_limits, size_t... i_t>
void make_system_2_len(nb::module_ &m, std::index_sequence<i_t...>) {
  static constexpr auto seq = std::make_integer_sequence<index_type, 2>{};
  ((make_system_2_len_env<system_limits, i_t>(m, seq)), ...);
}

template <index_type... dims_t>
void make_system_2(nb::module_ &m) {
  static constexpr auto system_limits =
      std::array{std::array{dims_t, dims_t}...};
  make_system_2_len<system_limits>(
      m, std::make_index_sequence<system_limits.size()>{});
}

NB_MODULE(planning_ext, m) {
  nb::enum_<Reward>(m, "Reward")
      .value("linear_2", Reward::linear_2)
      .value("convex_2", Reward::convex_2);

  make_system_2<3, 7, 10, 15, 20, 25, 30, 40, 50>(m);
}

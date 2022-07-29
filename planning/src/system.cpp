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

template <typename Type, size_t N>
struct nb::detail::type_caster<std::array<Type, N>>
    : nb::detail::list_caster<std::array<Type, N>, Type> {};

template <typename system_type>
auto make_system_name() {
  return "system_" + std::to_string(system_type::n_env) + "_" +
         std::to_string(system_type::class_dims[0] - 1) + "_" +
         std::to_string(system_type::class_dims[1] - 1);
}

template <typename data_type>
static constexpr auto make_return_tensor(const data_type *data, size_t size) {
  constexpr auto n_dim = 1;
  const size_t dims[] = {size};
  return nb::tensor<nb::numpy, data_type>(const_cast<data_type *>(data), n_dim,
                                          dims);
}

template <typename data_type, bool condition_t = true>
static constexpr auto make_return_tensor(const auto &v) {
  if constexpr (condition_t) {
    return make_return_tensor<data_type>(v.data(),
                                         static_cast<size_t>(v.size()));
  }
}

template <typename float_type>
void make_sparse(nb::module_ &m) {
  using sp_vec_type = Eigen::SparseVector<float_type>;
  auto cls = nb::class_<sp_vec_type>(m, "sp_vec_type");
  cls.def("__len__", [](const sp_vec_type &v) { return v.nonZeros(); })
      .def("__getitem__",
           [](const sp_vec_type &v, size_t i) {
             return std::make_pair(v.innerIndexPtr()[i], v.valuePtr()[i]);
           })
      .def_property_readonly(
          "keys",
          [](const sp_vec_type &v) {
            return make_return_tensor<typename sp_vec_type::StorageIndex>(
                v.innerIndexPtr(), v.nonZeros());
          })
      .def_property_readonly("values", [](const sp_vec_type &v) {
        return make_return_tensor<float_type>(v.valuePtr(), v.nonZeros());
      });
}

template <typename float_type, size_t n_class_t, size_t n_env_t>
void make_probs_map(nb::module_ &m) {
  static constexpr auto n_env_state = ([]() {
    size_t res = 1;
    for (size_t i = 0; i < n_class_t; ++i) {
      res *= n_env_t;
    }
    return res;
  })();

  using probs_map_type =
      tsl::sparse_map<index_type, Eigen::Matrix<float_type, 1, n_env_state>>;

  nb::class_<probs_map_type>(
      m,
      ("state_cls_trans_probs_map_type_" + std::to_string(n_env_state)).c_str())
      .def("__len__", [](const probs_map_type &map) { return map.size(); })
      .def("__getitem__",
           [](probs_map_type &map, index_type k) {
             return make_return_tensor<float_type>(map[k]);
           })
      .def("keys", [](probs_map_type &map) {
        std::vector<index_type> keys;
        for (const auto &[k, _] : map) {
          keys.push_back(k);
        }
        return keys;
      });
}

void make_n_cls_trans_map(nb::module_ &m) {
  using n_cls_trans_type = tsl::sparse_map<index_type, uint64_t>;
  nb::class_<n_cls_trans_type>(m, "n_cls_trans_type")
      .def("__len__", [](const n_cls_trans_type &map) { return map.size(); })
      .def("__getitem__",
           [](n_cls_trans_type &map, index_type k) { return map[k]; })
      .def("keys",
           [](n_cls_trans_type &map) {
             std::vector<index_type> keys;
             for (const auto &[k, _] : map) {
               keys.push_back(k);
             }
             return keys;
           })
      .def("values", [](n_cls_trans_type &map) {
        std::vector<uint64_t> values;
        for (const auto &[_, v] : map) {
          values.push_back(v);
        }
        return values;
      });
}

template <typename system_type>
void make_system(nb::module_ &m) {
  using float_type = typename system_type::float_type;
  using param_type = nb::tensor<nb::numpy, float_type>;

  static const auto name = make_system_name<system_type>();

  auto cls = nb::class_<system_type>(m, name.c_str());
  using py_type = decltype(cls);

  /* --------------------------------- init --------------------------------- */

  static constexpr auto init = [](system_type *s, param_type costs,
                                  param_type arrivals, param_type departures,
                                  param_type env_trans_probs,
                                  Reward reward_type,
                                  const nb::kwargs &kwargs) {
    static constexpr auto offset = system_type::n_class;

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

  cls.def_property_readonly_static(
         "states", [](const py_type &) { return system_type::states; })
      .def_property_readonly_static(
          "cls_states", [](const py_type &) { return system_type::cls_states; })
      .def_property_readonly_static("env_states", [](const py_type &) {
        return system_type::env_states;
      });

  /* -------------------- variables - system transitions -------------------- */

  cls.def_property_readonly(
      "trans_probs",
      [](const system_type &s)
          -> const typename system_type::trans_probs_type & {
        return s.trans_probs;
      });

  /* -------------------------- variables - rewards ------------------------- */

  cls.def_property_readonly("rewards", [](const system_type &s) {
    return make_return_tensor<float_type>(s.rewards);
  });

  /* ----------- variables - additional precomputed probabilities ----------- */

  cls.def_property_readonly(
         "state_cls_trans_probs",
         [](const system_type &s)
             -> const typename system_type::state_cls_trans_probs_type & {
           return s.state_cls_trans_probs;
         })
      .def_property_readonly("env_trans_probs", [](const system_type &s) {
        return make_return_tensor<float_type, system_type::n_env != 1>(
            s.env_trans_probs);
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
      .def_property_readonly(
          "i_cls_trans_probs",
          [](const system_type &s)
              -> const typename system_type::cls_trans_probs_type & {
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
      .def_property_readonly(
          "t_cls_trans_probs",
          [](const system_type &s)
              -> const typename system_type::cls_trans_probs_type & {
            return s.t_cls_trans_probs();
          })
      .def_property_readonly("t_cls_rewards", [](const system_type &s) {
        return make_return_tensor<float_type>(s.t_cls_rewards());
      });
}

template <auto system_limits, typename f_t, size_t i_t, index_type... n_env_t>
void make_system_2_len_env(nb::module_ &m,
                           std::integer_sequence<index_type, n_env_t...>) {
  if constexpr (i_t == 0) {
    ((make_probs_map<f_t, 2, n_env_t + 1>(m)), ...);
    make_n_cls_trans_map(m);
  }

  ((make_system<
       System<n_env_t + 1, f_t, system_limits[i_t][0], system_limits[i_t][1]>>(
       m)),
   ...);
}

template <auto system_limits, typename f_t, size_t... i_t>
void make_system_2_len(nb::module_ &m, std::index_sequence<i_t...>) {
  static constexpr auto seq = std::make_integer_sequence<index_type, 2>{};
  ((make_system_2_len_env<system_limits, f_t, i_t>(m, seq)), ...);
}

template <typename f_t, bool product_t, index_type... dims_t>
void make_system_2(nb::module_ &m) {
  static constexpr auto n_dim_t = sizeof...(dims_t);
  static constexpr auto use_product = product_t && (n_dim_t == 2);
  static constexpr auto size = use_product ? (dims_t * ... * 1) : n_dim_t;
  static constexpr auto system_limits = ([]() {
    if constexpr (use_product) {
      return make_set_product<dims_t + 1 ...>();
    } else {
      return std::array<std::array<index_type, 2>, size>{
          std::array{dims_t, dims_t}...};
    }
  })();
  make_system_2_len<system_limits, f_t>(m, std::make_index_sequence<size>{});
}

NB_MODULE(planning_ext, m) {
  nb::enum_<Reward>(m, "Reward")
      .value("linear_2", Reward::linear_2)
      .value("convex_2", Reward::convex_2);

  using f_t = double;
  using sp_vec_type = Eigen::SparseVector<f_t>;

  make_sparse<f_t>(m);
  make_system_2<f_t, false, 3, 7, 10, 15, 20, 25, 30, 40, 50>(m);
}

#include "planning/env.h"

#include <array>
#include <span>
#include <string>

#include "nanobind/nanobind.h"
#include "nanobind/stl/vector.h"
#include "nanobind/tensor.h"

namespace nb = nanobind;

using namespace nb::literals;

template <bool condition, typename data_type>
static constexpr auto make_return_tensor(const auto &v) {
  if constexpr (condition) {
    constexpr auto n_dim = 1;
    const size_t dims[] = {static_cast<size_t>(v.size())};
    return nb::tensor<nb::numpy, data_type>(const_cast<data_type *>(v.data()),
                                            n_dim, dims);
  }
}

template <typename env_type>
auto make_env_name() {
  return "env_" + std::to_string(env_type::n_env) + "_" +
         std::to_string(env_type::save_qs) + "_" +
         std::to_string(env_type::dims_queue[0] - 1) + "_" +
         std::to_string(env_type::dims_queue[1] - 1);
}

template <typename env_type>
void make_env(nb::module_ &m) {
  using env_float_type = typename env_type::float_type;
  using param_type = nb::tensor<nb::numpy, env_float_type>;

  static const auto name = make_env_name<env_type>();

  auto cls = nb::class_<env_type>(m, name.c_str());

  static constexpr auto init = [](env_type *env, param_type env_cost,
                                  param_type env_arrival,
                                  param_type env_departure, param_type env_prob,
                                  Reward reward_type, env_float_type cost_eps) {
    static constexpr auto offset = env_type::offset_full_obs;

    auto reward_func = typename env_type::reward_func_type{};
    switch (reward_type) {
      case Reward::linear_2:
        reward_func = [](const auto &env_cost, const auto &state) {
          return linear_reward_2(env_cost, state, offset);
        };
        break;
      case Reward::convex_2:
        reward_func = [cost_eps](const auto &env_cost, const auto &state) {
          return convex_reward_2(env_cost, state, offset, cost_eps);
        };
        break;
    }
    new (env)
        env_type(static_cast<env_float_type *>(env_cost.data()),
                 static_cast<env_float_type *>(env_arrival.data()),
                 static_cast<env_float_type *>(env_departure.data()),
                 static_cast<env_float_type *>(env_prob.data()), reward_func);
  };

  cls.def("__init__", init);

  cls.def("train_q", &env_type::train_q)
      .def_property_readonly(
          "q",
          [](const env_type &e) {
            return make_return_tensor<true, env_float_type>(e.q());
          })
      .def_property_readonly(
          "n_visit",
          [](const env_type &e) {
            return make_return_tensor<true, uint64_t>(e.n_visit());
          })
      .def_property_readonly("qs", [](const env_type &e) {
        return make_return_tensor<env_type::save_qs, env_float_type>(e.qs());
      });

  cls.def("train_v", &env_type::train_v)
      .def_property_readonly(
          "v",
          [](const env_type &e) {
            return make_return_tensor<env_type::n_env == 1, env_float_type>(
                e.v());
          })
      .def_property_readonly("policy_v", [](const env_type &e) {
        return make_return_tensor<env_type::n_env == 1, index_type>(
            e.policy_v());
      });
}

template <auto env_dims_queue, typename f_t, size_t i_t, index_type... n_env_t>
void make_env_2_len_env(nb::module_ &m,
                        std::integer_sequence<index_type, n_env_t...>) {
  ((make_env<Env<n_env_t + 1, f_t, false, env_dims_queue[i_t][0],
                 env_dims_queue[i_t][1]>>(m)),
   ...);
  ((make_env<Env<n_env_t + 1, f_t, true, env_dims_queue[i_t][0],
                 env_dims_queue[i_t][1]>>(m)),
   ...);
}

template <auto env_dims_queue, typename f_t, size_t... i_t>
void make_env_2_len(nb::module_ &m, std::index_sequence<i_t...>) {
  static constexpr auto seq = std::make_integer_sequence<index_type, 2>{};
  ((make_env_2_len_env<env_dims_queue, f_t, i_t>(m, seq)), ...);
}

template <typename f_t, bool product_t, index_type... dims_t>
void make_env_2(nb::module_ &m) {
  static constexpr auto n_dim_t = sizeof...(dims_t);
  static constexpr auto use_product = product_t && (n_dim_t == 2);
  static constexpr auto size = use_product ? (dims_t * ... * 1) : n_dim_t;
  static constexpr auto env_dims_queue = ([]() {
    if constexpr (use_product) {
      return make_set_product<dims_t + 1 ...>();
    } else {
      return std::array<std::array<index_type, 2>, size>{
          std::array{dims_t, dims_t}...};
    }
  })();
  make_env_2_len<env_dims_queue, f_t>(m, std::make_index_sequence<size>{});
}

NB_MODULE(planning_ext, m) {
  nb::enum_<Reward>(m, "Reward")
      .value("linear_2", Reward::linear_2)
      .value("convex_2", Reward::convex_2);

  make_env_2<double, false, 3, 7, 10, 15, 20, 25, 30, 40, 50>(m);
}

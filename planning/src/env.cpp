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
  return std::string(env_type::env_name) + "_" +
         std::to_string(env_type::n_env) + "_" +
         std::to_string(env_type::save_qs) + "_" +
         std::to_string(env_type::dims_queue[0] - 1) + "_" +
         std::to_string(env_type::dims_queue[1] - 1);
}

template <typename env_type>
void make_env(nb::module_ &m) {
  using env_float_type = typename env_type::float_type;
  static const auto name = make_env_name<env_type>();
  auto cls =
      nb::class_<env_type>(m, name.c_str())
          .def(
              "__init__",
              [](env_type *env,
                 nb::tensor<nb::numpy, const env_float_type> env_cost,
                 nb::tensor<nb::numpy, const env_float_type> env_arrival,
                 nb::tensor<nb::numpy, const env_float_type> env_departure,
                 nb::tensor<nb::numpy, const env_float_type> env_prob,
                 env_float_type cost_eps) {
                if constexpr (env_is_convex<env_type>) {
                  new (env) env_type(
                      static_cast<const env_float_type *>(env_cost.data()),
                      static_cast<const env_float_type *>(env_arrival.data()),
                      static_cast<const env_float_type *>(env_departure.data()),
                      static_cast<const env_float_type *>(env_prob.data()),
                      cost_eps);
                } else {
                  new (env) env_type(
                      static_cast<const env_float_type *>(env_cost.data()),
                      static_cast<const env_float_type *>(env_arrival.data()),
                      static_cast<const env_float_type *>(env_departure.data()),
                      static_cast<const env_float_type *>(env_prob.data()));
                }
              })
          .def("train_q",
               [](env_type &e, env_float_type gamma, env_float_type eps,
                  env_float_type decay, size_t epoch, uint64_t ls,
                  uint64_t seed) {
                 e.train_q(gamma, eps, decay, epoch, ls, seed);
               })
          .def_property_readonly(
              "q",
              [](const env_type &e) {
                return make_return_tensor<true, env_float_type>(e.q());
              })
          .def_property_readonly(
              "n_visit",
              [](const env_type &e) {
                return make_return_tensor<true, index_type>(e.n_visit());
              })
          .def_property_readonly(
              "qs",
              [](const env_type &e) {
                return make_return_tensor<env_type::save_qs, env_float_type>(
                    e.qs());
              })
          .def("train_v", [](env_type &e, env_float_type gamma,
                             uint64_t ls) { e.train_v(gamma, ls); })
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

template <typename f_t, bool product_t, index_type... dims_t>
void make_env_2_queue(nb::module_ &m) {
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

  ([&m]<index_type... i>(std::integer_sequence<index_type, i...>) {
    const auto make_env_i = [&m]<index_type j, index_type... n_env_t>(
        std::integer_sequence<index_type, n_env_t...>) {
      const auto make_env_i_n_env =
          [&m]<index_type k, index_type n_env, index_type... save_qs_t>(
              std::integer_sequence<index_type, save_qs_t...>) {
        (make_env<LinearEnv<n_env, f_t, save_qs_t == 1, env_dims_queue[k][0],
                            env_dims_queue[k][1]>>(m),
         ...);
        (make_env<ConvexEnv<n_env, f_t, save_qs_t == 1, env_dims_queue[k][0],
                            env_dims_queue[k][1]>>(m),
         ...);
      };

      (make_env_i_n_env.template operator()<j, n_env_t + 1>(
           std::make_integer_sequence<index_type, 2>{}),
       ...);
    };

    (make_env_i.template operator()<i>(
         std::make_integer_sequence<index_type, 2>{}),
     ...);
  })(std::make_integer_sequence<index_type, size>{});
}

NB_MODULE(planning_ext, m) {
  make_env_2_queue<double, false, 3, 7, 10, 15, 20, 25, 30, 40, 50>(m);
}

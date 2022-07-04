#include "planning/env.h"

#include <array>
#include <span>
#include <string>

#include "nanobind/nanobind.h"
#include "nanobind/stl/vector.h"
#include "nanobind/tensor.h"

namespace nb = nanobind;

using namespace nb::literals;

template <bool condition, typename env_type, typename data_type,
          size_t... postfixes_t, size_t... i>
static constexpr auto make_return_tensor(const auto &v,
                                         std::index_sequence<i...>) {
  if constexpr (condition) {
    constexpr auto n_dim = env_type::n_queue + sizeof...(postfixes_t);
    constexpr size_t dims[] = {env_type::dims_queue[i]..., postfixes_t...};

    return nb::tensor<nb::numpy, data_type,
                      nb::shape<env_type::dims_queue[i]..., postfixes_t...>>(
        const_cast<data_type *>(v.data()), n_dim, dims);
  }
}

template <bool condition, typename env_type, typename data_type,
          size_t... postfixes_t, size_t... i>
static constexpr auto make_return_tensor(const auto &v, size_t n_first,
                                         std::index_sequence<i...>) {
  if constexpr (condition) {
    constexpr auto n_dim = 1 + env_type::n_queue + sizeof...(postfixes_t);
    const size_t dims[] = {n_first, env_type::dims_queue[i]..., postfixes_t...};

    return nb::tensor<
        nb::numpy, data_type,
        nb::shape<nb::any, env_type::dims_queue[i]..., postfixes_t...>>(
        const_cast<data_type *>(v.data()), n_dim, dims);
  }
}

template <typename env_type, size_t... i>
static constexpr auto make_from_array(std::index_sequence<i...>) {
  using env_float_type = typename env_type::float_type;
  return
      [](env_type &e,
         nb::tensor<nb::numpy, const env_float_type,
                    nb::shape<env_type::dims_queue[i]..., env_type::n_queue>>
             q,
         nb::tensor<nb::numpy, const size_t,
                    nb::shape<env_type::dims_queue[i]..., env_type::n_queue>>
             n_visit,
         nb::tensor<
             nb::numpy, const env_float_type,
             nb::shape<nb::any, env_type::dims_queue[i]..., env_type::n_queue>>
             qs,
         size_t qs_size) {
        e.from_array(static_cast<env_float_type *>(q.data()),
                     static_cast<size_t *>(n_visit.data()),
                     {static_cast<env_float_type *>(qs.data()), qs_size});
      };
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
                 nb::tensor<nb::numpy, const env_float_type,
                            nb::shape<env_type::n_queue, env_type::n_env>>
                     env_cost,
                 nb::tensor<nb::numpy, const env_float_type,
                            nb::shape<env_type::n_queue, env_type::n_env>>
                     env_arrival,
                 nb::tensor<nb::numpy, const env_float_type,
                            nb::shape<env_type::n_queue, env_type::n_env>>
                     env_departure,
                 nb::tensor<nb::numpy, const env_float_type,
                            nb::shape<env_type::n_queue, env_type::n_env>>
                     env_prob,
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
                  pcg32::state_type seed) {
                 e.train_q(gamma, eps, decay, epoch, ls, seed);
               })
          .def_property_readonly("q",
                                 [](const env_type &e) {
                                   return make_return_tensor<true, env_type,
                                                             env_float_type,
                                                             env_type::n_queue>(
                                       e.q(), env_type::idx_nq);
                                 })
          .def_property_readonly("n_visit",
                                 [](const env_type &e) {
                                   return make_return_tensor<true, env_type,
                                                             size_t,
                                                             env_type::n_queue>(
                                       e.n_visit(), env_type::idx_nq);
                                 })
          .def_property_readonly("qs",
                                 [](const env_type &e) {
                                   if constexpr (env_type::save_qs) {
                                     const auto n_first =
                                         e.qs().size() /
                                         (env_type::n_obs_state *
                                          env_type::n_queue);
                                     return make_return_tensor<
                                         env_type::save_qs, env_type,
                                         env_float_type, env_type::n_queue>(
                                         e.qs(), n_first, env_type::idx_nq);
                                   }
                                 })
          .def("train_v", [](env_type &e, env_float_type gamma,
                             uint64_t ls) { e.train_v(gamma, ls); })
          .def_property_readonly("v",
                                 [](const env_type &e) {
                                   return make_return_tensor<
                                       env_type::n_env == 1, env_type,
                                       env_float_type>(e.v(), env_type::idx_nq);
                                 })
          .def_property_readonly("policy_v",
                                 [](const env_type &e) {
                                   return make_return_tensor<
                                       env_type::n_env == 1, env_type, size_t>(
                                       e.policy_v(), env_type::idx_nq);
                                 })
          .def("from_array", make_from_array<env_type>(env_type::idx_nq));
}

template <typename f_t, bool product_t, size_t... dims_t>
void make_env_2_queue(nb::module_ &m) {
  static constexpr auto n_dim_t = sizeof...(dims_t);
  static constexpr auto use_product = product_t && (n_dim_t == 2);
  static constexpr auto size = use_product ? (dims_t * ... * 1) : n_dim_t;
  static constexpr auto env_dims_queue = ([]() {
    if constexpr (use_product) {
      return make_set_product<dims_t + 1 ...>();
    } else {
      return std::array<std::array<size_t, 2>, size>{
          std::array{dims_t, dims_t}...};
    }
  })();

  ([&m]<size_t... i>(std::index_sequence<i...>) {
    const auto make_env_i =
        [&m]<size_t j, size_t... n_env_t>(std::index_sequence<n_env_t...>) {
      const auto make_env_i_n_env =
          [&m]<size_t k, size_t n_env, size_t... save_qs_t>(
              std::index_sequence<save_qs_t...>) {
        (make_env<LinearEnv<n_env, f_t, save_qs_t == 1, env_dims_queue[k][0],
                            env_dims_queue[k][1]>>(m),
         ...);
        (make_env<ConvexEnv<n_env, f_t, save_qs_t == 1, env_dims_queue[k][0],
                            env_dims_queue[k][1]>>(m),
         ...);
      };

      (make_env_i_n_env.template operator()<j, n_env_t + 1>(
           std::make_index_sequence<2>{}),
       ...);
    };

    (make_env_i.template operator()<i>(std::make_index_sequence<2>{}), ...);
  })(std::make_index_sequence<size>{});
}

NB_MODULE(planning_ext, m) {
  make_env_2_queue<double, false, 3, 7, 10, 15, 20, 25, 30, 40, 50>(m);
}

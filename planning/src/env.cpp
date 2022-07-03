#include "planning/env.h"

#include <array>
#include <span>
#include <string>
#include <utility>

#include "nanobind/nanobind.h"
#include "nanobind/stl/vector.h"
#include "nanobind/tensor.h"

namespace nb = nanobind;

using namespace nb::literals;

template <typename env_type, typename data_type, typename vec_type, size_t... i>
constexpr auto gen_q(const vec_type &v, std::index_sequence<i...>) {
  constexpr auto dim = std::array<size_t, env_type::n_queue + 1>{
      env_type::dims_queue[i]..., env_type::n_queue};

  return nb::tensor<nb::numpy, data_type,
                    nb::shape<env_type::dims_queue[i]..., env_type::n_queue>>(
      v.data(), env_type::n_queue + 1, dim.data());
}

template <typename env_type, typename data_type, typename vec_type, size_t... i>
constexpr auto gen_qs(const vec_type &v, std::index_sequence<i...>) {
  if constexpr (env_type::save_qs) {
    const auto n_first = v.size() / (env_type::n_obs_state * env_type::n_queue);
    const auto dim = std::array<size_t, env_type::n_queue + 2>{
        n_first, env_type::dims_queue[i]..., env_type::n_queue};

    return nb::tensor<
        nb::numpy, data_type,
        nb::shape<nb::any, env_type::dims_queue[i]..., env_type::n_queue>>(
        const_cast<data_type *>(v.data()), env_type::n_queue + 2, dim.data());
  } else {
    return nullptr;
  }
}

template <typename env_type, size_t... i>
constexpr auto gen_from_array(std::index_sequence<i...>) {
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

template <typename env_type, const char *prefix>
const auto gen_env(nb::module_ &m) {
  static const auto name = prefix + std::to_string(env_type::n_env) + "_" +
                           std::to_string(env_type::save_qs) + "_" +
                           std::to_string(env_type::dims_queue[0] - 1) + "_" +
                           std::to_string(env_type::dims_queue[1] - 1);

  using env_float_type = typename env_type::float_type;
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
                                   return gen_q<env_type, env_float_type>(
                                       e.q(), env_type::idx_nq);
                                 })
          .def_property_readonly("n_visit",
                                 [](const env_type &e) {
                                   return gen_q<env_type, size_t>(
                                       e.n_visit(), env_type::idx_nq);
                                 })
          .def_property_readonly("qs",
                                 [](const env_type &e) {
                                   return gen_qs<env_type, env_float_type>(
                                       e.qs(), env_type::idx_nq);
                                 })
          .def("from_array", gen_from_array<env_type>(env_type::idx_nq));
}

template <
    template <size_t n_env_t, typename F, bool save_qs_t, size_t... max_ls>
    class EnvT,
    size_t n_env_t, typename f_t, const char *prefix, bool save_qs, size_t i,
    size_t sj, size_t... j>
const auto gen_env_1d(nb::module_ &m, std::integer_sequence<size_t, j...>) {
  (gen_env<EnvT<n_env_t, f_t, save_qs, i, j + sj>, prefix>(m), ...);
}

template <
    template <size_t n_env_t, typename F, bool save_qs_t, size_t... max_ls>
    class EnvT,
    typename f_t, const char *prefix, size_t si, size_t sj, size_t... i,
    size_t... j>
const auto gen_env_2d(nb::module_ &m, std::integer_sequence<size_t, i...>,
                      std::integer_sequence<size_t, j...> js) {
  (gen_env_1d<EnvT, 1, f_t, prefix, true, i + si, sj>(m, js), ...);
  (gen_env_1d<EnvT, 1, f_t, prefix, false, i + si, sj>(m, js), ...);
  (gen_env_1d<EnvT, 2, f_t, prefix, true, i + si, sj>(m, js), ...);
  (gen_env_1d<EnvT, 2, f_t, prefix, false, i + si, sj>(m, js), ...);
}

template <
    template <size_t n_env_t, typename F, bool save_qs_t, size_t... max_ls>
    class EnvT,
    typename f_t, const char *prefix, size_t si, size_t... i>
const auto gen_env_square(nb::module_ &m, std::integer_sequence<size_t, i...>) {
  (gen_env<EnvT<1, f_t, true, i + si, i + si>, prefix>(m), ...);
  (gen_env<EnvT<1, f_t, false, i + si, i + si>, prefix>(m), ...);
  (gen_env<EnvT<2, f_t, true, i + si, i + si>, prefix>(m), ...);
  (gen_env<EnvT<2, f_t, false, i + si, i + si>, prefix>(m), ...);
}

NB_MODULE(planning_ext, m) {
  static constexpr auto is =
      std::index_sequence<3, 7, 10, 15, 20, 25, 30, 40, 50>{};
  using f_t = double;

  static constexpr char linear_prefix[] = "linear_env_";
  static constexpr char convex_prefix[] = "convex_env_";

  gen_env_square<LinearEnv, f_t, linear_prefix, 0>(m, is);
  gen_env_square<ConvexEnv, f_t, convex_prefix, 0>(m, is);
}

#include "planning/env.h"

#include <memory>
#include <string>

#include "nanobind/nanobind.h"
#include "nanobind/stl/vector.h"
#include "nanobind/tensor.h"

namespace nb = nanobind;

using namespace nb::literals;

template <typename T, int_type... i>
constexpr auto gen_shape(std::integer_sequence<int_type, i...>) {
  return nb::shape<T::q_dim[i]...>{};
}

template <typename env_type, typename data_type, typename vec_type,
          int_type... i>
constexpr auto gen_q(const vec_type &v, std::integer_sequence<int_type, i...>) {
  constexpr auto dim = std::array<size_t, env_type::n_queue + 1>{
      env_type::dim_queue[i]..., env_type::n_queue};
  return nb::tensor<nb::numpy, data_type,
                    nb::shape<env_type::dim_queue[i]..., env_type::n_queue>>(
      v.data(), env_type::n_queue + 1, dim.data());
}

template <typename env_type, typename data_type, typename vec_type,
          int_type... i>
constexpr auto gen_qs(const vec_type &v,
                      std::integer_sequence<int_type, i...>) {
  const auto n_first = v.size() / env_type::n_total;
  return nb::tensor<
      nb::numpy, data_type,
      nb::shape<nb::any, env_type::dim_queue[i]..., env_type::n_queue>>(
      const_cast<data_type *>(v.data()), env_type::n_queue + 2,
      std::array<size_t, env_type::n_queue + 2>{
          n_first, env_type::dim_queue[i]..., env_type::n_queue}
          .data());
}

template <typename env_type, typename data_type, typename vec_type,
          int_type... i>
constexpr auto gen_reward(const vec_type &v,
                          std::integer_sequence<int_type, i...>) {
  constexpr auto dim =
      std::array<size_t, env_type::n_queue>{env_type::dim_queue[i]...};
  return nb::tensor<nb::numpy, data_type, nb::shape<env_type::dim_queue[i]...>>(
      const_cast<data_type *>(v.data()), env_type::n_queue, dim.data());
}

template <typename env_type, int_type... i>
constexpr auto gen_from_array(std::integer_sequence<int_type, i...>) {
  using env_float_type = typename env_type::float_type;
  return
      [](env_type &e,
         nb::tensor<nb::numpy, env_float_type,
                    nb::shape<env_type::dim_queue[i]..., env_type::n_queue>>
             q,
         nb::tensor<nb::numpy, int_type,
                    nb::shape<env_type::dim_queue[i]..., env_type::n_queue>>
             n_visit,
         nb::tensor<
             nb::numpy, env_float_type,
             nb::shape<nb::any, env_type::dim_queue[i]..., env_type::n_queue>>
             qs,
         size_t qs_size) {
        e.from_array(static_cast<env_float_type *>(q.data()),
                     static_cast<int_type *>(n_visit.data()),
                     static_cast<env_float_type *>(qs.data()), qs_size);
      };
}

template <typename env_type, const char *prefix>
const auto gen_env(nb::module_ &m) {
  static const auto name = prefix + std::to_string(env_type::n_env) + "_" +
                           std::to_string(env_type::save_qs) + "_" +
                           std::to_string(env_type::dim_queue[0] - 1) + "_" +
                           std::to_string(env_type::dim_queue[1] - 1);

  using env_float_type = typename env_type::float_type;
  auto cls =
      nb::class_<env_type>(m, name.c_str())
          .def("__init__",
               [](env_type *env,
                  nb::tensor<nb::numpy, env_float_type,
                             nb::shape<env_type::n_env, env_type::n_queue>>
                      env_cost,
                  nb::tensor<nb::numpy, env_float_type,
                             nb::shape<env_type::n_env, env_type::n_queue, 2>>
                      env_param,
                  nb::tensor<nb::numpy, env_float_type,
                             nb::shape<env_type::n_env, env_type::n_queue>>
                      env_prob,
                  env_float_type cost_eps) {
                 if constexpr (env_is_convex<env_type>) {
                   new (env)
                       env_type(static_cast<env_float_type *>(env_cost.data()),
                                static_cast<env_float_type *>(env_param.data()),
                                static_cast<env_float_type *>(env_prob.data()),
                                cost_eps);
                 } else {
                   new (env)
                       env_type(static_cast<env_float_type *>(env_cost.data()),
                                static_cast<env_float_type *>(env_param.data()),
                                static_cast<env_float_type *>(env_prob.data()));
                 }
               })
          .def("train",
               [](env_type &e, env_float_type gamma, env_float_type eps,
                  env_float_type decay, int_type epoch, uint64_t ls,
                  env_float_type lr_pow, pcg32::state_type seed) {
                 e.train(gamma, eps, decay, epoch, ls, lr_pow, seed);
               })
          .def_property_readonly("q",
                                 [](const env_type &e) {
                                   return gen_q<env_type, env_float_type>(
                                       e.q(), env_type::idx_nq);
                                 })
          .def_property_readonly("n_visit",
                                 [](const env_type &e) {
                                   return gen_q<env_type, int_type>(
                                       e.n_visit(), env_type::idx_nq);
                                 })
          .def_property_readonly("qs",
                                 [](const env_type &e) {
                                   return gen_qs<env_type, env_float_type>(
                                       e.qs(), env_type::idx_nq);
                                 })
          .def("from_array", gen_from_array<env_type>(env_type::idx_nq))
          .def_property_readonly("exceed_size", [](const env_type &e) {
            return env_type::exceed_size;
          });
  if constexpr (!env_type::exceed_size) {
    cls.def_property_readonly(
           "full_state_indices",
           [](const env_type &e) {
             static constexpr std::array<size_t, 2> dims{
                 env_type::n_combination, env_type::n_full_dim};
             static constexpr auto res = ([]() {
               std::array<int_type,
                          env_type::n_combination * env_type::n_full_dim>
                   res;
               for (int_type i = 0; i < env_type::n_combination; ++i) {
                 std::copy_n(env_type::full_state_indicies[i].begin(),
                             env_type::n_full_dim,
                             res.begin() + i * env_type::n_full_dim);
               }
               return res;
             })();
             return nb::tensor<
                 nb::numpy, int_type,
                 nb::shape<env_type::n_combination, env_type::n_full_dim>>(
                 const_cast<int_type *>(res.data()), 2, dims.data());
           })
        .def("init_reward_vec",
             [](env_type &e) {
               [[maybe_unused]] static const auto init = ([&e]() {
                 e.init_reward_vec();
                 return true;
               })();
             })
        .def_property_readonly(
            "reward_vec",
            [](const env_type &e) {
              static constexpr std::array<size_t, 1> dims{
                  env_type::n_combination};
              return nb::tensor<nb::numpy, env_float_type,
                                nb::shape<env_type::n_combination>>(
                  const_cast<env_float_type *>(e.reward_vec().data()), 1,
                  dims.data());
            })
        .def("init_prob_mat",
             [](env_type &e) {
               [[maybe_unused]] static const auto init = ([&e]() {
                 e.init_prob_mat();
                 return true;
               })();
             })
        .def_property_readonly(
            "prob_mat",
            [](env_type &e) {
              static constexpr std::array<size_t, 3> dims{
                  env_type::n_combination, env_type::n_combination,
                  env_type::n_queue};
              return nb::tensor<
                  nb::numpy, env_float_type,
                  nb::shape<env_type::n_combination, env_type::n_combination,
                            env_type::n_queue>>(
                  const_cast<env_float_type *>(e.prob_mat().data()), 3,
                  dims.data());
            })
        .def("train_v", [](env_type &e, uint64_t ls,
                           env_float_type lr) { e.train_v(ls, lr); })
        .def_property_readonly(
            "v",
            [](const env_type &e) {
              static constexpr std::array<size_t, 1> dims{
                  env_type::n_combination};
              return nb::tensor<nb::numpy, env_float_type,
                                nb::shape<env_type::n_combination>>(
                  const_cast<env_float_type *>(e.v().data()), 1, dims.data());
            })
        .def_property_readonly("policy_v", [](const env_type &e) {
          static constexpr std::array<size_t, 1> dims{env_type::n_combination};
          return nb::tensor<nb::numpy, int_type,
                            nb::shape<env_type::n_combination>>(
              const_cast<int_type *>(e.policy_v().data()), 1, dims.data());
        });
  }
}

template <
    template <int_type n_env_t, typename F, bool save_qs_t, int_type... max_ls>
    class EnvT,
    int_type n_env_t, typename f_t, const char *prefix, bool save_qs,
    int_type i, int_type sj, int_type... j>
const auto gen_env_1d(nb::module_ &m, std::integer_sequence<int_type, j...>) {
  (gen_env<EnvT<n_env_t, f_t, save_qs, i, j + sj>, prefix>(m), ...);
}

template <
    template <int_type n_env_t, typename F, bool save_qs_t, int_type... max_ls>
    class EnvT,
    typename f_t, const char *prefix, int_type si, int_type sj, int_type... i,
    int_type... j>
const auto gen_env_2d(nb::module_ &m, std::integer_sequence<int_type, i...>,
                      std::integer_sequence<int_type, j...> js) {
  (gen_env_1d<EnvT, 1, f_t, prefix, true, i + si, sj>(m, js), ...);
  (gen_env_1d<EnvT, 1, f_t, prefix, false, i + si, sj>(m, js), ...);
  (gen_env_1d<EnvT, 2, f_t, prefix, true, i + si, sj>(m, js), ...);
  (gen_env_1d<EnvT, 2, f_t, prefix, false, i + si, sj>(m, js), ...);
}

template <
    template <int_type n_env_t, typename F, bool save_qs_t, int_type... max_ls>
    class EnvT,
    typename f_t, const char *prefix, int_type si, int_type... i>
const auto gen_env_square(nb::module_ &m,
                          std::integer_sequence<int_type, i...>) {
  (gen_env<EnvT<1, f_t, true, i + si, i + si>, prefix>(m), ...);
  (gen_env<EnvT<1, f_t, false, i + si, i + si>, prefix>(m), ...);
  (gen_env<EnvT<2, f_t, true, i + si, i + si>, prefix>(m), ...);
  (gen_env<EnvT<2, f_t, false, i + si, i + si>, prefix>(m), ...);
}

NB_MODULE(planning_ext, m) {
  static constexpr auto is =
      std::integer_sequence<int_type, 3, 7, 10, 15, 20, 25, 30, 40, 50>{};
  using f_t = double;

  static constexpr char linear_prefix[] = "linear_env_";
  static constexpr char convex_prefix[] = "convex_env_";

  gen_env_square<LinearEnv, f_t, linear_prefix, 0>(m, is);
  gen_env_square<ConvexEnv, f_t, convex_prefix, 0>(m, is);
}

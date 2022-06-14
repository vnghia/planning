#include "planning/env.h"

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
      env_type::dim_ls[i]..., env_type::n_queue};
  return nb::tensor<nb::numpy, data_type,
                    nb::shape<env_type::dim_ls[i]..., env_type::n_queue>>(
      v.data(), env_type::n_queue + 1, dim.data());
}

template <typename env_type, typename data_type, typename vec_type,
          int_type... i>
constexpr auto gen_qs(const vec_type &v,
                      std::integer_sequence<int_type, i...>) {
  const auto n_first = v.size() / env_type::n_total;
  return nb::tensor<
      nb::numpy, data_type,
      nb::shape<nb::any, env_type::dim_ls[i]..., env_type::n_queue>>(
      const_cast<data_type *>(v.data()), env_type::n_queue + 2,
      std::array<size_t, env_type::n_queue + 2>{n_first, env_type::dim_ls[i]...,
                                                env_type::n_queue}
          .data());
}

template <typename env_type>
constexpr auto gen_env(nb::module_ &m, const std::string &name) {
  using env_float_type = typename env_type::float_type;
  nb::class_<env_type>(m, name.c_str())
      .def(nb::init<const typename env_type::std_vector_f_type &,
                    const typename env_type::std_vector_f_type &,
                    const typename env_type::std_vector_f_type &>())
      .def(
          "train",
          [](env_type &e, env_float_type gamma = 0.9, env_float_type eps = 0.01,
             env_float_type decay = 0.5, int_type epoch = 1,
             int_type ls = 1000000, env_float_type lr_pow = 0.51) {
            e.train(gamma, eps, decay, epoch, ls, lr_pow);
          },
          "gamma"_a = 0.9, "eps"_a = 0.01, "decay"_a = 0.5, "epoch"_a = 1,
          "ls"_a = 1000000, "lr_pow"_a = 0.51)
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
      .def_property_readonly("qs", [](const env_type &e) {
        return gen_qs<env_type, env_float_type>(e.qs(), env_type::idx_nq);
      });
}

template <bool save_q, int_type i, int_type j>
constexpr auto gen_linear_env(nb::module_ &m) {
  gen_env<LinearEnv<double, save_q, i, j>>(
      m, "linear_env_" + std::to_string(save_q) + "_" + std::to_string(i) +
             "_" + std::to_string(j));
}

template <bool save_q, int_type i, int_type sj, int_type... j>
constexpr auto gen_linear_env_1d(nb::module_ &m,
                                 std::integer_sequence<int_type, j...>) {
  (gen_linear_env<save_q, i, j + sj>(m), ...);
}

template <int_type si, int_type sj, int_type... i, int_type... j>
constexpr auto gen_linear_env_2d(nb::module_ &m,
                                 std::integer_sequence<int_type, i...>,
                                 std::integer_sequence<int_type, j...> js) {
  (gen_linear_env_1d<true, i + si, sj>(m, js), ...);
  (gen_linear_env_1d<false, i + si, sj>(m, js), ...);
}

template <bool save_q, int_type i, int_type j>
constexpr auto gen_convex_env(nb::module_ &m) {
  gen_env<ConvexEnv<double, save_q, i, j>>(
      m, "convex_env_" + std::to_string(save_q) + "_" + std::to_string(i) +
             "_" + std::to_string(j));
}

template <bool save_q, int_type i, int_type sj, int_type... j>
constexpr auto gen_convex_env_1d(nb::module_ &m,
                                 std::integer_sequence<int_type, j...>) {
  (gen_convex_env<save_q, i, j + sj>(m), ...);
}

template <int_type si, int_type sj, int_type... i, int_type... j>
constexpr auto gen_convex_env_2d(nb::module_ &m,
                                 std::integer_sequence<int_type, i...>,
                                 std::integer_sequence<int_type, j...> js) {
  (gen_convex_env_1d<true, i + si, sj>(m, js), ...);
  (gen_convex_env_1d<false, i + si, sj>(m, js), ...);
}

NB_MODULE(planning_ext, m) {
  static constexpr auto is = std::integer_sequence<int_type, 3, 6, 9>{};
  static constexpr auto js = is;
  static constexpr int_type si = 0;
  static constexpr int_type sj = 0;
  gen_linear_env_2d<si, sj>(m, is, js);
  gen_convex_env_2d<si, sj>(m, is, js);
}

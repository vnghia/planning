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
  nb::class_<env_type>(m, name.c_str())
      .def(nb::init<const typename env_type::std_vector_f_type &,
                    const typename env_type::std_vector_f_type &,
                    const typename env_type::std_vector_f_type &>())
      .def("train", &env_type::train, "gamma"_a = 0.9, "eps"_a = 0.01,
           "decay"_a = 0.5, "epoch"_a = 1, "ls"_a = 1000000, "lr_pow"_a = 0.51)
      .def_property_readonly(
          "q",
          [](const env_type &e) {
            return gen_q<env_type, typename env_type::float_type>(
                e.q(), env_type::idx_nq);
          })
      .def_property_readonly("n_visit",
                             [](const env_type &e) {
                               return gen_q<env_type, int_type>(
                                   e.n_visit(), env_type::idx_nq);
                             })
      .def_property_readonly("qs", [](const env_type &e) {
        return gen_qs<env_type, typename env_type::float_type>(
            e.qs(), env_type::idx_nq);
      });
}

template <bool save_q, int_type i, int_type j>
constexpr auto gen_linear_env(nb::module_ &m) {
  gen_env<Env<2, double, save_q, i, j>>(
      m, "linear_env_" + std::to_string(save_q) + "_" + std::to_string(i) +
             "_" + std::to_string(j));
}

template <bool save_q, int_type i, int_type... j>
constexpr auto gen_linear_env_1d(nb::module_ &m,
                                 std::integer_sequence<int_type, j...>) {
  (gen_linear_env<save_q, i, j + 2>(m), ...);
}

template <int_type... i, int_type... j>
constexpr auto gen_linear_env_2d(nb::module_ &m,
                                 std::integer_sequence<int_type, i...>,
                                 std::integer_sequence<int_type, j...> js) {
  (gen_linear_env_1d<true, i + 2>(m, js), ...);
  (gen_linear_env_1d<false, i + 2>(m, js), ...);
}

NB_MODULE(planning_ext, m) {
  gen_linear_env_2d(m, std::make_integer_sequence<int_type, 15>{},
                    std::make_integer_sequence<int_type, 15>{});
}

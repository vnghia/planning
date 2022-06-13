#include "planning/env.h"

#include <string>

#include "nanobind/nanobind.h"
#include "nanobind/stl/vector.h"
#include "nanobind/tensor.h"

namespace nb = nanobind;

using namespace nb::literals;

template <typename T, int_type... i>
constexpr auto gen_shape(std::integer_sequence<int_type, i...>) {
  return nb::shape<T::dim_ls[i]..., T::n_queue>{};
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
            return nb::tensor<nb::numpy, typename env_type::float_type,
                              decltype(gen_shape<env_type>(env_type::idx_nq))>(
                e.q().data(), env_type::n_queue + 1, env_type::q_dim.data());
          })
      .def_property_readonly("n_visit", [](const env_type &e) {
        return nb::tensor<nb::numpy, int_type,
                          decltype(gen_shape<env_type>(env_type::idx_nq))>(
            e.n_visit().data(), env_type::n_queue + 1, env_type::q_dim.data());
      });
}

template <int_type i, int_type j>
constexpr auto gen_linear_env(nb::module_ &m) {
  using env_type = Env<2, double, i, j>;
  gen_env<Env<2, double, i, j>>(
      m, "linear_env_" + std::to_string(i) + "_" + std::to_string(j));
}

template <int_type i, int_type... j>
constexpr auto gen_linear_env_1d(nb::module_ &m,
                                 std::integer_sequence<int_type, j...>) {
  (gen_linear_env<i, j + 2>(m), ...);
}

template <int_type... i, int_type... j>
constexpr auto gen_linear_env_2d(nb::module_ &m,
                                 std::integer_sequence<int_type, i...>,
                                 std::integer_sequence<int_type, j...> js) {
  (gen_linear_env_1d<i + 2>(m, js), ...);
}

NB_MODULE(planning_ext, m) {
  gen_linear_env_2d(m, std::make_integer_sequence<int_type, 20>{},
                    std::make_integer_sequence<int_type, 20>{});
}

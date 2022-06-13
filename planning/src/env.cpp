#include "planning/env.h"

#include "nanobind/nanobind.h"
#include "nanobind/stl/vector.h"
#include "nanobind/tensor.h"

namespace nb = nanobind;

using namespace nb::literals;
using SmallEnv = Env<2, double, 3, 3>;
using env_type = SmallEnv;

template <int_type... i>
constexpr auto gen_shape(std::integer_sequence<int_type, i...>) {
  return nb::shape<env_type::dim_ls[i]..., env_type::n_queue>{};
}

NB_MODULE(planning_ext, m) {
  auto cls = nb::class_<env_type>(m, "SmallEnv")
                 .def(nb::init<const env_type::std_vector_f_type &,
                               const env_type::std_vector_f_type &,
                               const env_type::std_vector_f_type &>())
                 .def("train", &env_type::train, "gamma"_a = 0.9,
                      "eps"_a = 0.01, "decay"_a = 0.5, "epoch"_a = 1,
                      "ls"_a = 1000000, "lr_pow"_a = 0.51)
                 .def_property_readonly(
                     "q",
                     [](const env_type &e) {
                       return nb::tensor<nb::numpy, env_type::float_type,
                                         decltype(gen_shape(env_type::idx_nq))>(
                           e.q().data(), env_type::n_queue + 1,
                           env_type::q_dim.data());
                     })
                 .def_property_readonly("n_visit", [](const env_type &e) {
                   return nb::tensor<nb::numpy, int_type,
                                     decltype(gen_shape(env_type::idx_nq))>(
                       e.n_visit().data(), env_type::n_queue + 1,
                       env_type::q_dim.data());
                 });
}

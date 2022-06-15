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

template <typename env_type, int_type... i>
constexpr auto gen_from_array(std::integer_sequence<int_type, i...>) {
  using env_float_type = typename env_type::float_type;
  return [](env_type &e,
            nb::tensor<nb::numpy, env_float_type,
                       nb::shape<env_type::dim_ls[i]..., env_type::n_queue>>
                q,
            nb::tensor<nb::numpy, int_type,
                       nb::shape<env_type::dim_ls[i]..., env_type::n_queue>>
                n_visit,
            nb::tensor<
                nb::numpy, env_float_type,
                nb::shape<nb::any, env_type::dim_ls[i]..., env_type::n_queue>>
                qs,
            size_t qs_size) {
    e.from_array(static_cast<env_float_type *>(q.data()),
                 static_cast<int_type *>(n_visit.data()),
                 static_cast<env_float_type *>(qs.data()), qs_size);
  };
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
             uint64_t ls = 20000000, env_float_type lr_pow = 0.51,
             std::mt19937_64::result_type seed = 42) {
            e.train(gamma, eps, decay, epoch, ls, lr_pow, seed);
          },
          "gamma"_a = 0.9, "eps"_a = 0.01, "decay"_a = 0.5, "epoch"_a = 1,
          "ls"_a = 20000000, "lr_pow"_a = 0.51, "seed"_a = 42)
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
      .def("from_array", gen_from_array<env_type>(env_type::idx_nq));
}

template <typename EnvT, const char *prefix>
constexpr auto gen_env(nb::module_ &m) {
  gen_env<EnvT>(m, prefix + std::to_string(EnvT::save_qs) + "_" +
                       std::to_string(EnvT::dim_ls[0] - 1) + "_" +
                       std::to_string(EnvT::dim_ls[1] - 1));
}

template <template <typename F, bool save_qs_t, int_type... max_ls> class EnvT,
          typename f_t, const char *prefix, bool save_qs, int_type i,
          int_type sj, int_type... j>
constexpr auto gen_env_1d(nb::module_ &m,
                          std::integer_sequence<int_type, j...>) {
  (gen_env<EnvT<f_t, save_qs, i, j + sj>, prefix>(m), ...);
}

template <template <typename F, bool save_qs_t, int_type... max_ls> class EnvT,
          typename f_t, const char *prefix, int_type si, int_type sj,
          int_type... i, int_type... j>
constexpr auto gen_env_2d(nb::module_ &m, std::integer_sequence<int_type, i...>,
                          std::integer_sequence<int_type, j...> js) {
  (gen_env_1d<EnvT, f_t, prefix, true, i + si, sj>(m, js), ...);
  (gen_env_1d<EnvT, f_t, prefix, false, i + si, sj>(m, js), ...);
}

template <template <typename F, bool save_qs_t, int_type... max_ls> class EnvT,
          typename f_t, const char *prefix, int_type si, int_type... i>
constexpr auto gen_env_square(nb::module_ &m,
                              std::integer_sequence<int_type, i...>) {
  (gen_env<EnvT<f_t, true, i + si, i + si>, prefix>(m), ...);
  (gen_env<EnvT<f_t, false, i + si, i + si>, prefix>(m), ...);
}

NB_MODULE(planning_ext, m) {
  static constexpr auto is = std::make_integer_sequence<int_type, 18>{};
  static constexpr int_type si = 3;
  using f_t = double;

  static constexpr char linear_prefix[] = "linear_env_";
  gen_env_square<LinearEnv, f_t, linear_prefix, si>(m, is);

  static constexpr char convex_prefix[] = "convex_env_";
  gen_env_square<ConvexEnv, f_t, convex_prefix, si>(m, is);
}

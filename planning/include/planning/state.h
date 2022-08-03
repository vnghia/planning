#pragma once

#include <array>
#include <optional>
#include <utility>

#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"

using index_type = Eigen::Index;
static constexpr auto storage_order = Eigen::RowMajor;

template <index_type begin, index_type end>
static constexpr auto make_iota() {
  std::array<index_type, end - begin> res;
  std::iota(res.begin(), res.end(), begin);
  return res;
}

template <index_type... lens_t>
struct CartesianProduct {
  static constexpr auto n = (lens_t * ... * 1);
  static constexpr auto d = sizeof...(lens_t);
  static constexpr auto f = ([]() {
    constexpr auto lens = std::array{lens_t...};
    std::array<index_type, n * d> f;

    for (index_type i = 0; i < n; ++i) {
      auto cur = i;

      for (index_type j = 0; j < d; ++j) {
        const auto rj = d - 1 - j;
        f[i * d + rj] = cur % lens[rj];
        cur /= lens[rj];
      }
    }

    return f;
  })();

  using mat_type =
      Eigen::Map<const Eigen::Matrix<index_type, n, d, storage_order>>;
  using row_type = typename mat_type::ConstRowXpr;
  static inline const auto a = mat_type(f.data());
};

template <index_type n_env_t, index_type... limits_t>
struct State {
  static constexpr auto n_env = n_env_t;

  static constexpr index_type n_cls = sizeof...(limits_t);
  static constexpr auto cls_dims = std::array{limits_t + 1 ...};

  static constexpr auto seq_n_cls = std::make_index_sequence<n_cls>{};
  using cls = decltype(([]<size_t... i>(std::index_sequence<i...>) {
    return CartesianProduct<cls_dims[i]...>{};
  })(seq_n_cls));
  using env = decltype(([]<size_t... i>(std::index_sequence<i...>) {
    return CartesianProduct<n_env*(i - i + 1)...>{};
  })(seq_n_cls));
  using sys = decltype(([]<size_t... i>(std::index_sequence<i...>) {
    return CartesianProduct<cls_dims[i]..., n_env*(i - i + 1)...>{};
  })(seq_n_cls));

  static constexpr auto to_cls = ([]() {
    std::array<index_type, sys::n> to_cls;
    for (index_type i = 0; i < sys::n; ++i) {
      to_cls[i] = i / env::n;
    }
    return to_cls;
  })();

  static constexpr auto to_env = ([]() {
    std::array<index_type, sys::n> to_env;
    for (index_type i = 0; i < sys::n; ++i) {
      to_env[i] = i % env::n;
    }
    return to_env;
  })();

  static constexpr auto to_sys_f = make_iota<0, sys::n>();
  static inline const auto to_sys =
      Eigen::TensorMap<const Eigen::TensorFixedSize<
          index_type, Eigen::Sizes<cls::n, env::n>, storage_order>>(
          to_sys_f.data(), cls::n, env::n);

  static constexpr std::optional<index_type> next_to(
      const typename sys::row_type& s1, const typename sys::row_type& s2,
      index_type a) {
    const auto diff = s2.array() - s1.array();
    if (diff.count() != 1) {
      return std::nullopt;
    }
    for (index_type i = 0; i < sys::d; ++i) {
      if (diff(i) &&
          ((i >= cls::d) || ((diff(i) == -1 && a == i) || diff(i) == 1))) {
        return i;
      }
    }
    return std::nullopt;
  }
};

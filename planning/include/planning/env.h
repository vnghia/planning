#pragma once

#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "Fastor/Fastor.h"

using int_type = int;

template <typename T>
concept is_indexable = requires(T t) {
  t[0];
};

template <typename T, int_type... i>
static constexpr T gen_iota(std::integer_sequence<int_type, i...>) {
  return {i...};
}

template <typename T, int_type... i>
static constexpr T gen_trans(std::integer_sequence<int_type, i...>) {
  return {i - i + 1 ..., i - i - 1 ..., 0};
}

template <typename T, typename... prefixes_type, int_type... i,
          typename... postfixes_type>
static constexpr auto to_index(const T& a, prefixes_type... prefixes,
                               std::integer_sequence<int_type, i...>,
                               postfixes_type... postfixes) {
  if constexpr (is_indexable<T>) {
    return std::make_tuple(prefixes..., a[i]..., postfixes...);
  } else {
    return std::make_tuple(prefixes..., a(i)..., postfixes...);
  }
}

template <int_type n_queue_t, typename F, bool save_qs_t, int_type... max_lens>
class Env {
 public:
  using float_type = F;

  static constexpr auto n_queue = n_queue_t;
  static constexpr auto save_qs = save_qs_t;
  static constexpr auto n_transition = 2 * n_queue + 1;
  static constexpr auto n_total = ((max_lens + 1) * ...) * n_queue;

  using env_param_type = Fastor::Tensor<float_type, n_queue, 3>;
  using prob_type = Fastor::Tensor<float_type, n_transition>;

  using q_type = Fastor::Tensor<float_type, max_lens + 1 ..., n_queue>;
  using n_visit_type = Fastor::Tensor<int_type, max_lens + 1 ..., n_queue>;
  using qs_type = std::vector<float_type>;
  using reward_mat_type = Fastor::Tensor<float_type, max_lens + 1 ...>;

  using array_fq_type = Fastor::Tensor<float_type, n_queue>;
  using array_iq_type = Fastor::Tensor<int_type, n_queue>;
  using array_bq_type = Fastor::Tensor<bool, n_queue>;

  using iq_type = Fastor::TensorMap<int_type, n_queue>;

  static constexpr std::array<int_type, n_queue> dim_queue = {max_lens + 1 ...};
  static constexpr auto idx_nq =
      std::make_integer_sequence<int_type, n_queue>{};
  static constexpr auto iota_nq =
      gen_iota<std::array<int_type, n_queue>>(idx_nq);

  Env(const env_param_type& env_param, const reward_mat_type& reward_mat)
      : env_param_(env_param), reward_mat_(reward_mat) {}

  void reset_train() {
    q_.zeros();

    std::apply(
        [this](auto&&... idx) {
          ((std::apply(this->q_, idx) =
                -std::numeric_limits<float_type>::infinity()),
           ...);
        },
        gen_inf_indices(idx_nq));
    std::apply(
        q_, to_index(std::array<decltype(Fastor::fix<0>), n_queue + 1>{},
                     std::make_integer_sequence<int_type, n_queue + 1>{})) = 0;

    n_visit_.zeros();
  }

  void reset(std::mt19937_64::result_type seed = 42) {
    rng.seed(seed);
    states_.zeros();
  }

  prob_type prob(int_type action) {
    array_bq_type allow_u = (states_ < max_lens_);
    array_fq_type pus = env_param_(Fastor::all, Fastor::fix<1>);
    pus *= allow_u.template cast<float_type>();

    array_bq_type allow_d = (action == actions_) && (states_ > 0);
    array_fq_type pds = env_param_(Fastor::all, Fastor::fix<2>);
    pds *= allow_d.template cast<float_type>();

    prob_type p;
    p(Fastor::fseq<0, n_queue>{}) = pus;
    p(Fastor::fseq<n_queue, 2 * n_queue>{}) = pds;
    p(Fastor::fix<2 * n_queue>) = 1 - sum(pus) - sum(pds);
    return p;
  }

  array_iq_type step(int_type action) {
    auto p = prob(action);
    auto idx = std::discrete_distribution<int_type>(
        p.data(), p.data() + n_transition)(rng);
    array_iq_type nstates = states_;
    nstates(idx % n_queue) += transitions_[idx];
    return nstates;
  }

  void train(float_type gamma = 0.9, float_type eps = 0.01,
             float_type decay = 0.5, int_type epoch = 1, uint64_t ls = 20000000,
             float_type lr_pow = 0.51, std::mt19937_64::result_type seed = 42) {
    reset_train();

    if constexpr (save_qs) {
      qs_.resize(epoch * ls * n_total);
    }

    for (int_type i = 0; i < epoch; ++i) {
      reset(seed);

      for (int_type j = 0; j < ls; ++j) {
        int_type a{};
        array_bq_type as = states_ != 0;
        if (Fastor::any_of(as)) {
          std::vector<int_type> idx_as;
          for (int_type k = 0; k < n_queue; ++k) {
            if (as(k)) {
              idx_as.push_back(k);
            }
          }

          if (eps_dis(rng) < eps) {
            a = idx_as[std::uniform_int_distribution<int_type>(
                0, idx_as.size() - 1)(rng)];
          } else [[likely]] {
            auto max_it = std::max_element(
                iota_nq.begin(), iota_nq.end(),
                [this](int_type largest, int_type current) {
                  return std::apply(this->q_,
                                    to_index(this->states_, idx_nq, largest)) <
                         std::apply(this->q_,
                                    to_index(this->states_, idx_nq, current));
                });
            a = *max_it;
          }
        }

        auto nstates = step(a);

        auto max_nit = std::max_element(
            iota_nq.begin(), iota_nq.end(),
            [this, &nstates](int_type largest, int_type current) {
              return std::apply(this->q_, to_index(nstates, idx_nq, largest)) <
                     std::apply(this->q_, to_index(nstates, idx_nq, current));
            });
        float_type nq = std::apply(q_, to_index(nstates, idx_nq, *max_nit));

        const auto st_index = to_index(states_, idx_nq, a);

        auto reward = std::apply(reward_mat_, to_index(nstates, idx_nq));
        const float_type lr =
            std::pow(std::apply(n_visit_, st_index) + 1, -lr_pow);
        std::apply(q_, st_index) +=
            lr * (reward + gamma * nq - std::apply(q_, st_index));

        if constexpr (save_qs) {
          std::copy(q_.data(), q_.data() + n_total,
                    qs_.begin() + i * ls + j * n_total);
        }

        std::apply(n_visit_, st_index) += 1;
        states_ = nstates;
      }
    }
  }

  void from_array(const float_type* q, const int_type* n_visit, float_type* qs,
                  size_t qs_size) {
    q_ = q;
    n_visit_ = n_visit;
    if constexpr (save_qs) {
      std::copy(qs, qs + qs_size, qs_.begin());
    }
  }

  const q_type& q() const { return q_; }
  const n_visit_type& n_visit() const { return n_visit_; }
  const qs_type& qs() const { return qs_; }
  const reward_mat_type& reward_mat() const { return reward_mat_; }

 private:
  const env_param_type env_param_;

  array_iq_type states_;

  q_type q_;
  n_visit_type n_visit_;
  qs_type qs_;

  const reward_mat_type reward_mat_;

  static constexpr auto actions_ =
      iq_type(const_cast<int_type*>(iota_nq.data()));
  static constexpr array_iq_type max_lens_ = {max_lens...};
  static constexpr auto transitions_ =
      gen_trans<std::array<int_type, n_transition>>(idx_nq);

  std::mt19937_64 rng;
  std::uniform_real_distribution<float_type> eps_dis;

  template <int_type current, int_type... before, int_type... after>
  static constexpr auto gen_inf_index(
      std::integer_sequence<int_type, before...>,
      std::integer_sequence<int_type, after...>) {
    return std::make_tuple(Fastor::fseq<0, dim_queue[before]>{}...,
                           Fastor::fseq<0, 1>{},
                           Fastor::fseq<0, dim_queue[after + current + 1]>{}...,
                           Fastor::fix<current>);
  }

  template <int_type... i>
  static constexpr auto gen_inf_indices(
      std::integer_sequence<int_type, i...> is) {
    return std::make_tuple(gen_inf_index<i>(
        std::make_integer_sequence<int_type, i>{},
        std::make_integer_sequence<int_type, is.size() - i - 1>{})...);
  }
};

template <typename F, bool save_qs_t, int_type... max_lens>
class LinearEnv : public Env<2, F, save_qs_t, max_lens...> {
 public:
  using parent_type = Env<2, F, save_qs_t, max_lens...>;
  using parent_type::dim_queue;
  using float_type = typename parent_type::float_type;

  LinearEnv(const typename parent_type::env_param_type& env_param)
      : parent_type(env_param, init_reward(env_param)) {}

  typename parent_type::reward_mat_type init_reward(
      const typename parent_type::env_param_type& env_param) const {
    typename parent_type::reward_mat_type res;
    float_type c1 = env_param(0, 0);
    float_type c2 = env_param(1, 0);
    for (int_type i = 0; i < dim_queue[0]; ++i) {
      for (int_type j = 0; j < dim_queue[1]; ++j) {
        res(i, j) = -(c1 * i + c2 * j);
      }
    }
    return res;
  }
};

template <typename F, bool save_qs_t, int_type... max_lens>
class ConvexEnv : public Env<2, F, save_qs_t, max_lens...> {
 public:
  using parent_type = Env<2, F, save_qs_t, max_lens...>;
  using parent_type::dim_queue;
  using float_type = typename parent_type::float_type;

  ConvexEnv(const typename parent_type::env_param_type& env_param)
      : parent_type(env_param, init_reward(env_param)) {}

  typename parent_type::reward_mat_type init_reward(
      const typename parent_type::env_param_type& env_param) const {
    typename parent_type::reward_mat_type res;
    float_type c1 = env_param(0, 0);
    float_type c2 = env_param(1, 0);
    for (int_type i = 0; i < dim_queue[0]; ++i) {
      for (int_type j = 0; j < dim_queue[1]; ++j) {
        res(i, j) = -(c1 * i + c2 * j * j);
      }
    }
    return res;
  }
};

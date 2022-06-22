#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <random>
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

template <int_type n_transition, int_type n_queue, int_type... i>
static constexpr auto gen_trans(std::integer_sequence<int_type, i...> is) {
  std::array<int_type, n_transition> trans{};
  std::fill_n(trans.begin(), n_queue, 1);
  std::fill_n(trans.begin() + n_queue, n_queue, -1);
  (std::fill_n(trans.begin() + 2 * n_queue + i * n_queue, n_queue, i), ...);
  return trans;
}

template <const auto& a, int_type i, int_type... j, int_type... k>
static constexpr auto gen_inf_index(std::integer_sequence<int_type, j...>,
                                    std::integer_sequence<int_type, k...>) {
  return std::make_tuple(Fastor::fseq<0, a[j]>{}..., Fastor::fix<0>,
                         Fastor::fseq<0, a[k + i + 1]>{}..., Fastor::fix<i>);
}

template <const auto& a, int_type... i>
static constexpr auto gen_inf_indices(std::integer_sequence<int_type, i...> is) {
  return std::make_tuple(
      gen_inf_index<a, i>(std::make_integer_sequence<int_type, i>{},
                          std::make_integer_sequence<int_type, is.size() - i - 1>{})...);
}

template <int_type offset, int_type idx, int_type... i>
static constexpr auto state_prob_assign(auto& p, const auto& e, const auto& s, const auto& a,
                                        std::integer_sequence<int_type, i...>) {
  ((p(offset + i) = a(i) * e(s(i), i, idx)), ...);
}

template <int_type offset, int_type idx, int_type... i>
static constexpr auto env_prob_assign(auto& p, const auto& e, const auto& a,
                                      std::integer_sequence<int_type, i...>) {
  ((p(offset + i) = a(i) * e(idx, i)), ...);
}

template <typename... prefixes_type, int_type... i, typename... postfixes_type>
static constexpr auto&& access(auto& q, const auto& a, prefixes_type... prefixes,
                               std::integer_sequence<int_type, i...>, postfixes_type... postfixes) {
  if constexpr (is_indexable<decltype(a)>) {
    return q(prefixes..., a[i]..., postfixes...);
  } else {
    return q(prefixes..., a(i)..., postfixes...);
  }
}

template <int_type n_env_t, int_type n_queue_t, typename F, bool save_qs_t, int_type... max_lens_t>
class Env {
 public:
  static constexpr auto n_env = n_env_t;
  static constexpr auto n_queue = n_queue_t;

  using float_type = F;

  static constexpr auto save_qs = save_qs_t;

  static constexpr auto dim_queue = std::array{max_lens_t + 1 ...};

  static constexpr auto n_transition = 2 * n_queue + n_queue * n_env;

  using env_param_type = Fastor::Tensor<float_type, n_env, n_queue, 3>;
  using env_prob_type = Fastor::Tensor<float_type, n_env, n_queue>;

  using q_type = Fastor::Tensor<float_type, max_lens_t + 1 ..., n_queue>;
  using n_visit_type = Fastor::Tensor<int_type, max_lens_t + 1 ..., n_queue>;

  using qs_type = std::vector<float_type>;
  static constexpr auto n_total = q_type::size();

  using array_fq_type = Fastor::Tensor<float_type, n_queue>;
  using array_iq_type = Fastor::Tensor<int_type, n_queue>;
  using array_bq_type = Fastor::Tensor<bool, n_queue>;

  using iq_type = Fastor::TensorMap<int_type, n_queue>;

  static constexpr auto idx_nq = std::make_integer_sequence<int_type, n_queue>{};
  static constexpr auto idx_ne = std::make_integer_sequence<int_type, n_env>{};
  static constexpr auto iota_nq = gen_iota<std::array<int_type, n_queue>>(idx_nq);

  static constexpr auto inf_indices = gen_inf_indices<dim_queue>(idx_nq);

  Env(const env_param_type& env_param, const env_prob_type& env_prob)
      : env_param_(env_param), env_prob_(env_prob) {}

  void reset_train() {
    q_.zeros();

    std::apply(
        [this](auto&&... idx) {
          ((std::apply(this->q_, idx) = -std::numeric_limits<float_type>::infinity()), ...);
        },
        inf_indices);
    access(q_, std::array<int_type, n_queue + 1>{},
           std::make_integer_sequence<int_type, n_queue + 1>{}) = 0;

    n_visit_.zeros();
  }

  void reset(std::mt19937_64::result_type seed = 42) {
    rng_.seed(seed);
    states_.zeros();
    env_states_.zeros();
  }

  auto prob(int_type action) {
    array_bq_type allow_u = (states_ < max_lens_);
    array_bq_type allow_d = (action == actions_) && (states_ > 0);

    Fastor::Tensor<float_type, n_transition + 1> p;
    state_prob_assign<0, 1>(p, env_param_, env_states_, allow_u, idx_nq);
    state_prob_assign<n_queue, 2>(p, env_param_, env_states_, allow_d, idx_nq);

    array_bq_type allow_e;
    ([ this, &allow_e, &p ]<int_type... i>(std::integer_sequence<int_type, i...>) {
      ((allow_e = (this->env_states_ != i),
        env_prob_assign<2 * n_queue + i * n_queue, i>(p, env_prob_, allow_e, idx_nq)),
       ...);
    })(idx_ne);

    p(-1) = 1 - sum(p(Fastor::fseq<0, n_transition>{}));

    return p;
  }

  array_iq_type step(int_type action) {
    auto p = prob(action);
    auto idx = std::discrete_distribution<int_type>(p.data(), p.data() + n_transition + 1)(rng_);
    array_iq_type nstates = states_;

    const auto idx_q = idx % n_queue;
    if (idx < 2 * n_queue) {
      nstates(idx_q) += transitions_[idx];
    } else if (idx < n_transition) {
      env_states_(idx_q) = transitions_[idx];
    }
    return nstates;
  }

  virtual float_type reward(const array_iq_type& states, const array_iq_type& env_states) const {
    return 0;
  }

  void train(float_type gamma = 0.9, float_type eps = 0.01, float_type decay = 0.5,
             int_type epoch = 1, uint64_t ls = 20000000, float_type lr_pow = 0.51,
             std::mt19937_64::result_type seed = 42) {
    reset_train();

    if constexpr (save_qs) {
      qs_.resize(epoch * ls * n_total);
    }

    for (int_type i = 0; i < epoch; ++i) {
      reset(seed);

      for (int_type j = 0; j < ls; ++j) {
        int_type a{};
        if (eps_dis_(rng_) < eps) {
          array_iq_type as = states_ != 0;
          a = std::discrete_distribution<int_type>(as.data(), as.data() + n_queue)(rng_);
        } else {
          auto max_it = std::max_element(
              iota_nq.begin(), iota_nq.end(), [this](int_type largest, int_type current) {
                return access(q_, states_, idx_nq, largest) < access(q_, states_, idx_nq, current);
              });
          a = *max_it;
        }

        auto nstates = step(a);

        auto max_nit = std::max_element(
            iota_nq.begin(), iota_nq.end(), [this, &nstates](int_type largest, int_type current) {
              return access(q_, nstates, idx_nq, largest) < access(q_, nstates, idx_nq, current);
            });

        const float_type nq = access(q_, nstates, idx_nq, *max_nit);
        const auto r = reward(nstates, env_states_);
        const float_type lr = std::pow(access(n_visit_, states_, idx_nq, a) + 1, -lr_pow);

        access(q_, states_, idx_nq, a) += lr * (r + gamma * nq - access(q_, states_, idx_nq, a));

        if constexpr (save_qs) {
          std::copy(q_.data(), q_.data() + n_total, qs_.begin() + i * ls + j * n_total);
        }

        access(n_visit_, states_, idx_nq, a) += 1;
        states_ = nstates;
      }
    }
  }

  void from_array(const float_type* q, const int_type* n_visit, float_type* qs, size_t qs_size) {
    q_ = q;
    n_visit_ = n_visit;
    if constexpr (save_qs) {
      std::copy(qs, qs + qs_size, qs_.begin());
    }
  }

  const q_type& q() const { return q_; }
  const n_visit_type& n_visit() const { return n_visit_; }
  const qs_type& qs() const { return qs_; }

 protected:
  const env_param_type env_param_;

 private:
  const env_prob_type env_prob_;

  array_iq_type states_;
  array_iq_type env_states_;

  q_type q_;
  n_visit_type n_visit_;
  qs_type qs_;

  static constexpr auto actions_ = iq_type(const_cast<int_type*>(iota_nq.data()));
  static constexpr array_iq_type max_lens_ = {max_lens_t...};
  static constexpr auto transitions_ = gen_trans<n_transition, n_queue>(idx_ne);

  std::mt19937_64 rng_;
  std::uniform_real_distribution<float_type> eps_dis_;
};

template <int_type n_env_t, typename F, bool save_qs_t, int_type... max_lens_t>
class LinearEnv : public Env<n_env_t, 2, F, save_qs_t, max_lens_t...> {
 public:
  using parent_type = Env<n_env_t, 2, F, save_qs_t, max_lens_t...>;

 protected:
  using parent_type::env_param_;

 public:
  using parent_type::parent_type;

  using float_type = typename parent_type::float_type;
  using array_iq_type = typename parent_type::array_iq_type;

  float_type reward(const array_iq_type& states, const array_iq_type& env_states) const override {
    return -(states(0) * env_param_(env_states(0), 0, 0) +
             states(1) * env_param_(env_states(1), 1, 0));
  }
};

template <int_type n_env_t, typename F, bool save_qs_t, int_type... max_lens_t>
class ConvexEnv : public Env<n_env_t, 2, F, save_qs_t, max_lens_t...> {
 public:
  using parent_type = Env<n_env_t, 2, F, save_qs_t, max_lens_t...>;

 protected:
  using parent_type::env_param_;

 public:
  using parent_type::parent_type;

  using float_type = typename parent_type::float_type;
  using array_iq_type = typename parent_type::array_iq_type;

  float_type reward(const array_iq_type& states, const array_iq_type& env_states) const override {
    return -(states(0) * env_param_(env_states(0), 0, 0) +
             states(1) * states(1) * env_param_(env_states(1), 1, 0));
  }
};

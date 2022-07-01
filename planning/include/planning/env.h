#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "Fastor/Fastor.h"
#include "pcg_random.hpp"

using int_type = int;

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

template <typename... prefixes_type, int_type... i, typename... postfixes_type>
static inline constexpr auto&& access(auto& q, const auto& a, prefixes_type... prefixes,
                                      std::integer_sequence<int_type, i...>,
                                      postfixes_type... postfixes) {
  return q(prefixes..., a[i]..., postfixes...);
}

template <int_type offset, int_type... i>
static constexpr auto prod(const auto& a, std::integer_sequence<int_type, i...>) {
  return (a[offset + i] * ... * 1);
}

template <int_type... dims_t>
static constexpr auto gen_state_indices() {
  constexpr auto base = ([]() {
    auto dims = std::array{dims_t...};
    constexpr auto ndim = dims.size();

    constexpr auto idx_d = std::make_integer_sequence<int_type, ndim - 1>{};

    auto res = ([&dims]<int_type... i>(std::integer_sequence<int_type, i...>) {
      return std::array<int_type, ndim>{
          prod<i + 1>(dims, std::make_integer_sequence<int_type, ndim - 1 - i>{})...};
    })(idx_d);
    res[ndim - 1] = 1;

    return res;
  })();

  constexpr auto ncomb = (dims_t * ... * 1);
  constexpr auto ndim = base.size();
  std::array<std::array<int_type, ndim>, ncomb> res;

  for (int_type i = 0; i < ncomb; ++i) {
    auto current = i;
    for (int_type j = 0; j < ndim; ++j) {
      res[i][j] = current / base[j];
      current -= res[i][j] * base[j];
    }
  }

  return std::make_pair(res, base);
}

template <int_type n_dim_t, int_type n_queue_t>
static constexpr std::pair<int_type, int_type> state_next_to(
    const std::array<int_type, n_dim_t>& idx1, const std::array<int_type, n_dim_t>& idx2,
    int_type action) {
  std::pair<int_type, int_type> res{-1, 0};

  for (int_type i = 0; i < n_dim_t; ++i) {
    if (idx1[i] != idx2[i]) {
      if (res.first != -1) {
        return {-1, 0};
      } else if (i < n_queue_t) {
        res.first = i;
        res.second = idx2[i];
      } else if (idx1[i] - 1 == idx2[i] && i - n_queue_t == action) {
        res.first = i;
        res.second = -1;
      } else if (idx1[i] + 1 == idx2[i]) {
        res.first = i;
        res.second = 1;
      } else {
        return {-1, 0};
      }
    }
  }
  return res;
}

template <typename T, int_type... i>
static inline constexpr auto split_full_states(const auto& s,
                                               std::integer_sequence<int_type, i...> is) {
  return std::make_pair(T{s[i]...}, T{s[is.size() + i]...});
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

  using env_cost_type = Fastor::Tensor<float_type, n_env, n_queue>;
  using env_param_type = Fastor::Tensor<float_type, n_env, n_queue, 2>;
  using env_prob_type = env_cost_type;

  using q_type = Fastor::Tensor<float_type, max_lens_t + 1 ..., n_queue>;
  using n_visit_type = Fastor::Tensor<int_type, max_lens_t + 1 ..., n_queue>;

  using qs_type = std::vector<float_type>;
  static constexpr auto n_total = q_type::size();

  using states_type = std::array<int_type, n_queue>;

  static constexpr auto idx_nq = std::make_integer_sequence<int_type, n_queue>{};
  static constexpr auto idx_ne = std::make_integer_sequence<int_type, n_env>{};
  static constexpr auto iota_nq = gen_iota<std::array<int_type, n_queue>>(idx_nq);

  static constexpr auto full_state_information =
      ([]<int_type... i>(std::integer_sequence<int_type, i...>) {
        return gen_state_indices<n_env*(i - i + 1)..., max_lens_t + 1 ...>();
      })(idx_nq);
  static constexpr auto full_state_indicies = full_state_information.first;
  static constexpr auto full_state_base = full_state_information.second;
  static constexpr auto n_full_dim = full_state_indicies[0].size();

  static constexpr auto n_combination = full_state_indicies.size();
  static constexpr auto exceed_size = n_combination > 1000;

  using reward_vec_type =
      std::conditional_t<!exceed_size, Fastor::Tensor<float_type, n_combination>, std::nullptr_t>;
  using prob_mat_type =
      std::conditional_t<!exceed_size,
                         Fastor::Tensor<float_type, n_combination, n_combination, n_queue>,
                         std::nullptr_t>;
  using v_type =
      std::conditional_t<!exceed_size, Fastor::Tensor<float_type, n_queue>, std::nullptr_t>;
  using policy_v_type =
      std::conditional_t<!exceed_size, Fastor::Tensor<int_type, n_combination>, std::nullptr_t>;

  static constexpr auto inf_indices = gen_inf_indices<dim_queue>(idx_nq);

  Env(const env_cost_type& env_cost, const env_param_type& env_param, const env_prob_type& env_prob)
      : env_cost_(env_cost), env_param_(env_param), env_prob_(env_prob) {}

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

  void reset(pcg32::state_type seed) {
    rng_.seed(seed);
    states_.fill(0);
    env_states_.fill(0);
    not_empty_.fill(0);
    not_full_.fill(1);
  }

  auto prob(int_type action) {
    std::array<float_type, n_transition + 1> p{};
    float_type prob_dummy = 1;

    for (int_type i = 0; i < n_queue; ++i) {
      if (not_full_[i]) {
        const auto temp = env_param_(env_states_[i], i, 0);
        prob_dummy -= temp;
        p[i] = temp;
      }
    }

    if (not_empty_[action]) {
      const auto temp = env_param_(env_states_[action], action, 1);
      prob_dummy -= temp;
      p[n_queue + action] = temp;
    }

    for (int_type i = 0; i < n_queue; ++i) {
      auto env_state = env_states_[i];
      for (int_type j = 0; j < n_env; ++j) {
        if (env_state != j) {
          const auto temp = env_prob_(env_state, i);
          prob_dummy -= temp;
          p[2 * n_queue + j * n_queue + i] = temp;
        }
      }
    }

    p[n_transition] = prob_dummy;

    return p;
  }

  void step(int_type action) {
    auto p = prob(action);
    auto idx = std::discrete_distribution<int_type>(p.data(), p.data() + p.size())(rng_);

    const auto idx_q = idx % n_queue;
    if (idx < 2 * n_queue) {
      auto state = states_[idx_q] + transitions_[idx];
      states_[idx_q] = state;
      not_empty_[idx_q] = state > 0;
      not_full_[idx_q] = state < max_lens_[idx_q];
    } else if (idx < n_transition) {
      env_states_[idx_q] = transitions_[idx];
    }
  }

  virtual float_type reward(const states_type& env_states, const states_type& states) const {
    return 0;
  }

  void init_reward_vec() {
    if constexpr (exceed_size) {
      return;
    } else {
      for (int_type i = 0; i < n_combination; ++i) {
        auto [env_states, states] = split_full_states<states_type>(full_state_indicies[i], idx_nq);
        reward_vec_[i] = reward(env_states, states);
      }
    }
  }

  void init_prob_mat() {
    if constexpr (exceed_size) {
      return;
    } else {
      for (int_type action = 0; action < n_queue; ++action) {
        for (int_type i = 0; i < n_combination; ++i) {
          const auto& full_states_idx_i = full_state_indicies[i];
          float_type dummy_prob = 1;

          for (int_type j = 0; j < n_combination; ++j) {
            if (i == j) continue;

            auto next_to = state_next_to<n_full_dim, n_queue>(full_states_idx_i,
                                                              full_state_indicies[j], action);
            if (next_to.first > 0) {
              if (next_to.first < n_queue) {
                prob_mat_(i, j, action) = env_prob_(next_to.second, next_to.first);
              } else {
                const auto idx_q = next_to.first - n_queue;
                prob_mat_(i, j, action) =
                    env_param_(full_states_idx_i[idx_q], idx_q, next_to.second == 1 ? 0 : 1);
              }
              dummy_prob -= prob_mat_(i, j, action);
            }
          }

          prob_mat_(i, i, action) = dummy_prob;
        }
      }
    }
  }

  void train(float_type gamma, float_type eps, float_type decay, int_type epoch, uint64_t ls,
             float_type lr_pow, pcg32::state_type seed) {
    reset_train();

    if constexpr (save_qs) {
      qs_.resize(epoch * ls * n_total);
    }

    for (int_type i = 0; i < epoch; ++i) {
      reset(seed);

      for (int_type j = 0; j < ls; ++j) {
        const auto states = states_;

        int_type a{};
        if (eps_dis_(rng_) < eps) {
          a = std::discrete_distribution<int_type>(not_empty_.data(),
                                                   not_empty_.data() + not_empty_.size())(rng_);
        } else {
          auto max_it = std::max_element(
              iota_nq.begin(), iota_nq.end(), [this, &states](int_type largest, int_type current) {
                return access(q_, states, idx_nq, largest) < access(q_, states, idx_nq, current);
              });
          a = *max_it;
        }

        step(a);

        const auto& nstates = states_;

        auto max_nit = std::max_element(
            iota_nq.begin(), iota_nq.end(), [this, &nstates](int_type largest, int_type current) {
              return access(q_, nstates, idx_nq, largest) < access(q_, nstates, idx_nq, current);
            });

        const float_type nq = access(q_, nstates, idx_nq, *max_nit);
        const auto r = reward(env_states_, nstates);
        const float_type lr = std::pow(access(n_visit_, states, idx_nq, a) + 1, -lr_pow);

        access(q_, states, idx_nq, a) += lr * (r + gamma * nq - access(q_, states, idx_nq, a));

        if constexpr (save_qs) {
          std::copy(q_.data(), q_.data() + n_total, qs_.begin() + i * ls + j * n_total);
        }

        access(n_visit_, states, idx_nq, a) += 1;
      }
    }
  }

  void train_v(uint64_t ls, float_type lr) {
    if constexpr (exceed_size) {
      return;
    } else {
      v_.fill(0);
      for (uint64_t i = 0; i < ls; ++i) {
        for (int_type s = 0; s < n_combination; ++s) {
          auto max_a = -std::numeric_limits<float_type>::infinity();
          for (int_type a = 0; a < n_queue; ++a) {
            float_type v_s_a =
                Fastor::sum(prob_mat_(s, Fastor::all, a) * (reward_vec_(s) + lr * v_));
            max_a = std::max(max_a, v_s_a);
          }
          v_[s] = max_a;
        }
      }
      for (int_type s = 0; s < n_combination; ++s) {
        auto max_a = -std::numeric_limits<float_type>::infinity();
        int_type max_a_idx;
        for (int_type a = 0; a < n_queue; ++a) {
          float_type v_s_a = Fastor::sum(prob_mat_(s, Fastor::all, a) * (reward_vec_(s) + lr * v_));
          if (max_a < v_s_a) {
            max_a = v_s_a;
            max_a_idx = a;
          }
        }
        policy_v_[s] = max_a_idx;
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
  const reward_vec_type& reward_vec() const { return reward_vec_; }
  const prob_mat_type& prob_mat() const { return prob_mat_; }
  const v_type& v() const { return v_; }
  const policy_v_type& policy_v() const { return policy_v_; }

 protected:
  const env_cost_type env_cost_;

 private:
  const env_param_type env_param_;
  const env_prob_type env_prob_;

  states_type states_;
  states_type env_states_;
  states_type not_empty_;
  states_type not_full_;

  q_type q_;
  n_visit_type n_visit_;
  qs_type qs_;
  reward_vec_type reward_vec_;
  prob_mat_type prob_mat_;
  v_type v_;
  policy_v_type policy_v_;

  static constexpr states_type max_lens_ = {max_lens_t...};
  static constexpr auto transitions_ = gen_trans<n_transition, n_queue>(idx_ne);

  pcg32 rng_;
  std::uniform_real_distribution<float_type> eps_dis_;
};

template <int_type n_env_t, typename F, bool save_qs_t, int_type... max_lens_t>
class LinearEnv : public Env<n_env_t, 2, F, save_qs_t, max_lens_t...> {
 public:
  using parent_type = Env<n_env_t, 2, F, save_qs_t, max_lens_t...>;

 protected:
  using parent_type::env_cost_;

 public:
  using parent_type::parent_type;

  using float_type = typename parent_type::float_type;
  using states_type = typename parent_type::states_type;

  float_type reward(const states_type& env_states, const states_type& states) const override {
    return -(states[0] * env_cost_(env_states[0], 0) + states[1] * env_cost_(env_states[1], 1));
  }
};

template <int_type n_env_t, typename F, bool save_qs_t, int_type... max_lens_t>
class ConvexEnv : public Env<n_env_t, 2, F, save_qs_t, max_lens_t...> {
 public:
  using parent_type = Env<n_env_t, 2, F, save_qs_t, max_lens_t...>;
  static constexpr auto is_convex = true;

 protected:
  using parent_type::env_cost_;

 public:
  using env_cost_type = typename parent_type::env_cost_type;
  using env_param_type = typename parent_type::env_param_type;
  using env_prob_type = typename parent_type::env_prob_type;

  using float_type = typename parent_type::float_type;
  using states_type = typename parent_type::states_type;

  ConvexEnv(const env_cost_type& env_cost, const env_param_type& env_param,
            const env_prob_type& env_prob, float_type cost_eps)
      : parent_type(env_cost, env_param, env_prob), cost_eps_(cost_eps) {}

  float_type reward(const states_type& env_states, const states_type& states) const override {
    return -(states[0] * env_cost_(env_states[0], 0) +
             (states[1] * states[1] * cost_eps_ + states[1]) * env_cost_(env_states[1], 1));
  }

 private:
  const float_type cost_eps_;
};

template <typename T>
concept env_is_convex = T::is_convex;

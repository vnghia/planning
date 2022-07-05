#pragma once

#include <array>
#include <cmath>
#include <execution>
#include <functional>
#include <limits>
#include <random>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "Fastor/Fastor.h"
#include "pcg_random.hpp"

template <size_t begin, size_t end>
static constexpr auto make_iota() {
  std::array<size_t, end - begin> res;
  std::iota(res.begin(), res.end(), begin);
  return res;
}

template <size_t... set_lens_t>
static constexpr auto make_set_product() {
  constexpr auto n_combination = (set_lens_t * ... * 1);
  constexpr auto n_set = sizeof...(set_lens_t);
  constexpr auto set_lens = std::array{set_lens_t...};

  std::array<std::array<size_t, n_set>, n_combination> product;

  for (size_t i = 0; i < n_combination; ++i) {
    auto current = i;

    for (size_t j = 0; j < n_set; ++j) {
      product[i][n_set - 1 - j] = current % set_lens[n_set - 1 - j];
      current /= set_lens[n_set - 1 - j];
    }
  }

  return product;
}

template <const auto idx, size_t i, size_t dim_i>
static constexpr auto make_inf_idx_i() {
  std::array<size_t, idx.size() / dim_i> res;
  size_t cur{};

  for (size_t j = 0; j < idx.size(); ++j) {
    if (idx[j][i] == 0) {
      res[cur++] = j;
    }
  }

  return res;
}

template <size_t dim_full_state, size_t n_queue>
static constexpr auto can_transition_to(const auto& s1, const auto& s2,
                                        size_t action) {
  std::pair<int, int> res{-1, 0};

  for (size_t i = 0; i < dim_full_state; ++i) {
    const auto diff = s2[i] - s1[i];
    if (diff) {
      if (res.first >= 0) {
        return std::make_pair(-1, 0);
      } else if (i >= n_queue) {
        res.first = i;
        res.second = s2[i];
      } else if ((diff == -1 && i == action) || (diff == 1)) {
        res.first = i;
        res.second = diff;
      } else {
        return std::make_pair(-1, 0);
      }
    }
  }

  return res;
}

template <size_t n_queue_t, size_t n_env_t, typename float_t, bool save_qs_t,
          size_t... queue_lens_t>
class Env {
 public:
  static constexpr auto n_queue = n_queue_t;
  static constexpr auto n_env = n_env_t;

  using float_type = float_t;
  static constexpr auto inf_v = std::numeric_limits<float_type>::infinity();

  static constexpr auto save_qs = save_qs_t;
  static constexpr std::array dims_queue = {queue_lens_t + 1 ...};

  using env_cost_type = Fastor::Tensor<float_type, n_queue, n_env>;
  using env_arrival_type = env_cost_type;
  using env_departure_type = env_cost_type;
  using env_prob_type = env_cost_type;

  static constexpr auto idx_nq = std::make_index_sequence<n_queue>{};

  static constexpr auto full_state_idx =
      ([]<size_t... i>(std::index_sequence<i...>) {
        return make_set_product<dims_queue[i]..., n_env*(i - i + 1)...>();
      })(idx_nq);

  static constexpr auto n_full_state = full_state_idx.size();
  using full_state_type = typename decltype(full_state_idx)::value_type;
  static constexpr auto dim_full_state =
      std::tuple_size<full_state_type>::value;

  static constexpr auto obs_state_idx =
      ([]<size_t... i>(std::index_sequence<i...>) {
        return make_set_product<dims_queue[i]...>();
      })(idx_nq);

  static constexpr auto n_obs_state = obs_state_idx.size();
  using obs_state_type = typename decltype(obs_state_idx)::value_type;
  static constexpr auto dim_obs_state = std::tuple_size<obs_state_type>::value;

  static constexpr auto offset_full_obs = dim_full_state - dim_obs_state;

  static constexpr auto ratio_full_obs =
      ((n_env * (queue_lens_t - queue_lens_t + 1)) * ... * 1);

  static constexpr auto map_full_obs = ([]() {
    std::array<size_t, n_full_state> res;
    for (size_t i = 0; i < n_full_state; ++i) {
      res[i] = i / ratio_full_obs;
    }
    return res;
  })();

  using transition_config_type = std::array<
      std::array<std::pair<std::vector<size_t>, std::vector<float_type>>,
                 n_queue>,
      n_full_state>;
  using transition_dist_type =
      std::array<std::array<std::discrete_distribution<size_t>, n_queue>,
                 n_full_state>;
  using action_dist_type =
      std::array<std::discrete_distribution<size_t>, n_obs_state>;
  using reward_config_type = std::array<float_type, n_full_state>;

  using reward_func_type =
      std::function<float_type(const env_cost_type&, const full_state_type&)>;

  using q_type = std::array<float_type, n_obs_state * n_queue>;
  using n_visit_type = std::array<size_t, n_obs_state * n_queue>;
  using qs_type =
      std::conditional_t<save_qs, std::vector<float_type>, std::nullptr_t>;

  using v_type =
      std::conditional_t<n_env == 1, std::array<float_type, n_obs_state>,
                         std::nullptr_t>;
  using policy_v_type =
      std::conditional_t<n_env == 1, std::array<size_t, n_obs_state>,
                         std::nullptr_t>;

  static constexpr auto q_inf_idx = ([]() {
    std::array<size_t, ((obs_state_idx.size() / (queue_lens_t + 1)) + ...)>
        inf_idx{};
    size_t cur{};
    for (size_t i = 0; i < dim_obs_state; ++i) {
      for (size_t j = 0; j < n_obs_state; ++j) {
        if (!obs_state_idx[j][i]) {
          inf_idx[cur++] = j * n_queue + i;
        }
      }
    }
    return inf_idx;
  })();

  Env(const env_cost_type& env_cost, const env_arrival_type& env_arrival,
      const env_departure_type& env_departure, const env_prob_type& env_prob,
      const reward_func_type& reward_func)
      : env_cost_(env_cost),
        env_arrival_(env_arrival),
        env_departure_(env_departure),
        env_prob_(env_prob),
        transition_config_(init_transition_config()),
        transition_dist_(init_transition_dist()),
        action_dist_(init_action_dist()),
        reward_config_(init_reward_config(reward_func)) {}

  void reset_q() {
    q_.fill(0);
    for (size_t i = 0; i < q_inf_idx.size(); ++i) {
      q_[q_inf_idx[i]] = -inf_v;
    }
    q_[0] = 0;

    n_visit_.fill(0);
  }

  void reset_q_epoch(pcg32::state_type seed) {
    rng_.seed(seed);
    state_ = 0;
  }

  void step(size_t action) {
    auto next_state_idx = transition_dist_[state_][action](rng_);
    state_ = transition_config_[state_][action].first[next_state_idx];
  }

  void train_q(float_type gamma, float_type eps, float_type decay, size_t epoch,
               uint64_t ls, pcg32::state_type seed) {
    static constexpr auto iota_n_queue = make_iota<0, n_queue>();
    reset_q();

    if constexpr (save_qs) {
      qs_.resize(epoch * ls * q_.size());
    }

    for (size_t i = 0; i < epoch; ++i) {
      reset_q_epoch(seed);

      for (size_t j = 0; j < ls; ++j) {
        const auto state = state_;
        const auto obs_state = map_full_obs[state];

        size_t a;
        if (greedy_dis_(rng_) < eps) {
          a = action_dist_[obs_state](rng_);
        } else {
          const auto it = q_.begin() + obs_state * n_queue;
          a = std::max_element(it, it + n_queue) - it;
        }

        step(a);
        const auto next_state = state_;
        const auto next_obs_state = map_full_obs[next_state];

        const auto next_it = q_.begin() + next_obs_state * n_queue;
        const auto next_q = *std::max_element(next_it, next_it + n_queue);
        const auto reward = reward_config_[state];

        const auto q_idx = obs_state * n_queue + a;
        q_[q_idx] += (static_cast<float_type>(1) / ++n_visit_[q_idx]) *
                     (reward + gamma * next_q - q_[q_idx]);

        if constexpr (save_qs) {
          std::copy(q_.data(), q_.data() + q_.size(),
                    qs_.begin() + (i * ls + j) * q_.size());
        }
      }
    }
  }

  void train_v(float_type gamma, uint64_t ls) {
    if constexpr (n_env == 1) {
      static constexpr auto iota_n_obs_state = make_iota<0, n_obs_state>();

      v_.fill(0);
      for (uint64_t i = 0; i < ls; ++i) {
        const auto v_i = v_;
        std::for_each(std::execution::par_unseq, iota_n_obs_state.begin(),
                      iota_n_obs_state.end(), [gamma, this, &v_i](size_t j) {
                        update_v_i<false>(gamma, j, v_i);
                      });
      }
      std::for_each(std::execution::par_unseq, iota_n_obs_state.begin(),
                    iota_n_obs_state.end(), [gamma, this](size_t j) {
                      update_v_i<true>(gamma, j, v_);
                    });
    }
  }

  void from_array(const float_type* q, const size_t* n_visit,
                  std::span<float_type> qs) {
    std::copy_n(q, q_.size(), q_.begin());
    std::copy_n(n_visit, n_visit_.size(), n_visit_.begin());
    if constexpr (save_qs) {
      qs_.insert(qs_.begin(), qs.begin(), qs.end());
    }
  }

  const q_type& q() const { return q_; }
  const n_visit_type& n_visit() const { return n_visit_; }
  const qs_type& qs() const { return qs_; }

  const v_type& v() const { return v_; }
  const policy_v_type& policy_v() const { return policy_v_; }

 private:
  const env_cost_type env_cost_;
  const env_arrival_type env_arrival_;
  const env_departure_type env_departure_;
  const env_prob_type env_prob_;

  const transition_config_type transition_config_;
  transition_dist_type transition_dist_;
  action_dist_type action_dist_;
  const reward_config_type reward_config_;

  size_t state_;

  q_type q_;
  n_visit_type n_visit_;
  qs_type qs_;

  v_type v_;
  policy_v_type policy_v_;

  pcg32 rng_;
  std::uniform_real_distribution<float_type> greedy_dis_;

  auto init_transition_config() {
    transition_config_type config;

    for (size_t i = 0; i < n_full_state; ++i) {
      const auto& s_i = full_state_idx[i];

      for (size_t a = 0; a < n_queue; ++a) {
        if ((i >= ratio_full_obs && !s_i[a]) || (i < ratio_full_obs && a != 0))
          continue;

        float_type dummy_prob = 1;
        auto& config_i = config[i][a].second;

        for (size_t j = 0; j < n_full_state; ++j) {
          if (i == j) continue;

          const auto next_to = can_transition_to<dim_full_state, n_queue>(
              s_i, full_state_idx[j], a);

          if (next_to.first >= 0) {
            config[i][a].first.push_back(j);

            const auto idx_q = next_to.first;

            if (idx_q >= n_queue) {
              config_i.push_back(env_prob_(idx_q - n_queue, next_to.second));
            } else {
              if (next_to.second == -1) {
                config_i.push_back(env_departure_(idx_q, s_i[idx_q + n_queue]));
              } else if (next_to.second == 1) {
                config_i.push_back(env_arrival_(idx_q, s_i[idx_q + n_queue]));
              }
            }

            dummy_prob -= config_i.back();
          }
        }
        config[i][a].first.push_back(i);
        config_i.push_back(dummy_prob);
      }
    }

    return config;
  }

  auto init_transition_dist() {
    transition_dist_type dist;

    for (size_t i = 0; i < n_full_state; ++i) {
      for (size_t a = 0; a < n_queue; ++a) {
        const auto& config = transition_config_[i][a].second;
        if (config.size()) {
          dist[i][a] =
              std::discrete_distribution<size_t>(config.begin(), config.end());
        }
      }
    }

    return dist;
  }

  auto init_action_dist() {
    action_dist_type dist;
    for (size_t i = 0; i < n_obs_state; ++i) {
      auto full_state = i * ratio_full_obs;
      std::array<size_t, n_queue> possible_action{};

      for (size_t j = 0; j < n_queue; ++j) {
        if (transition_config_[full_state][j].first.size()) {
          possible_action[j] = 1;
        }
      }

      dist[i] = std::discrete_distribution<size_t>(possible_action.begin(),
                                                   possible_action.end());
    }

    return dist;
  }

  auto init_reward_config(const reward_func_type& reward_func) {
    reward_config_type config;

    for (size_t i = 0; i < n_full_state; ++i) {
      config[i] = reward_func(env_cost_, full_state_idx[i]);
    }

    return config;
  }

  template <bool update_policy>
  void update_v_i(float_type gamma, size_t j, const v_type& v_i) {
    float_type max_v = -inf_v;
    size_t max_a = 0;

    for (size_t a = 0; a < n_queue; ++a) {
      float_type val_v = reward_config_[j];
      const auto& [states, probs] = transition_config_[j][a];

      if (states.empty()) continue;

      for (size_t k = 0; k < states.size(); ++k) {
        auto s = states[k];
        val_v += probs[k] * (gamma * v_i[s]);
      }

      if (max_v < val_v) {
        max_v = val_v;
        if constexpr (update_policy) {
          max_a = a;
        }
      }
    }

    if constexpr (update_policy) {
      policy_v_[j] = max_a;
    } else {
      v_[j] = max_v;
    }
  }
};

template <size_t n_env_t, typename float_t, bool save_qs_t,
          size_t... queue_lens_t>
class LinearEnv : public Env<2, n_env_t, float_t, save_qs_t, queue_lens_t...> {
 public:
  using env_type = Env<2, n_env_t, float_t, save_qs_t, queue_lens_t...>;

  using env_cost_type = typename env_type::env_cost_type;
  using env_arrival_type = typename env_type::env_arrival_type;
  using env_departure_type = typename env_type::env_departure_type;
  using env_prob_type = typename env_type::env_prob_type;
  using full_state_type = typename env_type::full_state_type;

  static constexpr char env_name[] = "linear";

  LinearEnv(const env_cost_type& env_cost, const env_arrival_type& env_arrival,
            const env_departure_type& env_departure,
            const env_prob_type& env_prob)
      : env_type(
            env_cost, env_arrival, env_departure, env_prob,
            [](const env_cost_type& env_cost, const full_state_type& state) {
              return -(
                  env_cost(0, state[env_type::offset_full_obs + 0]) * state[0] +
                  env_cost(1, state[env_type::offset_full_obs + 1]) * state[1]);
            }) {}
};

template <size_t n_env_t, typename float_t, bool save_qs_t,
          size_t... queue_lens_t>
class ConvexEnv : public Env<2, n_env_t, float_t, save_qs_t, queue_lens_t...> {
 public:
  using env_type = Env<2, n_env_t, float_t, save_qs_t, queue_lens_t...>;
  static constexpr auto is_convex = true;

  using env_cost_type = typename env_type::env_cost_type;
  using env_arrival_type = typename env_type::env_arrival_type;
  using env_departure_type = typename env_type::env_departure_type;
  using env_prob_type = typename env_type::env_prob_type;
  using full_state_type = typename env_type::full_state_type;

  static constexpr char env_name[] = "convex";

  using float_type = typename env_type::float_type;

  ConvexEnv(const env_cost_type& env_cost, const env_arrival_type& env_arrival,
            const env_departure_type& env_departure,
            const env_prob_type& env_prob, float_type cost_eps)
      : env_type(env_cost, env_arrival, env_departure, env_prob,
                 [cost_eps](const env_cost_type& env_cost,
                            const full_state_type& state) {
                   return -(env_cost(0, state[env_type::offset_full_obs + 0]) *
                                state[0] +
                            env_cost(1, state[env_type::offset_full_obs + 1]) *
                                (state[1] * state[1] * cost_eps + state[1]));
                 }) {}
};

template <typename T>
concept env_is_convex = T::is_convex;

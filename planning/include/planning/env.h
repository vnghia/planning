#pragma once

#include <array>
#include <cmath>
#include <execution>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "planning/xoshiro.h"

using index_type = int;

template <index_type begin, index_type end>
static constexpr auto make_iota() {
  std::array<index_type, end - begin> res;
  std::iota(res.begin(), res.end(), begin);
  return res;
}

template <index_type... set_lens_t>
static constexpr auto make_set_product() {
  constexpr auto n_combination = (set_lens_t * ... * 1);
  constexpr auto n_set = sizeof...(set_lens_t);
  constexpr auto set_lens = std::array{set_lens_t...};

  std::array<std::array<index_type, n_set>, n_combination> product;

  for (index_type i = 0; i < n_combination; ++i) {
    auto current = i;

    for (index_type j = 0; j < n_set; ++j) {
      product[i][j] = current % set_lens[j];
      current /= set_lens[j];
    }
  }

  return product;
}

using transition_status_type =
    std::optional<std::pair<index_type, std::variant<index_type, bool>>>;

template <index_type dim_full_state, index_type n_queue>
static constexpr transition_status_type can_transition_to(const auto& s1,
                                                          const auto& s2,
                                                          index_type action) {
  transition_status_type res{};

  for (index_type i = 0; i < dim_full_state; ++i) {
    const auto diff = s2[i] - s1[i];
    if (diff) {
      if (res) {
        return std::nullopt;
      } else if (i >= n_queue) {
        res = std::make_pair(i, s2[i]);
      } else if ((diff == -1 && i == action) || (diff == 1)) {
        res = std::make_pair(i, diff == 1);
      } else {
        return std::nullopt;
      }
    }
  }

  return res;
}

template <index_type n_env_t, typename float_t, bool save_qs_t,
          index_type... queue_lens_t>
class Env {
 public:
  static constexpr index_type n_queue = sizeof...(queue_lens_t);
  static constexpr auto n_env = n_env_t;

  using float_type = float_t;
  static constexpr auto inf_v = std::numeric_limits<float_type>::infinity();

  static constexpr auto save_qs = save_qs_t;
  static constexpr std::array dims_queue = {queue_lens_t + 1 ...};

  using env_cost_type = Eigen::Matrix<float_type, n_queue, n_env>;
  using env_arrival_type = env_cost_type;
  using env_departure_type = env_cost_type;
  using env_prob_type = env_cost_type;

  static constexpr auto idx_nq =
      std::make_integer_sequence<index_type, n_queue>{};

  static constexpr auto full_state_idx =
      ([]<index_type... i>(std::integer_sequence<index_type, i...>) {
        return make_set_product<dims_queue[i]..., n_env*(i - i + 1)...>();
      })(idx_nq);

  static constexpr auto n_full_state = full_state_idx.size();
  using full_state_type = typename decltype(full_state_idx)::value_type;
  static constexpr auto dim_full_state =
      std::tuple_size<full_state_type>::value;

  static constexpr auto obs_state_idx =
      ([]<index_type... i>(std::integer_sequence<index_type, i...>) {
        return make_set_product<dims_queue[i]...>();
      })(idx_nq);

  static constexpr auto n_obs_state = obs_state_idx.size();
  using obs_state_type = typename decltype(obs_state_idx)::value_type;
  static constexpr auto dim_obs_state = std::tuple_size<obs_state_type>::value;

  static constexpr auto offset_full_obs = dim_full_state - dim_obs_state;

  static constexpr auto map_full_obs = ([]() {
    std::array<index_type, n_full_state> res;
    for (index_type i = 0; i < n_full_state; ++i) {
      res[i] = i % n_obs_state;
    }
    return res;
  })();

  using transition_config_type =
      std::array<std::array<Eigen::SparseVector<float_type>, n_queue>,
                 n_full_state>;
  using transition_dist_type =
      std::array<std::array<std::discrete_distribution<index_type>, n_queue>,
                 n_full_state>;
  using action_dist_type =
      std::array<std::discrete_distribution<index_type>, n_obs_state>;
  using reward_config_type = std::array<float_type, n_full_state>;

  using reward_func_type =
      std::function<float_type(const env_cost_type&, const full_state_type&)>;

  using q_type = Eigen::Matrix<float_type, n_obs_state, n_queue>;
  using n_visit_type = Eigen::Matrix<uint64_t, n_obs_state, n_queue>;
  using qs_type =
      std::conditional_t<save_qs, std::vector<float_type>, std::nullptr_t>;

  using v_type =
      std::conditional_t<n_env == 1, Eigen::Matrix<float_type, n_obs_state, 1>,
                         std::nullptr_t>;
  using policy_v_type =
      std::conditional_t<n_env == 1, Eigen::Matrix<index_type, n_obs_state, 1>,
                         std::nullptr_t>;

  static constexpr auto q_inf_idx = ([]() {
    std::array<std::pair<index_type, index_type>,
               ((obs_state_idx.size() / (queue_lens_t + 1)) + ...)>
        inf_idx{};
    index_type cur{};
    for (index_type i = 0; i < dim_obs_state; ++i) {
      for (index_type j = 0; j < n_obs_state; ++j) {
        if (!obs_state_idx[j][i]) {
          inf_idx[cur++] = std::make_pair(j, i);
        }
      }
    }
    return inf_idx;
  })();

  Env(const float_type* env_cost, const float_type* env_arrival,
      const float_type* env_departure, const float_type* env_prob,
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
    q_.setZero();
    for (index_type i = 0; i < q_inf_idx.size(); ++i) {
      const auto [s, a] = q_inf_idx[i];
      q_(s, a) = -inf_v;
    }
    q_(0, 0) = 0;

    n_visit_.setZero();
  }

  void reset_q_epoch(uint64_t seed) {
    rng_ = decltype(rng_)(seed);
    state_ = 0;
  }

  void step(index_type action) {
    auto next_state_idx = transition_dist_[state_][action](rng_);
    state_ = transition_config_[state_][action].innerIndexPtr()[next_state_idx];
  }

  void train_q(float_type gamma, float_type eps, float_type decay, size_t epoch,
               uint64_t ls, uint64_t seed) {
    static constexpr auto iota_n_queue = make_iota<0, n_queue>();
    reset_q();

    if constexpr (save_qs) {
      qs_.resize(epoch * ls * q_.size());
    }

    for (index_type i = 0; i < epoch; ++i) {
      reset_q_epoch(seed);

      for (index_type j = 0; j < ls; ++j) {
        const auto state = state_;
        const auto obs_state = map_full_obs[state];

        index_type a;
        if (greedy_dis_(rng_) < eps) {
          a = action_dist_[obs_state](rng_);
        } else {
          q_(obs_state, Eigen::all).maxCoeff(&a);
        }

        step(a);
        const auto next_state = state_;
        const auto next_obs_state = map_full_obs[next_state];

        const auto next_q = q_(next_obs_state, Eigen::all).maxCoeff();
        const auto reward = reward_config_[state];

        q_(obs_state, a) +=
            (static_cast<float_type>(1) / ++n_visit_(obs_state, a)) *
            (reward + gamma * next_q - q_(obs_state, a));

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
                      iota_n_obs_state.end(),
                      [gamma, this, &v_i](index_type j) {
                        update_v_i<false>(gamma, j, v_i);
                      });
      }
      std::for_each(std::execution::par_unseq, iota_n_obs_state.begin(),
                    iota_n_obs_state.end(), [gamma, this](index_type j) {
                      update_v_i<true>(gamma, j, v_);
                    });
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

  index_type state_;

  q_type q_;
  n_visit_type n_visit_;
  qs_type qs_;

  v_type v_;
  policy_v_type policy_v_;

  XoshiroCpp::Xoshiro256Plus rng_;
  std::uniform_real_distribution<float_type> greedy_dis_;

  auto init_transition_config() {
    transition_config_type config;

    for (index_type i = 0; i < n_full_state; ++i) {
      const auto& s_i = full_state_idx[i];

      for (index_type a = 0; a < n_queue; ++a) {
        if ((map_full_obs[i] && !s_i[a]) || (!map_full_obs[i] && a)) continue;

        float_type dummy_prob = 1;
        auto& config_i_a = config[i][a];

        for (index_type j = 0; j < n_full_state; ++j) {
          if (i == j) continue;

          const auto next_to = can_transition_to<dim_full_state, n_queue>(
              s_i, full_state_idx[j], a);

          if (next_to) {
            float_type prob;

            const auto& [idx_q, change] = next_to.value();
            if (idx_q >= n_queue) {
              prob = env_prob_(idx_q - n_queue, std::get<index_type>(change));
            } else {
              if (std::get<bool>(change)) {
                prob = env_arrival_(idx_q, s_i[idx_q + n_queue]);
              } else {
                prob = env_departure_(idx_q, s_i[idx_q + n_queue]);
              }
            }
            config_i_a.insertBack(j) = prob;

            dummy_prob -= prob;
          }
        }

        config_i_a.coeffRef(i) = dummy_prob;
      }
    }

    return config;
  }

  auto init_transition_dist() {
    transition_dist_type dist;

    for (index_type i = 0; i < n_full_state; ++i) {
      for (index_type a = 0; a < n_queue; ++a) {
        const auto& config = transition_config_[i][a];
        const auto non_zero = config.nonZeros();
        if (non_zero) {
          dist[i][a] = std::discrete_distribution<index_type>(
              config.valuePtr(), config.valuePtr() + config.nonZeros());
        }
      }
    }

    return dist;
  }

  auto init_action_dist() {
    action_dist_type dist;
    for (index_type i = 0; i < n_obs_state; ++i) {
      std::array<index_type, n_queue> possible_action{};

      for (index_type j = 0; j < n_queue; ++j) {
        if (transition_config_[i][j].nonZeros()) {
          possible_action[j] = 1;
        }
      }

      dist[i] = std::discrete_distribution<index_type>(possible_action.begin(),
                                                       possible_action.end());
    }

    return dist;
  }

  auto init_reward_config(const reward_func_type& reward_func) {
    reward_config_type config;

    for (index_type i = 0; i < n_full_state; ++i) {
      config[i] = reward_func(env_cost_, full_state_idx[i]);
    }

    return config;
  }

  template <bool update_policy>
  void update_v_i(float_type gamma, index_type j, const v_type& v_i) {
    float_type max_v = -inf_v;
    index_type max_a = 0;

    for (index_type a = 0; a < n_queue; ++a) {
      float_type val_v = reward_config_[j];
      const auto& config = transition_config_[j][a];

      if (!config.nonZeros()) continue;
      val_v += gamma * config.transpose() * v_i;

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

enum class Reward { linear_2, convex_2 };

static constexpr auto linear_reward_2 = [](const auto& env_cost,
                                           const auto& state, auto offset) {
  return -(env_cost(0, state[offset + 0]) * state[0] +
           env_cost(1, state[offset + 1]) * state[1]);
};

static constexpr auto convex_reward_2 =
    [](const auto& env_cost, const auto& state, auto offset, auto cost_eps) {
      return -(env_cost(0, state[offset + 0]) * state[0] +
               env_cost(1, state[offset + 1]) *
                   (state[1] * state[1] * cost_eps + state[1]));
    };

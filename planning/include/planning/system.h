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
#include "tsl/sparse_map.h"
#include "unsupported/Eigen/CXX11/Tensor"

using index_type = Eigen::Index;

XoshiroCpp::Xoshiro256Plus rng;

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

using transition_status_type = std::optional<index_type>;

template <index_type dim_state, index_type n_class>
static constexpr transition_status_type can_transition_to(const auto& s1,
                                                          const auto& s2,
                                                          index_type action) {
  transition_status_type res{};

  for (index_type i = 0; i < dim_state; ++i) {
    const auto diff = s2[i] - s1[i];
    if (diff) {
      if (res) {
        return std::nullopt;
      } else if ((i >= n_class) ||
                 ((diff == -1 && i == action) || (diff == 1))) {
        res = i;
      } else {
        return std::nullopt;
      }
    }
  }

  return res;
}

template <index_type n_env_t, typename float_t, bool save_qs_t,
          index_type... limits_t>
class System {
 public:
  static constexpr index_type n_class = sizeof...(limits_t);
  static constexpr auto n_env = n_env_t;

  using float_type = float_t;
  static constexpr auto inf_v = std::numeric_limits<float_type>::infinity();

  static constexpr auto save_qs = save_qs_t;
  static constexpr std::array class_dims = {limits_t + 1 ...};

  using costs_type = Eigen::Matrix<float_type, n_class, n_env>;
  using arrivals_type = costs_type;
  using departures_type = costs_type;
  using env_trans_mats_type =
      Eigen::TensorFixedSize<float_type, Eigen::Sizes<n_class, n_env, n_env>>;

  static constexpr auto seq_nc =
      std::make_integer_sequence<index_type, n_class>{};

  static constexpr auto states =
      ([]<index_type... i>(std::integer_sequence<index_type, i...>) {
        return make_set_product<class_dims[i]..., n_env*(i - i + 1)...>();
      })(seq_nc);

  static constexpr auto n_state = states.size();
  using states_type = typename decltype(states)::value_type;
  static constexpr auto dim_state = std::tuple_size<states_type>::value;

  static constexpr auto cls_states =
      ([]<index_type... i>(std::integer_sequence<index_type, i...>) {
        return make_set_product<class_dims[i]...>();
      })(seq_nc);

  static constexpr auto n_cls_state = cls_states.size();
  using cls_state_type = typename decltype(cls_states)::value_type;
  static constexpr auto dim_cls_state = std::tuple_size<cls_state_type>::value;

  static constexpr auto to_cls_state = ([]() {
    std::array<index_type, n_state> res;
    for (index_type i = 0; i < n_state; ++i) {
      res[i] = i % n_cls_state;
    }
    return res;
  })();

  static constexpr auto env_states =
      ([]<index_type... i>(std::integer_sequence<index_type, i...>) {
        return make_set_product<n_env*(i - i + 1)...>();
      })(seq_nc);

  static constexpr auto n_env_state = env_states.size();
  using env_states_type = typename decltype(env_states)::value_type;
  static constexpr auto dim_env_state = std::tuple_size<env_states_type>::value;

  static constexpr auto to_env_state = ([]() {
    std::array<index_type, n_state> res;
    for (index_type i = 0; i < n_state; ++i) {
      res[i] = i / n_cls_state;
    }
    return res;
  })();

  using sp_vec_type = Eigen::SparseVector<float_type>;
  using sp_vec_it = typename sp_vec_type::InnerIterator;

  /* -------------------------- system transitions -------------------------- */

  using trans_probs_type =
      std::array<std::array<sp_vec_type, n_class>, n_state>;
  using trans_dists_type =
      std::array<std::array<std::discrete_distribution<index_type>, n_class>,
                 n_state>;
  using action_dists_type =
      std::array<std::discrete_distribution<index_type>, n_cls_state>;

  /* -------------------------------- rewards ------------------------------- */

  using rewards_type = std::array<float_type, n_state>;
  using reward_func_type =
      std::function<float_type(const costs_type&, const states_type&)>;

  /* --------------------------------- train -------------------------------- */

  using policy_type = Eigen::Matrix<index_type, n_cls_state, 1>;

  /* ------------------------------ q learning ------------------------------ */

  using q_type = Eigen::Matrix<float_type, n_cls_state, n_class>;
  using q_n_visit_type = Eigen::Matrix<uint64_t, n_cls_state, n_class>;
  using qs_type =
      std::conditional_t<save_qs, std::vector<float_type>, std::nullptr_t>;

  /* ---------------------------- value iteration --------------------------- */

  using cls_trans_probs_type =
      std::array<std::array<sp_vec_type, n_class>, n_cls_state>;
  using cls_rewards_type = std::array<float_type, n_cls_state>;

  using v_type = Eigen::Matrix<float_type, n_cls_state, 1>;

  /* ----------------- additional precomputed probabilities ----------------- */

  // P(S'|S, E, a) by (S', a, S, E)
  using state_cls_trans_probs_type = std::array<
      std::array<tsl::sparse_map<index_type,
                                 Eigen::Matrix<float_type, 1, n_env_state>>,
                 n_class>,
      n_cls_state>;

  // P(E'|E) by (E, E')
  using env_trans_probs_type =
      Eigen::Matrix<float_type, n_env_state, n_env_state>;

  /* --------------------------------- tilde -------------------------------- */

  using t_env_probs_type = Eigen::Matrix<float_type, n_env_state, 1>;

  /* ------------------------------ constructor ----------------------------- */

  System(const float_type* costs, const float_type* arrivals,
         const float_type* departures, const float_type* env_trans_mats,
         const reward_func_type& reward_func,
         const std::optional<float_type>& normalized_c = std::nullopt)
      : costs(costs),
        arrivals(arrivals),
        departures(departures),
        env_trans_mats(init_env_trans_mats(env_trans_mats)),
        normalized_c(init_normalized_c(normalized_c)),
        trans_probs(init_trans_probs()),
        trans_dists_(init_trans_dists()),
        action_dists_(init_action_dists()),
        state_cls_trans_probs(init_state_cls_trans_probs()),
        env_trans_probs(init_env_trans_probs()),
        rewards(init_rewards(reward_func)) {}

  /* -------------------------- system transitions -------------------------- */

  void step(index_type action) {
    auto next_state_idx = trans_dists_[state_][action](rng);
    state_ = trans_probs[state_][action].innerIndexPtr()[next_state_idx];
  }

  /* --------------------------- train q learning --------------------------- */

  void reset_q() {
    static constexpr auto q_inf_idx = ([]() {
      std::array<std::pair<index_type, index_type>,
                 ((cls_states.size() / (limits_t + 1)) + ...)>
          inf_idx{};
      index_type cur{};
      for (index_type i = 0; i < dim_cls_state; ++i) {
        for (index_type j = 0; j < n_cls_state; ++j) {
          if (!cls_states[j][i]) {
            inf_idx[cur++] = std::make_pair(j, i);
          }
        }
      }
      return inf_idx;
    })();

    q_.setZero();
    for (index_type i = 0; i < q_inf_idx.size(); ++i) {
      const auto [s, a] = q_inf_idx[i];
      q_(s, a) = -inf_v;
    }
    q_(0, 0) = 0;

    q_n_visit_.setZero();
  }

  void reset_q_epoch(uint64_t seed) {
    rng = decltype(rng)(seed);
    state_ = 0;
  }

  void train_q(float_type gamma, float_type greedy_eps, uint64_t ls,
               uint64_t seed) {
    reset_q();

    if constexpr (save_qs) {
      qs_.resize(ls * q_.size());
    }

    reset_q_epoch(seed);

    for (index_type i = 0; i < ls; ++i) {
      const auto state = state_;
      const auto cls_state = to_cls_state[state];

      index_type a;
      if (q_greedy_dis_(rng) < greedy_eps) {
        a = action_dists_[cls_state](rng);
      } else {
        q_.row(cls_state).maxCoeff(&a);
      }

      step(a);
      const auto next_state = state_;
      const auto next_cls_state = to_cls_state[next_state];

      const auto next_q = q_(next_cls_state, Eigen::all).maxCoeff();
      const auto reward = rewards[state];

      q_(cls_state, a) +=
          (static_cast<float_type>(1) / ++q_n_visit_(cls_state, a)) *
          (reward + gamma * next_q - q_(cls_state, a));

      if constexpr (save_qs) {
        std::copy(q_.data(), q_.data() + q_.size(),
                  qs_.begin() + i * q_.size());
      }
    }

    for (index_type i = 0; i < n_cls_state; ++i) {
      q_.row(i).maxCoeff(&q_policy_(i));
    }
  }

  /* ------------------------- train value iteration ------------------------ */

  void train_v(const cls_trans_probs_type& cls_trans_probs,
               const cls_rewards_type& cls_rewards, float_type gamma,
               uint64_t ls) {
    static constexpr auto iota_n_cls_state = make_iota<0, n_cls_state>();

    v_.fill(0);
    for (uint64_t i = 0; i < ls; ++i) {
      const auto v_i = v_;
      std::for_each(
          std::execution::par_unseq, iota_n_cls_state.begin(),
          iota_n_cls_state.end(),
          [&cls_trans_probs, &cls_rewards, gamma, this, &v_i](index_type j) {
            update_v_i<false>(cls_trans_probs, cls_rewards, gamma, j, v_i);
          });
    }
    std::for_each(std::execution::par_unseq, iota_n_cls_state.begin(),
                  iota_n_cls_state.end(),
                  [&cls_trans_probs, &cls_rewards, gamma, this](index_type j) {
                    update_v_i<true>(cls_trans_probs, cls_rewards, gamma, j,
                                     v_);
                  });
  }

  void train_v_s(float_type gamma, uint64_t ls) {
    if constexpr (n_env == 1) {
      train_v(trans_probs, rewards, gamma, ls);
    }
  }

  /* ------------------------------ train tilde ----------------------------- */

  void train_t(uint64_t ls) {
    t_env_probs_.setConstant(static_cast<float_type>(1) / n_env_state);

    for (uint64_t i = 0; i < ls; ++i) {
      t_env_probs_ = t_env_probs_.transpose() * env_trans_probs;
    }

    for (index_type j = 0; j < n_cls_state; ++j) {
      for (index_type a = 0; a < n_class; ++a) {
        const auto& probs = state_cls_trans_probs[j][a];

        for (const auto& [i, prob] : probs) {
          t_cls_trans_probs_[i][a].coeffRef(j) = prob * t_env_probs_;
        }
      }
    }

    for (index_type i = 0; i < n_state; ++i) {
      const auto i_cls = to_cls_state[i];
      const auto i_env = to_env_state[i];
      t_cls_rewards_[i_cls] += rewards[i] * t_env_probs_[i_env];
    }
  }

  void train_v_t(float_type gamma, uint64_t ls) {
    train_v(t_cls_trans_probs_, t_cls_rewards_, gamma, ls);
  }

  /* ------------------------------ q learning ------------------------------ */

  const q_type& q() const { return q_; }
  const q_n_visit_type& q_n_visit() const { return q_n_visit_; }
  const policy_type& q_policy() const { return q_policy_; }
  const qs_type& qs() const { return qs_; }

  /* ---------------------------- value iteration --------------------------- */

  const v_type& v() const { return v_; }
  const policy_type& v_policy() const { return v_policy_; }

  /* --------------------------------- tilde -------------------------------- */

  const t_env_probs_type& t_env_probs() const { return t_env_probs_; }
  const cls_trans_probs_type& t_cls_trans_probs() const {
    return t_cls_trans_probs_;
  }
  const cls_rewards_type& t_cls_rewards() const { return t_cls_rewards_; }

  /* --------------------- variable - system parameters --------------------- */

  const costs_type costs;
  const arrivals_type arrivals;
  const departures_type departures;
  const env_trans_mats_type env_trans_mats;
  const float_type normalized_c;

  /* -------------------- variables - system transitions -------------------- */

  const trans_probs_type trans_probs;

  /* -------------------------- variables - rewards ------------------------- */

  const rewards_type rewards;

  /* ----------- variables - additional precomputed probabilities ----------- */

  const state_cls_trans_probs_type state_cls_trans_probs;
  const env_trans_probs_type env_trans_probs;

 private:
  /* -------------------- variables - system transitions -------------------- */

  trans_dists_type trans_dists_;
  action_dists_type action_dists_;

  index_type state_;

  /* ------------------------------ q learning ------------------------------ */

  q_type q_;
  q_n_visit_type q_n_visit_;
  qs_type qs_;
  policy_type q_policy_;

  std::uniform_real_distribution<float_type> q_greedy_dis_;

  /* ---------------------------- value iteration --------------------------- */

  v_type v_;
  policy_type v_policy_;

  /* --------------------------------- tilde -------------------------------- */

  t_env_probs_type t_env_probs_;
  cls_trans_probs_type t_cls_trans_probs_;
  cls_rewards_type t_cls_rewards_;

  /* ------------------ init functions - system transitions ----------------- */

  auto init_env_trans_mats(const float_type* env_trans_mats) {
    env_trans_mats_type probs = Eigen::TensorMap<const env_trans_mats_type>(
        env_trans_mats, n_class, n_env, n_env);

    for (index_type i = 0; i < n_class; ++i) {
      for (index_type j = 0; j < n_env; ++j) {
        probs(i, j, j) = 0;
      }
    }

    return probs;
  }

  float_type init_normalized_c(const std::optional<float_type>& normalized_c) {
    if (normalized_c) return normalized_c.value();

    static const Eigen::array<index_type, 1> sum_dims({2});
    static const Eigen::array<index_type, 1> max_dims({1});

    return Eigen::Tensor<double, 0>(
        arrivals.rowwise().maxCoeff().sum() + departures.maxCoeff() +
        env_trans_mats.sum(sum_dims).maximum(max_dims).sum())(0);
  }

  auto init_trans_probs() {
    trans_probs_type probs;

    for (index_type i = 0; i < n_state; ++i) {
      const auto& s_i = states[i];

      for (index_type a = 0; a < n_class; ++a) {
        if ((to_cls_state[i] && !s_i[a]) || (!to_cls_state[i] && a)) continue;

        float_type dummy_prob = normalized_c;
        auto& prob_i_a = probs[i][a];

        for (index_type j = 0; j < n_state; ++j) {
          if (i == j) continue;

          const auto& s_j = states[j];
          const auto next_to =
              can_transition_to<dim_state, n_class>(s_i, s_j, a);

          if (next_to) {
            float_type prob;

            const auto idx = next_to.value();

            if (idx >= n_class) {
              prob = env_trans_mats(idx - n_class, s_i[idx], s_j[idx]);
            } else if (s_i[idx] < s_j[idx]) {
              prob = arrivals(idx, s_i[idx + n_class]);
            } else {
              prob = departures(idx, s_i[idx + n_class]);
            }

            prob_i_a.insertBack(j) = prob;

            dummy_prob -= prob;
          }
        }

        prob_i_a.coeffRef(i) = dummy_prob;
        prob_i_a /= normalized_c;
      }
    }

    return probs;
  }

  auto init_trans_dists() {
    trans_dists_type dists;

    for (index_type i = 0; i < n_state; ++i) {
      for (index_type a = 0; a < n_class; ++a) {
        const auto& probs = trans_probs[i][a];
        const auto n_non_zero = probs.nonZeros();
        if (n_non_zero) {
          dists[i][a] = std::discrete_distribution<index_type>(
              probs.valuePtr(), probs.valuePtr() + n_non_zero);
        }
      }
    }

    return dists;
  }

  auto init_action_dists() {
    action_dists_type dists;
    for (index_type i = 0; i < n_cls_state; ++i) {
      std::array<index_type, n_class> possible_actions{};

      for (index_type j = 0; j < n_class; ++j) {
        if (trans_probs[i][j].nonZeros()) {
          possible_actions[j] = 1;
        }
      }

      dists[i] = std::discrete_distribution<index_type>(
          possible_actions.begin(), possible_actions.end());
    }

    return dists;
  }

  /* ----------------------- init functions - rewards ----------------------- */

  auto init_rewards(const reward_func_type& reward_func) {
    rewards_type rewards;

    for (index_type i = 0; i < n_state; ++i) {
      rewards[i] = reward_func(costs, states[i]);
    }

    return rewards;
  }

  /* --------- init functions - additional precomputed probabilities -------- */

  auto init_state_cls_trans_probs() {
    state_cls_trans_probs_type probs;

    for (index_type i = 0; i < n_state; ++i) {
      const auto i_cls = to_cls_state[i];
      const auto i_env = to_env_state[i];

      for (index_type a = 0; a < n_class; ++a) {
        const auto& trans_prob = trans_probs[i][a];

        for (sp_vec_it it(trans_prob); it; ++it) {
          const auto j = it.index();
          const auto j_cls = to_cls_state[j];
          const auto prob = it.value();
          probs[j_cls][a]
              .try_emplace(i_cls,
                           Eigen::Matrix<float_type, 1, n_env_state>::Zero())
              .first.value()[i_env] += prob;
        }
      }
    }

    return probs;
  }

  auto init_env_trans_probs() {
    env_trans_probs_type probs;
    probs.setZero();

    for (index_type i = 0; i < n_env_state; ++i) {
      const auto& trans_prob = trans_probs[i * n_cls_state][0];

      for (sp_vec_it it(trans_prob); it; ++it) {
        const auto j = it.index();
        const auto j_env = to_env_state[j];
        const auto prob = it.value();

        probs(i, j_env) += prob;
      }
    }

    return probs;
  }

  /* -------------------- train value iteration internal -------------------- */

  template <bool update_policy>
  void update_v_i(const cls_trans_probs_type& cls_trans_probs,
                  const cls_rewards_type& cls_rewards, float_type gamma,
                  index_type j, const v_type& v_i) {
    float_type max_v = -inf_v;
    index_type max_a = 0;

    for (index_type a = 0; a < n_class; ++a) {
      float_type val_v = cls_rewards[j];
      const auto& probs = cls_trans_probs[j][a];

      if (!probs.nonZeros()) continue;
      val_v += gamma * probs.transpose() * v_i;

      if (max_v < val_v) {
        max_v = val_v;
        if constexpr (update_policy) {
          max_a = a;
        }
      }
    }

    if constexpr (update_policy) {
      v_policy_[j] = max_a;
    } else {
      v_[j] = max_v;
    }
  }
};

enum class Reward { linear_2, convex_2 };

static constexpr auto linear_reward_2 = [](const auto& costs, const auto& state,
                                           auto offset) {
  return -(costs(0, state[offset + 0]) * state[0] +
           costs(1, state[offset + 1]) * state[1]);
};

static constexpr auto convex_reward_2 = [](const auto& costs, const auto& state,
                                           auto offset, auto cost_eps) {
  return -(costs(0, state[offset + 0]) * state[0] +
           costs(1, state[offset + 1]) *
               (state[1] * state[1] * cost_eps + state[1]));
};

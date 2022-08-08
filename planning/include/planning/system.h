#pragma once

#include <array>
#include <cmath>
#include <execution>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "cereal/access.hpp"
#include "planning/serialize.h"
#include "planning/state.h"
#include "planning/xoshiro.h"
#include "tsl/robin_map.h"
#include "unsupported/Eigen/CXX11/Tensor"

using float_type = double;
using sp_mat_type = Eigen::SparseMatrix<float_type, storage_order, index_type>;
using sp_mat_it = typename sp_mat_type::InnerIterator;

XoshiroCpp::Xoshiro256Plus rng;

namespace std {

bool operator==(const sp_mat_type& lhs, const sp_mat_type& rhs) {
  return (lhs.isCompressed() == rhs.isCompressed()) &&
         (lhs.rows() == rhs.rows()) && (lhs.cols() == rhs.cols()) &&
         (lhs.nonZeros() == rhs.nonZeros()) &&
         std::equal(lhs.innerIndexPtr(), lhs.innerIndexPtr() + lhs.nonZeros(),
                    rhs.innerIndexPtr()) &&
         std::equal(lhs.outerIndexPtr(),
                    lhs.outerIndexPtr() + lhs.outerSize() + 1,
                    rhs.outerIndexPtr()) &&
         std::equal(lhs.valuePtr(), lhs.valuePtr() + lhs.nonZeros(),
                    rhs.valuePtr()) &&
         (lhs.isCompressed() ||
          std::equal(lhs.innerNonZeroPtr(),
                     lhs.innerNonZeroPtr() + lhs.outerSize(),
                     rhs.innerNonZeroPtr()));
}

}  // namespace std

bool operator==(const sp_mat_type& lhs, const sp_mat_type& rhs) {
  return std::operator==(lhs, rhs);
}

template <typename XprType>
typename XprType::Scalar to_scalar(const XprType& xpr) {
  return Eigen::Tensor<typename XprType::Scalar, 0, storage_order>(xpr)(0);
}

template <index_type n_env_t, index_type... limits_t>
class System {
 public:
  using state = State<n_env_t, limits_t...>;

  static constexpr auto& n_env = state::n_env;

  static constexpr auto& n_cls = state::n_cls;
  static constexpr auto& cls_dims = state::cls_dims;

  static constexpr auto inf_v = std::numeric_limits<float_type>::infinity();
  static constexpr auto eps_v = std::numeric_limits<float_type>::epsilon();

  using costs_type =
      Eigen::TensorFixedSize<float_type, Eigen::Sizes<n_cls, n_env>,
                             storage_order>;
  using arrivals_type = costs_type;
  using departures_type = costs_type;
  using env_trans_mats_type =
      Eigen::TensorFixedSize<float_type, Eigen::Sizes<n_cls, n_env, n_env>,
                             storage_order>;

  /* -------------------------- system transitions -------------------------- */
  using env_state_accessible_type =
      Eigen::Matrix<bool, 1, state::env::n, storage_order>;

  using trans_probs_type = std::array<sp_mat_type, state::sys::n>;
  using trans_dists_type =
      std::array<std::array<std::discrete_distribution<index_type>, n_cls>,
                 state::sys::n>;
  using action_masks_type =
      Eigen::Matrix<bool, state::cls::n, n_cls, storage_order>;
  using action_dists_type =
      std::array<std::discrete_distribution<index_type>, state::cls::n>;

  /* -------------------------------- rewards ------------------------------- */

  using rewards_type =
      Eigen::Matrix<float_type, 1, state::sys::n, storage_order>;
  using reward_func_type = std::function<float_type(
      const costs_type&, const typename state::sys::row_type&)>;

  /* ---------------------- class states - interactive ---------------------- */

  using n_cls_visit_type =
      Eigen::Matrix<uint64_t, state::cls::n, n_cls, storage_order>;
  using n_cls_trans_type =
      std::array<std::array<tsl::robin_map<index_type, uint64_t>, n_cls>,
                 state::cls::n>;
  using cls_cum_rewards_type =
      Eigen::Matrix<float_type, 1, state::cls::n, storage_order>;

  /* --------------------------------- train -------------------------------- */

  using policy_type =
      Eigen::Matrix<index_type, 1, state::cls::n, storage_order>;

  /* ------------------------------ q learning ------------------------------ */

  using q_type = Eigen::Matrix<float_type, state::cls::n, n_cls, storage_order>;
  using qs_type = Eigen::Tensor<float_type, 3, storage_order>;
  using q_greedy_dis_type = std::uniform_real_distribution<float_type>;

  /* ---------------------------- value iteration --------------------------- */

  using cls_trans_probs_type = std::array<sp_mat_type, state::cls::n>;
  using cls_rewards_type =
      Eigen::Matrix<float_type, 1, state::cls::n, storage_order>;

  using v_type = Eigen::Matrix<float_type, 1, state::cls::n, storage_order>;

  /* ----------------- additional precomputed probabilities ----------------- */

  // P(S'|S, E, a) by (S', a, S, E)
  using state_cls_trans_env_prob_type =
      Eigen::Matrix<float_type, 1, state::env::n, storage_order>;
  using state_cls_trans_probs_type = std::array<
      std::array<tsl::robin_map<index_type, state_cls_trans_env_prob_type>,
                 n_cls>,
      state::cls::n>;

  // P(E'|E) by (E, E')
  using env_trans_probs_type = sp_mat_type;

  /* --------------------------------- tilde -------------------------------- */

  using t_env_probs_type =
      Eigen::Matrix<float_type, 1, state::env::n, storage_order>;

  /* ------------------------------ constructor ----------------------------- */

  System(const float_type* costs, const float_type* arrivals,
         const float_type* departures, const float_type* env_trans_mats,
         const reward_func_type& reward_func,
         const std::optional<float_type>& normalized_c = std::nullopt)
      : costs(Eigen::TensorMap<const costs_type>(costs, n_cls, n_env)),
        arrivals(Eigen::TensorMap<const arrivals_type>(arrivals, n_cls, n_env)),
        departures(
            Eigen::TensorMap<const departures_type>(departures, n_cls, n_env)),
        env_trans_mats(init_env_trans_mats(env_trans_mats)),
        normalized_c(init_normalized_c(normalized_c)),
        env_state_accessible(init_env_state_accessible(env_trans_mats)),
        trans_probs(init_trans_probs()),
        trans_dists_(init_trans_dists()),
        action_masks(init_action_masks()),
        action_dists_(init_action_dists()),
        state_cls_trans_probs(init_state_cls_trans_probs()),
        env_trans_probs(init_env_trans_probs()),
        rewards(init_rewards(reward_func)) {}

  /* -------------------------- system transitions -------------------------- */

  void step(index_type action) {
    auto next_state_idx = trans_dists_[state_][action](rng);
    state_ = trans_probs[state_]
                 .innerIndexPtr()[trans_probs[state_].outerIndexPtr()[action] +
                                  next_state_idx];
  }

  void reset(uint64_t seed) {
    rng = decltype(rng)(seed);
    state_ = 0;
  }

  void reset_interactive() {
    n_cls_visit_.setZero();
    n_cls_trans_.fill({});
    cls_cum_rewards_.setZero();
    reset_cls_trans_probs(i_cls_trans_probs_);
    i_cls_rewards_.setZero();
  }

  void reset_cls_trans_probs(cls_trans_probs_type& t) {
    static constexpr auto reserve_sizes = ([]() {
      std::array<index_type, n_cls> sizes{};
      sizes.fill(n_cls + 1 + 1);
      return sizes;
    })();

    for (size_t i = 0; i < state::cls::n; ++i) {
      t[i] = sp_mat_type(n_cls, state::cls::n);
      t[i].reserve(reserve_sizes);
    }
  }

  /* --------------------------- train q learning --------------------------- */

  void reset_q() {
    static constexpr auto q_inf_idx = ([]() {
      std::array<std::pair<index_type, index_type>,
                 ((state::cls::n / (limits_t + 1)) + ...)>
          inf_idx{};
      index_type c = 0;
      for (index_type i = 0; i < state::cls::n; ++i) {
        for (index_type a = 0; a < n_cls; ++a) {
          if (!state::cls::f[i * n_cls + a]) {
            inf_idx[c++] = std::make_pair(i, a);
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
  }

  template <bool log_i_t, bool log_qs_t>
  void train_q(float_type gamma, float_type greedy_eps, uint64_t ls,
               uint64_t seed) {
    reset(seed);
    reset_interactive();
    reset_q();

    static constexpr uint64_t qs_limit = 100000000;
    const uint64_t qs_step = std::ceil(ls / qs_limit);

    if constexpr (log_qs_t) {
      qs_ = qs_type(std::max(ls, qs_limit) + (ls % qs_step == 0), state::cls::n,
                    n_cls);
    }

    for (index_type i = 0; i < ls; ++i) {
      const auto cls_state = state::to_cls[state_];

      index_type a;
      if (q_greedy_dis_(rng) < greedy_eps) {
        a = action_dists_[cls_state](rng);
      } else {
        q_.row(cls_state).maxCoeff(&a);
      }

      const auto reward = rewards[state_];

      step(a);
      ++n_cls_visit_(cls_state, a);

      const auto next_cls_state = state::to_cls[state_];

      const auto next_q = q_.row(next_cls_state).maxCoeff();
      q_(cls_state, a) +=
          (static_cast<float_type>(1) / n_cls_visit_(cls_state, a)) *
          (reward + gamma * next_q - q_(cls_state, a));

      if constexpr (log_i_t) {
        ++n_cls_trans_[cls_state][a].try_emplace(next_cls_state).first.value();
        cls_cum_rewards_[cls_state] += reward;
      }

      if constexpr (log_qs_t) {
        using qs_element_type = Eigen::TensorMap<Eigen::TensorFixedSize<
            float_type, Eigen::Sizes<state::cls::n, n_cls>, storage_order>>;
        if (!(i % qs_step)) {
          qs_.chip<0>(i / qs_step) =
              qs_element_type(q_.data(), state::cls::n, n_cls);
        } else if (i == ls) {
          qs_.chip<0>(qs_.dimension(0) - 1) =
              qs_element_type(q_.data(), state::cls::n, n_cls);
        }
      }
    }

    if constexpr (log_i_t) {
      for (index_type i = 0; i < state::cls::n; ++i) {
        for (index_type a = 0; a < n_cls; ++a) {
          const auto n_visit = n_cls_visit_(i, a);
          for (const auto& [j, v] : n_cls_trans_[i][a]) {
            i_cls_trans_probs_[i].insert(a, j) =
                static_cast<float_type>(v) / n_visit;
          }
        }
      }
      for (index_type i = 0; i < state::cls::n; ++i) {
        i_cls_trans_probs_[i].makeCompressed();
      }

      i_cls_rewards_ =
          cls_cum_rewards_.transpose().array() /
          n_cls_visit_.rowwise().sum().array().template cast<float_type>();
    }

    for (index_type i = 0; i < state::cls::n; ++i) {
      q_.row(i).maxCoeff(&q_policy_(i));
    }
  }

  /* ------------------------- train value iteration ------------------------ */

  void train_v(float_type gamma, uint64_t ls) {
    if constexpr (n_env == 1) {
      train_t(0);
    }
    static constexpr auto iota_cls = make_iota<0, state::cls::n>();

    v_.setZero();
    v_policy_.setZero();

    for (uint64_t i = 0; i < ls; ++i) {
      const auto v_i = v_;
      std::for_each(std::execution::par_unseq, iota_cls.begin(), iota_cls.end(),
                    [gamma, this, &v_i](index_type j) {
                      update_v_i<false>(gamma, j, v_i);
                    });
    }
    std::for_each(
        std::execution::par_unseq, iota_cls.begin(), iota_cls.end(),
        [gamma, this](index_type j) { update_v_i<true>(gamma, j, v_); });
  }

  /* ------------------------------ train tilde ----------------------------- */

  void train_t(uint64_t ls) {
    const auto initial_prob =
        static_cast<float_type>(1) / env_state_accessible.count();
    for (index_type i = 0; i < state::env::n; ++i) {
      t_env_probs_[i] = env_state_accessible[i] ? initial_prob : 0;
    }

    for (uint64_t i = 0; i < ls; ++i) {
      t_env_probs_ *= env_trans_probs;
    }

    reset_cls_trans_probs(t_cls_trans_probs_);

    for (index_type a = 0; a < n_cls; ++a) {
      for (index_type j = 0; j < state::cls::n; ++j) {
        const auto& probs = state_cls_trans_probs[j][a];

        for (const auto& [i, prob] : probs) {
          const auto p_i_a_j = prob.dot(t_env_probs_);
          if (p_i_a_j > eps_v) {
            t_cls_trans_probs_[i].insert(a, j) = p_i_a_j;
          }
        }
      }
    }
    for (index_type i = 0; i < state::cls::n; ++i) {
      t_cls_trans_probs_[i].makeCompressed();
    }

    t_cls_rewards_.setZero();

    for (index_type i = 0; i < state::sys::n; ++i) {
      const auto i_cls = state::to_cls[i];
      const auto i_env = state::to_env[i];
      t_cls_rewards_[i_cls] += rewards[i] * t_env_probs_[i_env];
    }
  }

  /* ---------------------- class states - interactive ---------------------- */

  const n_cls_visit_type& n_cls_visit() const { return n_cls_visit_; }

  /* ------------------------------ interactive ----------------------------- */

  const cls_trans_probs_type& i_cls_trans_probs() const {
    return i_cls_trans_probs_;
  }
  const cls_rewards_type& i_cls_rewards() const { return i_cls_rewards_; }

  /* ------------------------------ q learning ------------------------------ */

  const q_type& q() const { return q_; }
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

  /* ----------------------------- serialization ---------------------------- */

  void to_stream(std::ostream& os) const {
    cereal::BinaryOutputArchive ar(os);
    ar(*this);
  }

  static System from_stream(std::istream& is) {
    System sys;
    cereal::BinaryInputArchive ar(is);
    ar(sys);
    return sys;
  }

  void to_file(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    to_stream(os);
  }

  static System from_file(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    return from_stream(is);
  }

  std::string to_str() const {
    std::stringstream os;
    to_stream(os);
    return os.str();
  }

  static System from_str(const std::string& str) {
    std::stringstream is(str);
    return from_stream(is);
  }

  bool operator==(const System& other) const {
    return to_scalar((costs == other.costs).all()) &&
           to_scalar((arrivals == other.arrivals).all()) &&
           to_scalar((departures == other.departures).all()) &&
           to_scalar((env_trans_mats == other.env_trans_mats).all()) &&
           (normalized_c == other.normalized_c) &&
           (env_state_accessible == other.env_state_accessible) &&
           (trans_probs == other.trans_probs) &&
           (action_masks == other.action_masks) && (rewards == other.rewards) &&
           (state_cls_trans_probs == other.state_cls_trans_probs) &&
           (env_trans_probs == other.env_trans_probs) &&
           (trans_dists_ == other.trans_dists_) &&
           (action_dists_ == other.action_dists_) && (state_ == other.state_) &&
           (n_cls_visit_ == other.n_cls_visit_) &&
           (n_cls_trans_ == other.n_cls_trans_) &&
           (cls_cum_rewards_ == other.cls_cum_rewards_) &&
           (i_cls_trans_probs_ == other.i_cls_trans_probs_) &&
           (i_cls_rewards_ == other.i_cls_rewards_) && (q_ == other.q_) &&
           to_scalar((qs_ == other.qs_).all()) &&
           (q_policy_ == other.q_policy_) &&
           (q_greedy_dis_ == other.q_greedy_dis_) && (v_ == other.v_) &&
           (v_policy_ == other.v_policy_) &&
           (t_env_probs_ == other.t_env_probs_) &&
           (t_cls_trans_probs_ == other.t_cls_trans_probs_) &&
           (t_cls_rewards_ == other.t_cls_rewards_);
  }

  /* --------------------- variable - system parameters --------------------- */

  const costs_type costs;
  const arrivals_type arrivals;
  const departures_type departures;
  const env_trans_mats_type env_trans_mats;
  const float_type normalized_c = 0;

  /* -------------------- variables - system transitions -------------------- */

  const env_state_accessible_type env_state_accessible;
  const trans_probs_type trans_probs;
  const action_masks_type action_masks;

  /* -------------------------- variables - rewards ------------------------- */

  const rewards_type rewards;

  /* ----------- variables - additional precomputed probabilities ----------- */

  const state_cls_trans_probs_type state_cls_trans_probs;
  const env_trans_probs_type env_trans_probs;

 private:
  friend class cereal::access;

  /* -------------------- variables - system transitions -------------------- */

  trans_dists_type trans_dists_;
  action_dists_type action_dists_;

  /* ----------------------- variables - system states ---------------------- */

  index_type state_;
  n_cls_visit_type n_cls_visit_;
  n_cls_trans_type n_cls_trans_;
  cls_cum_rewards_type cls_cum_rewards_;

  /* ------------------------------ interactive ----------------------------- */

  cls_trans_probs_type i_cls_trans_probs_;
  cls_rewards_type i_cls_rewards_;

  /* ------------------------------ q learning ------------------------------ */

  q_type q_;
  qs_type qs_;
  policy_type q_policy_;

  q_greedy_dis_type q_greedy_dis_;

  /* ---------------------------- value iteration --------------------------- */

  v_type v_;
  policy_type v_policy_;

  /* --------------------------------- tilde -------------------------------- */

  t_env_probs_type t_env_probs_;
  cls_trans_probs_type t_cls_trans_probs_;
  cls_rewards_type t_cls_rewards_;

  /* ----------------------------- serialization ---------------------------- */

  System() {}

  System(costs_type&& costs, arrivals_type&& arrivals,
         departures_type&& departures, env_trans_mats_type&& env_trans_mats,
         float_type&& normalized_c,
         env_state_accessible_type&& env_state_accessible,
         trans_probs_type&& trans_probs, action_masks_type&& action_masks,
         rewards_type&& rewards,
         state_cls_trans_probs_type&& state_cls_trans_probs,
         env_trans_probs_type&& env_trans_probs, index_type&& state,
         n_cls_visit_type&& n_cls_visit, n_cls_trans_type&& n_cls_trans,
         cls_cum_rewards_type&& cls_cum_rewards,
         cls_trans_probs_type&& i_cls_trans_probs,
         cls_rewards_type&& i_cls_rewards, q_type&& q, qs_type&& qs,
         policy_type&& q_policy, v_type&& v, policy_type&& v_policy,
         t_env_probs_type&& t_env_probs,
         cls_trans_probs_type&& t_cls_trans_probs,
         cls_rewards_type&& t_cls_rewards)
      : costs(costs),
        arrivals(arrivals),
        departures(departures),
        env_trans_mats(env_trans_mats),
        normalized_c(normalized_c),
        env_state_accessible(env_state_accessible),
        trans_probs(trans_probs),
        action_masks(action_masks),
        rewards(rewards),
        state_cls_trans_probs(state_cls_trans_probs),
        env_trans_probs(env_trans_probs),
        trans_dists_(init_trans_dists()),
        action_dists_(init_action_dists()),
        state_(state),
        n_cls_visit_(n_cls_visit),
        n_cls_trans_(n_cls_trans),
        cls_cum_rewards_(cls_cum_rewards),
        i_cls_trans_probs_(i_cls_trans_probs),
        i_cls_rewards_(i_cls_rewards),
        q_(q),
        qs_(qs),
        q_policy_(q_policy),
        v_(v),
        v_policy_(v_policy),
        t_env_probs_(t_env_probs),
        t_cls_trans_probs_(t_cls_trans_probs),
        t_cls_rewards_(t_cls_rewards) {}

  void save(cereal::BinaryOutputArchive& ar) const {
    ar(costs, arrivals, departures, env_trans_mats, normalized_c,
       env_state_accessible, trans_probs, action_masks, rewards,
       state_cls_trans_probs, env_trans_probs, state_, n_cls_visit_,
       n_cls_trans_, cls_cum_rewards_, i_cls_trans_probs_, i_cls_rewards_, q_,
       qs_, q_policy_, v_, v_policy_, t_env_probs_, t_cls_trans_probs_,
       t_cls_rewards_);
  }

  void load(cereal::BinaryInputArchive& ar) {
    costs_type costs;
    arrivals_type arrivals;
    departures_type departures;
    env_trans_mats_type env_trans_mats;
    float_type normalized_c;
    env_state_accessible_type env_state_accessible;
    trans_probs_type trans_probs;
    action_masks_type action_masks;
    rewards_type rewards;
    state_cls_trans_probs_type state_cls_trans_probs;
    env_trans_probs_type env_trans_probs;
    index_type state;
    n_cls_visit_type n_cls_visit;
    n_cls_trans_type n_cls_trans;
    cls_cum_rewards_type cls_cum_rewards;
    cls_trans_probs_type i_cls_trans_probs;
    cls_rewards_type i_cls_rewards;
    q_type q;
    qs_type qs;
    policy_type q_policy;
    v_type v;
    policy_type v_policy;
    t_env_probs_type t_env_probs;
    cls_trans_probs_type t_cls_trans_probs;
    cls_rewards_type t_cls_rewards;

    ar(costs, arrivals, departures, env_trans_mats, normalized_c,
       env_state_accessible, trans_probs, action_masks, rewards,
       state_cls_trans_probs, env_trans_probs, state, n_cls_visit, n_cls_trans,
       cls_cum_rewards, i_cls_trans_probs, i_cls_rewards, q, qs, q_policy, v,
       v_policy, t_env_probs, t_cls_trans_probs, t_cls_rewards);

    new (this)
        System(std::move(costs), std::move(arrivals), std::move(departures),
               std::move(env_trans_mats), std::move(normalized_c),
               std::move(env_state_accessible), std::move(trans_probs),
               std::move(action_masks), std::move(rewards),
               std::move(state_cls_trans_probs), std::move(env_trans_probs),
               std::move(state), std::move(n_cls_visit), std::move(n_cls_trans),
               std::move(cls_cum_rewards), std::move(i_cls_trans_probs),
               std::move(i_cls_rewards), std::move(q), std::move(qs),
               std::move(q_policy), std::move(v), std::move(v_policy),
               std::move(t_env_probs), std::move(t_cls_trans_probs),
               std::move(t_cls_rewards));
  }

  /* ------------------ init functions - system transitions ----------------- */

  auto init_env_trans_mats(const float_type* env_trans_mats) {
    env_trans_mats_type probs = Eigen::TensorMap<const env_trans_mats_type>(
        env_trans_mats, n_cls, n_env, n_env);

    for (index_type i = 0; i < n_cls; ++i) {
      for (index_type j = 0; j < n_env; ++j) {
        probs(i, j, j) = 0;
      }
    }

    return probs;
  }

  float_type init_normalized_c(const std::optional<float_type>& normalized_c) {
    if (normalized_c) return normalized_c.value();
    return to_scalar(
        arrivals.maximum(std::array{1}).sum() + departures.maximum() +
        env_trans_mats.sum(std::array{2}).maximum(std::array{1}).sum());
  }

  auto init_env_state_accessible(const float_type* env_trans_mats) {
    env_state_accessible_type res;

    Eigen::TensorMap<const env_trans_mats_type> mats(env_trans_mats, n_cls,
                                                     n_env, n_env);
    Eigen::TensorFixedSize<float_type, Eigen::Sizes<n_cls, n_env>,
                           storage_order>
        sums = mats.sum(std::array{2});

    for (index_type i = 0; i < state::env::n; ++i) {
      index_type j;

      for (j = 0; j < state::env::d; ++j) {
        if (!sums(j, state::env::a(i, j))) {
          break;
        }
      }

      res[i] = (j == state::env::d);
    }

    return res;
  }

  auto init_trans_probs() {
    trans_probs_type probs;
    static constexpr auto reserve_sizes = ([]() {
      std::array<index_type, n_cls> sizes{};
      sizes.fill(n_cls + 1 + n_cls * (n_env - 1) + 1);
      return sizes;
    })();

    for (index_type i = 0; i < state::sys::n; ++i) {
      const auto& s_i = state::sys::a.row(i);

      auto prob_i = sp_mat_type(n_cls, state::sys::n);
      prob_i.reserve(reserve_sizes);

      for (index_type a = 0; a < n_cls; ++a) {
        if ((state::to_cls[i] && !s_i[a]) || (!state::to_cls[i] && a)) continue;

        float_type dummy_prob = normalized_c;

        for (index_type j = 0; j < state::sys::n; ++j) {
          const auto& s_j = state::sys::a.row(j);
          const auto next_to = state::next_to(s_i, s_j, a);

          if (next_to) {
            float_type prob;
            const auto n = next_to.value();

            if (n >= n_cls) {
              prob = env_trans_mats(n - n_cls, s_i[n], s_j[n]);
            } else if (s_i[n] < s_j[n]) {
              prob = arrivals(n, s_i[n + n_cls]);
            } else {
              prob = departures(n, s_i[n + n_cls]);
            }

            if (prob > eps_v) {
              prob_i.insert(a, j) = prob;
              dummy_prob -= prob;
            }
          }
        }

        if (dummy_prob > eps_v) {
          prob_i.insert(a, i) = dummy_prob;
        }
      }

      prob_i.makeCompressed();
      prob_i /= normalized_c;
      probs[i] = std::move(prob_i);
    }

    return probs;
  }

  auto init_trans_dists() {
    trans_dists_type dists;

    for (index_type i = 0; i < state::sys::n; ++i) {
      const auto& prob_i = trans_probs[i];

      for (index_type a = 0; a < n_cls; ++a) {
        const auto start = prob_i.outerIndexPtr()[a];
        const auto end = prob_i.outerIndexPtr()[a + 1];
        if (start != end) {
          dists[i][a] = std::discrete_distribution<index_type>(
              prob_i.valuePtr() + start, prob_i.valuePtr() + end);
        }
      }
    }

    return dists;
  }

  auto init_action_masks() {
    action_masks_type masks;
    for (index_type i = 0; i < state::cls::n; ++i) {
      const auto& prob_i = trans_probs[state::to_sys(i, 0)];
      masks.row(i) =
          (prob_i * Eigen::Matrix<float_type, state::cls::n, 1>::Ones())
              .template cast<bool>();
    }
    return masks;
  }

  auto init_action_dists() {
    action_dists_type dists;

    for (index_type i = 0; i < state::cls::n; ++i) {
      dists[i] = std::discrete_distribution<index_type>(
          action_masks.row(i).begin(), action_masks.row(i).end());
    }

    return dists;
  }

  /* ----------------------- init functions - rewards ----------------------- */

  auto init_rewards(const reward_func_type& reward_func) {
    rewards_type rewards;

    for (index_type i = 0; i < state::sys::n; ++i) {
      rewards[i] = reward_func(costs, state::sys::a.row(i));
    }

    return rewards;
  }

  /* --------- init functions - additional precomputed probabilities -------- */

  auto init_state_cls_trans_probs() {
    state_cls_trans_probs_type probs;

    for (index_type i = 0; i < state::sys::n; ++i) {
      const auto& prob_i = trans_probs[i];
      const auto i_cls = state::to_cls[i];
      const auto i_env = state::to_env[i];

      for (index_type a = 0; a < n_cls; ++a) {
        for (sp_mat_it it(prob_i, a); it; ++it) {
          const auto j = it.index();
          const auto j_cls = state::to_cls[j];
          const auto prob = it.value();

          probs[j_cls][a]
              .try_emplace(i_cls, Eigen::Matrix<float_type, 1, state::env::n,
                                                storage_order>::Zero())
              .first.value()[i_env] += prob;
        }
      }
    }

    return probs;
  }

  auto init_env_trans_probs() {
    env_trans_probs_type probs = sp_mat_type(state::env::n, state::env::n);
    static constexpr auto reserve_sizes = ([]() {
      std::array<index_type, state::env::n> sizes{};
      sizes.fill(n_cls * (n_env - 1) + 1);
      return sizes;
    })();
    probs.reserve(reserve_sizes);

    for (index_type i = 0; i < state::env::n; ++i) {
      if (!env_state_accessible[i]) {
        continue;
      }

      const auto& prob_i = trans_probs[state::to_sys(0, i)];

      for (sp_mat_it it(prob_i, 0); it; ++it) {
        const auto j = it.index();
        const auto j_env = state::to_env[j];
        const auto prob = it.value();

        probs.coeffRef(i, j_env) += prob;
      }
    }
    probs.makeCompressed();

    return probs;
  }

  /* -------------------- train value iteration internal -------------------- */

  template <bool update_policy_t>
  void update_v_i(float_type gamma, index_type j, const v_type& v_i) {
    index_type max_a;
    const auto& prob_j = t_cls_trans_probs_[j];
    const auto max_v =
        t_cls_rewards_[j] +
        gamma * action_masks.row(j)
                    .select(v_i * prob_j.transpose(), v_type::Constant(-inf_v))
                    .maxCoeff(&max_a);

    if constexpr (update_policy_t) {
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

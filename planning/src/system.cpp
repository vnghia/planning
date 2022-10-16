#include "planning/system.hpp"

#include <algorithm>
#include <execution>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

template <typename Derived>
bool operator==(const Eigen::SparseCompressedBase<Derived>& lhs,
                const Eigen::SparseCompressedBase<Derived>& rhs) {
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

template <typename Derived>
requires std::derived_from<Derived, Eigen::SparseCompressedBase<Derived>>
bool sps_equal(const std::vector<Derived>& lhs,
               const std::vector<Derived>& rhs) {
  return (lhs.size() == rhs.size()) &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                    [](const Eigen::SparseCompressedBase<Derived>& lhs,
                       const Eigen::SparseCompressedBase<Derived>& rhs) {
                      return lhs == rhs;
                    });
}

System::System(const index_type n_env, const VectorAI& limits,
               const float_type* costs, const float_type* arrivals,
               const float_type* departures, const float_type* env_trans_mats,
               const reward_func_type& reward_func,
               const std::optional<float_type>& normalized_c)
    : n_env(n_env),
      limits(limits),
      states(n_env, limits),
      n_cls(states.n_cls),
      costs(Eigen::TensorMap<const Tensor2F>(costs, n_cls, n_env)),
      arrivals(Eigen::TensorMap<const Tensor2F>(arrivals, n_cls, n_env)),
      departures(Eigen::TensorMap<const Tensor2F>(departures, n_cls, n_env)),
      env_trans_mats(([this, env_trans_mats]() {
        {
          Tensor3F res = Eigen::TensorMap<const Tensor3F>(
              env_trans_mats, n_cls, this->n_env, this->n_env);
          for (index_type i = 0; i < n_cls; ++i) {
            for (index_type j = 0; j < this->n_env; ++j) {
              res(i, j, j) = 0;
            }
          }
          return res;
        }
      })()),
      normalized_c(([this, normalized_c]() {
        {
          if (normalized_c) return normalized_c.value();
          return to_scalar(this->arrivals.maximum(std::array{1}).sum() +
                           this->departures.maximum() +
                           this->env_trans_mats.sum(std::array{2})
                               .maximum(std::array{1})
                               .sum());
        }
      })()),
      rewards(([this, &reward_func]() {
        VectorAF res(states.sys.n);
        for (index_type i = 0; i < states.sys.n; ++i) {
          res[i] = reward_func(this->costs, states.sys.a.row(i), n_cls);
        }
        return res;
      })()),
      env_state_accessible(([this, env_trans_mats]() {
        VectorAB res(states.env.n);
        Eigen::Tensor<float_type, 2, storage_order> sums =
            Eigen::TensorMap<const Tensor3F>(env_trans_mats, n_cls, this->n_env,
                                             this->n_env)
                .sum(std::array{2});
        for (index_type i = 0; i < states.env.n; ++i) {
          index_type j;
          for (j = 0; j < this->n_env; ++j) {
            if (!sums(j, states.env.a(i, j))) {
              break;
            }
          }
          res[i] = (j == this->n_env);
        }
        return res;
      })()),
      trans_probs(([this]() {
        SpMats res(states.sys.n);
        for (index_type i = 0; i < states.sys.n; ++i) {
          const auto& s_i = states.sys.a.row(i);
          auto prob_i = sp_mat_type(n_cls, states.sys.n);
          prob_i.reserve(VectorAI::Constant(
              n_cls, n_cls + 1 + n_cls * (this->n_env - 1) + 1));
          for (index_type a = 0; a < n_cls; ++a) {
            if ((states.to_cls(i) && !s_i(a)) || (!states.to_cls(i) && a))
              continue;
            float_type dummy_prob = this->normalized_c;
            for (index_type j = 0; j < states.sys.n; ++j) {
              const auto& s_j = states.sys.a.row(j);
              const auto next_to = states.next_to_up(s_i, s_j, a);
              if (next_to) {
                float_type prob;
                const auto n = next_to.value();
                if (n >= n_cls) {
                  prob = this->env_trans_mats(n - n_cls, s_i(n), s_j[n]);
                } else if (s_i[n] < s_j[n]) {
                  prob = this->arrivals(n, s_i(n + n_cls));
                } else {
                  prob = this->departures(n, s_i(n + n_cls));
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
          prob_i /= this->normalized_c;
          res[i] = std::move(prob_i);
        }
        return res;
      })()),
      action_masks(([this]() {
        ArrayB masks(states.cls.n, n_cls);
        for (index_type i = 0; i < states.cls.n; ++i) {
          const auto& prob_i = trans_probs[states.to_sys(i, 0)];
          masks.row(i) = (prob_i * MatrixF::Ones(states.cls.n, 1))
                             .template cast<bool>()
                             .transpose();
        }
        return masks;
      })()),
      state_cls_trans_probs(([this]() {
        SpMats res(states.cls.n * n_cls);
        std::vector<std::vector<Eigen::Triplet<float_type>>> triplets(
            states.cls.n * n_cls);
        for (index_type i = 0; i < states.sys.n; ++i) {
          const auto& prob = trans_probs[i];
          const auto i_cls = states.to_cls[i];
          const auto i_env = states.to_env[i];
          for (index_type a = 0; a < n_cls; ++a) {
            for (SpMatIt it(prob, a); it; ++it) {
              triplets[states.to_cls_action(states.to_cls[it.index()], a)]
                  .emplace_back(i_cls, i_env, it.value());
            }
          }
        }
        for (index_type i = 0; i < states.cls.n; ++i) {
          for (index_type a = 0; a < n_cls; ++a) {
            const auto j = states.to_cls_action(i, a);
            res[j] = SpMat(states.cls.n, states.env.n);
            res[j].setFromTriplets(triplets[j].begin(), triplets[j].end());
            res[j].makeCompressed();
          }
        }
        return res;
      })()),
      env_trans_probs(([this]() {
        auto res = SpMat(states.env.n, states.env.n);
        res.reserve(
            VectorAI::Constant(states.env.n, n_cls * (this->n_env - 1) + 1));
        for (index_type i = 0; i < states.env.n; ++i) {
          if (!env_state_accessible[i]) {
            continue;
          }
          const auto& prob_i = trans_probs[states.to_sys(0, i)];
          for (sp_mat_it it(prob_i, 0); it; ++it) {
            const auto j = it.index();
            const auto j_env = states.to_env[j];
            const auto prob = it.value();
            res.coeffRef(i, j_env) += prob;
          }
        }
        res.makeCompressed();
        return res;
      })()),
      cls_dims(states.cls_dims.cast<size_t>()),
      cls_action_dims((VectorAS(n_cls + 1) << cls_dims, n_cls).finished()),
      trans_dists_(init_trans_dists()),
      action_dists_(init_action_dists()),
      state_(),
      n_cls_visit_(states.cls.n, n_cls),
      n_cls_trans_(states.cls.n),
      cls_cum_rewards_(states.cls.n),
      i_cls_trans_probs_(states.cls.n),
      i_cls_rewards_(states.cls.n),
      q_(states.cls.n, n_cls),
      q_policy_(states.cls.n),
      v_(states.cls.n),
      v_policy_(states.cls.n),
      t_env_probs_(states.env.n),
      t_cls_trans_probs_(states.cls.n),
      t_cls_rewards_(states.cls.n) {
  n_cls_visit_.setZero();
  cls_cum_rewards_.setZero();
  i_cls_rewards_.setZero();
  q_.setZero();
  q_policy_.setZero();
  v_.setZero();
  v_policy_.setZero();
  t_env_probs_.setZero();
  t_cls_rewards_.setZero();
}

dists_type System::init_trans_dists() const {
  dists_type res(states.sys.n * n_cls);
  for (index_type i = 0; i < states.sys.n; ++i) {
    const auto& prob_i = trans_probs[i];
    for (index_type a = 0; a < n_cls; ++a) {
      const auto start = prob_i.outerIndexPtr()[a];
      const auto end = prob_i.outerIndexPtr()[a + 1];
      if (start != end) {
        res[states.to_sys_action(i, a)] =
            std::discrete_distribution<index_type>(prob_i.valuePtr() + start,
                                                   prob_i.valuePtr() + end);
      }
    }
  }
  return res;
}

dists_type System::init_action_dists() const {
  dists_type res(states.sys.n);
  for (index_type i = 0; i < states.cls.n; ++i) {
    res[i] = std::discrete_distribution<index_type>(action_masks.row(i).begin(),
                                                    action_masks.row(i).end());
  }
  return res;
}

index_type System::step(index_type current_state, index_type action) {
  auto next_state_idx =
      trans_dists_[states.to_sys_action(current_state, action)](rng);
  return trans_probs[current_state]
      .innerIndexPtr()[trans_probs[current_state].outerIndexPtr()[action] +
                       next_state_idx];
}

void System::step(index_type action) {
  auto next_state_idx = trans_dists_[states.to_sys_action(state_, action)](rng);
  state_ = trans_probs[state_]
               .innerIndexPtr()[trans_probs[state_].outerIndexPtr()[action] +
                                next_state_idx];
  state_ = step(state_, action);
}

void System::reset(uint64_t seed) {
  rng = decltype(rng)(seed);
  state_ = 0;
}

void System::reset_i() {
  n_cls_visit_.setZero();
  for (size_t i = 0; i < states.cls.n; ++i) {
    n_cls_trans_[i] = SpMatU64(n_cls, states.cls.n);
    n_cls_trans_[i].reserve(VectorAI::Constant(n_cls, n_cls + 1 + 1));
  }
  reset_cls_trans_probs(i_cls_trans_probs_);
  cls_cum_rewards_.setZero();
  i_cls_rewards_.setZero();
}

void System::reset_cls_trans_probs(SpMats& mats) const {
  for (size_t i = 0; i < states.cls.n; ++i) {
    mats[i] = SpMat(n_cls, states.cls.n);
    mats[i].reserve(VectorAI::Constant(n_cls, n_cls + 1 + 1));
  }
}

void System::reset_q() {
  q_.setZero();
  for (index_type i = 0; i < states.cls.n; ++i) {
    for (index_type a = 0; a < n_cls; ++a) {
      if (!states.cls.a(i, a)) {
        q_(i, a) = -inf_v;
      }
    }
  }
  q_(0, 0) = 0;
  q_policy_.setZero();
}

void System::reset_v() {
  v_.setZero();
  v_policy_.setZero();
}

void System::reset_t() {
  t_env_probs_ = env_state_accessible.select(
      VectorAF::Constant(states.env.n, static_cast<float_type>(1) /
                                           env_state_accessible.count()),
      VectorAF::Constant(states.env.n, 0));
  reset_cls_trans_probs(t_cls_trans_probs_);
  t_cls_rewards_.setZero();
}

template <bool log_i_t, bool log_qs_t>
void System::train_q_impl(float_type gamma, float_type greedy_eps, uint64_t ls,
                          uint64_t seed) {
  reset(seed);
  reset_i();
  reset_q();

  static constexpr uint64_t qs_limit = 1000;
  const uint64_t qs_step = std::ceil(ls / static_cast<float_type>(qs_limit));

  if constexpr (log_qs_t) {
    qs_ = Tensor3F(std::min(ls, qs_limit) + (ls % qs_step == 0 && qs_step != 1),
                   states.cls.n, n_cls);
  }

  for (index_type i = 0; i < ls; ++i) {
    const auto cls_state = states.to_cls[state_];

    index_type a;
    if (q_greedy_dis_(rng) < greedy_eps) {
      a = action_dists_[cls_state](rng);
    } else {
      q_.row(cls_state).maxCoeff(&a);
    }

    const auto reward = rewards[state_];

    step(a);
    ++n_cls_visit_(cls_state, a);

    const auto next_cls_state = states.to_cls[state_];

    const auto next_q = q_.row(next_cls_state).maxCoeff();
    q_(cls_state, a) +=
        (static_cast<float_type>(1) / n_cls_visit_(cls_state, a)) *
        (reward + gamma * next_q - q_(cls_state, a));

    if constexpr (log_i_t) {
      ++n_cls_trans_[cls_state].coeffRef(a, next_cls_state);
      cls_cum_rewards_[cls_state] += reward;
    }

    if constexpr (log_qs_t) {
      using qs_element_type = Eigen::TensorMap<Tensor2F>;
      if (!(i % qs_step)) {
        qs_.chip<0>(i / qs_step) =
            qs_element_type(q_.data(), states.cls.n, n_cls);
      } else if (i == ls - 1) {
        qs_.chip<0>(qs_.dimension(0) - 1) =
            qs_element_type(q_.data(), states.cls.n, n_cls);
      }
    }
  }

  if constexpr (log_i_t) {
    for (index_type i = 0; i < states.cls.n; ++i) {
      for (index_type a = 0; a < n_cls; ++a) {
        i_cls_trans_probs_[i].row(a) =
            (n_cls_trans_[i].row(a) / n_cls_visit_(i, a))
                .template cast<float_type>();
      }
    }

    i_cls_rewards_ = cls_cum_rewards_.transpose() /
                     n_cls_visit_.rowwise().sum().template cast<float_type>();
  }

  for (index_type i = 0; i < states.cls.n; ++i) {
    n_cls_trans_[i].makeCompressed();
    i_cls_trans_probs_[i].makeCompressed();
  }

  for (index_type i = 0; i < states.cls.n; ++i) {
    q_.row(i).maxCoeff(&q_policy_(i));
  }
}

void System::train_q(float_type gamma, float_type greedy_eps, uint64_t ls,
                     uint64_t seed) {
  train_q_impl<false, false>(gamma, greedy_eps, ls, seed);
}

void System::train_q_i(float_type gamma, float_type greedy_eps, uint64_t ls,
                       uint64_t seed) {
  train_q_impl<true, false>(gamma, greedy_eps, ls, seed);
}

void System::train_qs(float_type gamma, float_type greedy_eps, uint64_t ls,
                      uint64_t seed) {
  train_q_impl<false, true>(gamma, greedy_eps, ls, seed);
}

void System::train_q_full(float_type gamma, float_type greedy_eps, uint64_t ls,
                          uint64_t seed) {
  train_q_impl<true, true>(gamma, greedy_eps, ls, seed);
}

void System::train_q_off(float_type gamma, uint64_t ls, uint64_t seed) {
  reset(seed);
  reset_i();
  reset_q();

  for (uint64_t i = 0; i < ls; ++i) {
    for (index_type s = 0; s < states.cls.n; ++s) {
      for (index_type a = 0; a < n_cls; ++a) {
        if (!action_masks(s, a)) continue;

        const auto sys_state = states.to_sys(s, 0);
        const auto reward = rewards[sys_state];
        const auto next_cls_state = states.to_cls[step(sys_state, a)];
        const auto next_q = q_.row(next_cls_state).maxCoeff();

        ++n_cls_visit_(s, a);
        q_(s, a) += (static_cast<float_type>(1) / n_cls_visit_(s, a)) *
                    (reward + gamma * next_q - q_(s, a));
      }
    }
  }

  for (index_type i = 0; i < states.cls.n; ++i) {
    q_.row(i).maxCoeff(&q_policy_(i));
  }
}

template <bool update_policy_t>
void System::update_v(float_type gamma, index_type i, const MatrixF& old_v) {
  index_type max_a;
  float_type max_v;
  const auto& prob = t_cls_trans_probs_[i];

  const VectorAF new_a = action_masks.row(i).select(
      old_v * prob.transpose(), VectorAF::Constant(n_cls, -inf_v));
  const VectorAF new_v = t_cls_rewards_[i] + gamma * new_a;

  if ((new_v == VectorAF::Constant(n_cls, new_v(0))).all()) {
    max_v = new_v(0);
    max_a = n_cls;
  } else {
    max_v = new_v.maxCoeff(&max_a);
  }

  if constexpr (update_policy_t) {
    v_policy_[i] = max_a;
  } else {
    v_[i] = max_v;
  }
}

void System::train_v(float_type gamma) {
  reset_v();
  train_t();

  const auto iota_cls = ([this]() {
    std::vector<index_type> res(states.cls.n);
    std::iota(res.begin(), res.end(), 0);
    return res;
  })();

  VectorMF old_v_ = VectorMF::Constant(states.cls.n, inf_v);
  while ((old_v_ - v_).norm() >= 2 * eps_v) {
    old_v_ = v_;
    std::for_each(std::execution::par_unseq, iota_cls.begin(), iota_cls.end(),
                  [gamma, this, &old_v_](index_type i) {
                    update_v<false>(gamma, i, old_v_);
                  });
  }
  std::for_each(std::execution::par_unseq, iota_cls.begin(), iota_cls.end(),
                [gamma, this](index_type i) { update_v<true>(gamma, i, v_); });
}

void System::train_t() {
  reset_t();

  VectorMF old_t_env_probs_ = VectorMF::Constant(states.env.n, inf_v);
  while ((old_t_env_probs_ - t_env_probs_).norm() >= 2 * eps_v) {
    old_t_env_probs_ = t_env_probs_;
    t_env_probs_ *= env_trans_probs;
  }

  for (index_type a = 0; a < n_cls; ++a) {
    for (index_type j = 0; j < states.cls.n; ++j) {
      const auto& prob = state_cls_trans_probs[states.to_cls_action(j, a)];
      for (index_type i = 0; i < states.cls.n; ++i) {
        const auto& row = prob.row(i);
        if (row.nonZeros()) {
          const auto p_i_a_j = row.dot(t_env_probs_);
          if (p_i_a_j > eps_v) {
            t_cls_trans_probs_[i].insert(a, j) = p_i_a_j;
          }
        }
      }
    }
  }

  for (index_type i = 0; i < states.cls.n; ++i) {
    t_cls_trans_probs_[i].makeCompressed();
  }

  for (index_type i = 0; i < states.sys.n; ++i) {
    const auto i_cls = states.to_cls[i];
    const auto i_env = states.to_env[i];
    t_cls_rewards_[i_cls] += rewards[i] * t_env_probs_[i_env];
  }
}

bool System::operator==(const System& other) const {
  return (n_env == other.n_env) && (limits == other.limits).all() &&
         (states == other.states) && (n_cls == other.n_cls) &&
         to_scalar((costs == other.costs).all()) &&
         to_scalar((arrivals == other.arrivals).all()) &&
         to_scalar((departures == other.departures).all()) &&
         to_scalar((env_trans_mats == other.env_trans_mats).all()) &&
         (normalized_c == other.normalized_c) &&
         (env_state_accessible == other.env_state_accessible).all() &&
         sps_equal(trans_probs, other.trans_probs) &&
         (action_masks == other.action_masks).all() &&
         (rewards == other.rewards).all() &&
         sps_equal(state_cls_trans_probs, other.state_cls_trans_probs) &&
         (env_trans_probs == other.env_trans_probs) &&
         (cls_dims == other.cls_dims).all() &&
         (cls_action_dims == other.cls_action_dims).all() &&
         (trans_dists_ == other.trans_dists_) &&
         (action_dists_ == other.action_dists_) && (state_ == other.state_) &&
         (n_cls_visit_ == other.n_cls_visit_).all() &&
         sps_equal(n_cls_trans_, other.n_cls_trans_) &&
         (cls_cum_rewards_ == other.cls_cum_rewards_).all() &&
         sps_equal(i_cls_trans_probs_, other.i_cls_trans_probs_) &&
         (i_cls_rewards_ == other.i_cls_rewards_).all() &&
         (q_ == other.q_).all() && to_scalar((qs_ == other.qs_).all()) &&
         (q_policy_ == other.q_policy_).all() &&
         (q_greedy_dis_ == other.q_greedy_dis_) && (v_ == other.v_) &&
         (v_policy_ == other.v_policy_).all() &&
         (t_env_probs_ == other.t_env_probs_) &&
         sps_equal(t_cls_trans_probs_, other.t_cls_trans_probs_) &&
         (t_cls_rewards_ == other.t_cls_rewards_).all();
}

System::System(index_type&& n_env, VectorAI&& limits, State&& states,
               index_type&& n_cls, Tensor2F&& costs, Tensor2F&& arrivals,
               Tensor2F&& departures, Tensor3F&& env_trans_mats,
               float_type&& normalized_c, VectorAF&& rewards,
               VectorAB&& env_state_accessible, SpMats&& trans_probs,
               ArrayB&& action_masks, SpMats&& state_cls_trans_probs,
               SpMat&& env_trans_probs, VectorAS&& cls_dims,
               VectorAS&& cls_action_dims, index_type&& state_,
               ArrayU64&& n_cls_visit_, SpMatU64s&& n_cls_trans_,
               VectorAF&& cls_cum_rewards_, SpMats&& i_cls_trans_probs_,
               VectorAF&& i_cls_rewards_, ArrayF&& q_, Tensor3F&& qs_,
               VectorAI&& q_policy_, VectorMF&& v_, VectorAI&& v_policy_,
               VectorMF&& t_env_probs_, SpMats&& t_cls_trans_probs_,
               VectorAF&& t_cls_rewards_)
    : n_env(n_env),
      limits(limits),
      states(states),
      n_cls(n_cls),
      costs(costs),
      arrivals(arrivals),
      departures(departures),
      env_trans_mats(env_trans_mats),
      normalized_c(normalized_c),
      rewards(rewards),
      env_state_accessible(env_state_accessible),
      trans_probs(trans_probs),
      action_masks(action_masks),
      state_cls_trans_probs(state_cls_trans_probs),
      env_trans_probs(env_trans_probs),
      cls_dims(cls_dims),
      cls_action_dims(cls_action_dims),
      trans_dists_(init_trans_dists()),
      action_dists_(init_action_dists()),
      state_(state_),
      n_cls_visit_(n_cls_visit_),
      n_cls_trans_(n_cls_trans_),
      cls_cum_rewards_(cls_cum_rewards_),
      i_cls_trans_probs_(i_cls_trans_probs_),
      i_cls_rewards_(i_cls_rewards_),
      q_(q_),
      qs_(qs_),
      q_policy_(q_policy_),
      v_(v_),
      v_policy_(v_policy_),
      t_env_probs_(t_env_probs_),
      t_cls_trans_probs_(t_cls_trans_probs_),
      t_cls_rewards_(t_cls_rewards_) {}

void System::save(cereal::BinaryOutputArchive& ar) const {
  ar(n_env, limits, states, n_cls, costs, arrivals, departures, env_trans_mats,
     normalized_c, rewards, env_state_accessible, trans_probs, action_masks,
     state_cls_trans_probs, env_trans_probs, cls_dims, cls_action_dims, state_,
     n_cls_visit_, n_cls_trans_, cls_cum_rewards_, i_cls_trans_probs_,
     i_cls_rewards_, q_, qs_, q_policy_, v_, v_policy_, t_env_probs_,
     t_cls_trans_probs_, t_cls_rewards_);
}

void System::load(cereal::BinaryInputArchive& ar) {
  index_type n_env;
  VectorAI limits;
  State states;
  index_type n_cls;

  Tensor2F costs;
  Tensor2F arrivals;
  Tensor2F departures;
  Tensor3F env_trans_mats;
  float_type normalized_c;

  VectorAF rewards;

  VectorAB env_state_accessible;

  SpMats trans_probs;

  ArrayB action_masks;

  SpMats state_cls_trans_probs;

  SpMat env_trans_probs;

  VectorAS cls_dims;
  VectorAS cls_action_dims;

  index_type state_;
  ArrayU64 n_cls_visit_;
  SpMatU64s n_cls_trans_;
  VectorAF cls_cum_rewards_;

  SpMats i_cls_trans_probs_;
  VectorAF i_cls_rewards_;

  ArrayF q_;
  Tensor3F qs_;
  VectorAI q_policy_;

  VectorMF v_;
  VectorAI v_policy_;

  VectorMF t_env_probs_;
  SpMats t_cls_trans_probs_;
  VectorAF t_cls_rewards_;

  ar(n_env, limits, states, n_cls, costs, arrivals, departures, env_trans_mats,
     normalized_c, rewards, env_state_accessible, trans_probs, action_masks,
     state_cls_trans_probs, env_trans_probs, cls_dims, cls_action_dims, state_,
     n_cls_visit_, n_cls_trans_, cls_cum_rewards_, i_cls_trans_probs_,
     i_cls_rewards_, q_, qs_, q_policy_, v_, v_policy_, t_env_probs_,
     t_cls_trans_probs_, t_cls_rewards_);

  new (this) System(
      std::move(n_env), std::move(limits), std::move(states), std::move(n_cls),
      std::move(costs), std::move(arrivals), std::move(departures),
      std::move(env_trans_mats), std::move(normalized_c), std::move(rewards),
      std::move(env_state_accessible), std::move(trans_probs),
      std::move(action_masks), std::move(state_cls_trans_probs),
      std::move(env_trans_probs), std::move(cls_dims),
      std::move(cls_action_dims), std::move(state_), std::move(n_cls_visit_),
      std::move(n_cls_trans_), std::move(cls_cum_rewards_),
      std::move(i_cls_trans_probs_), std::move(i_cls_rewards_), std::move(q_),
      std::move(qs_), std::move(q_policy_), std::move(v_), std::move(v_policy_),
      std::move(t_env_probs_), std::move(t_cls_trans_probs_),
      std::move(t_cls_rewards_));
}

void System::to_stream(std::ostream& os) const {
  cereal::BinaryOutputArchive ar(os);
  ar(*this);
}

System System::from_stream(std::istream& is) {
  System sys;
  cereal::BinaryInputArchive ar(is);
  ar(sys);
  return sys;
}

void System::to_file(const std::string& path) const {
  std::ofstream os(path, std::ios::binary);
  to_stream(os);
}

System System::from_file(const std::string& path) {
  std::ifstream is(path, std::ios::binary);
  return from_stream(is);
}

std::string System::to_str() const {
  std::stringstream os;
  to_stream(os);
  return os.str();
}

System System::from_str(const std::string& str) {
  std::stringstream is(str);
  return from_stream(is);
}

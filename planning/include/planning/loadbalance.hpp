#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "planning/config.hpp"
#include "planning/state.hpp"
#include "unsupported/Eigen/CXX11/Tensor"

class LoadBalance {
 public:
  LoadBalance(const index_type n_env, const VectorAI& limits,
              const float_type* costs, const float_type* arrivals,
              const float_type* departures, const float_type* env_trans_mats,
              const reward_func_type& reward_func,
              const std::optional<float_type>& normalized_c = std::nullopt);
  LoadBalance() = default;

  const index_type n_env{};
  const VectorAI limits;
  const State states{};
  const index_type n_cls{};

  const Tensor2F costs;
  const Tensor2F arrivals;
  const Tensor2F departures;
  const Tensor3F env_trans_mats;
  const float_type normalized_c = 0;

  const VectorAF rewards;

  const VectorAB env_state_accessible;

  const SpMats trans_probs{};

  const ArrayB action_masks;

  // P(S'|S, E, a) by ((S', a), S, E)
  const SpMats state_cls_trans_probs{};

  // P(E'|E) by (E, E')
  const SpMat env_trans_probs;

  const VectorAS cls_dims;
  const VectorAS cls_action_dims;

  index_type step(index_type current_state, index_type action);
  void step(index_type action);
  void reset(u_int64_t seed);
  void reset_i();
  void reset_q();
  void reset_v();
  void reset_t();

  void train_q(float_type gamma, float_type greedy_eps, uint64_t ls,
               uint64_t seed);
  void train_q_i(float_type gamma, float_type greedy_eps, uint64_t ls,
                 uint64_t seed);
  void train_qs(float_type gamma, float_type greedy_eps, uint64_t ls,
                uint64_t seed);
  void train_q_full(float_type gamma, float_type greedy_eps, uint64_t ls,
                    uint64_t seed);

  void train_q_off(float_type gamma, uint64_t ls, uint64_t seed);

  void train_v(float_type gamma);

  void train_t();

  const auto& n_cls_visit() const { return n_cls_visit_; }

  const auto& i_cls_trans_probs() const { return i_cls_trans_probs_; }
  const auto& i_cls_rewards() const { return i_cls_rewards_; }

  const auto& q() const { return q_; }
  const auto& q_policy() const { return q_policy_; }
  const auto& qs() const { return qs_; }

  const auto& v() const { return v_; }
  const auto& v_policy() const { return v_policy_; }

  const auto& t_env_probs() const { return t_env_probs_; }
  const auto& t_cls_trans_probs() const { return t_cls_trans_probs_; }
  const auto& t_cls_rewards() const { return t_cls_rewards_; }

  bool operator==(const LoadBalance& other) const;

  void to_stream(std::ostream& os) const;
  static LoadBalance from_stream(std::istream& is);
  void to_file(const std::string& path) const;
  static LoadBalance from_file(const std::string& path);
  std::string to_str() const;
  static LoadBalance from_str(const std::string& str);

 private:
  dists_type trans_dists_;
  dists_type action_dists_;

  index_type state_;
  ArrayU64 n_cls_visit_;
  SpMatU64s n_cls_trans_;
  VectorAF cls_cum_rewards_;

  SpMats i_cls_trans_probs_;
  VectorAF i_cls_rewards_;

  ArrayF q_;
  Tensor3F qs_;
  VectorAI q_policy_;
  std::uniform_real_distribution<float_type> q_greedy_dis_;

  VectorMF v_;
  VectorAI v_policy_;

  VectorMF t_env_probs_;
  SpMats t_cls_trans_probs_;
  VectorAF t_cls_rewards_;

  dists_type init_trans_dists() const;
  dists_type init_action_dists() const;

  void reset_cls_trans_probs(SpMats& mats) const;

  template <bool log_i_t, bool log_qs_t>
  void train_q_impl(float_type gamma, float_type greedy_eps, uint64_t ls,
                    uint64_t seed);

  template <bool update_policy_t>
  void update_v(float_type gamma, index_type i, const MatrixF& old_v);

  LoadBalance(index_type&& n_env, VectorAI&& limits, State&& states,
              index_type&& n_cls, Tensor2F&& costs, Tensor2F&& arrivals,
              Tensor2F&& departures, Tensor3F&& env_trans_mats,
              float_type&& normalize_c, VectorAF&& rewards,
              VectorAB&& env_state_accessible, SpMats&& trans_probs,
              ArrayB&& action_masks, SpMats&& state_cls_trans_probs,
              SpMat&& env_trans_probs, VectorAS&& cls_dims,
              VectorAS&& cls_action_dims, index_type&& state_,
              ArrayU64&& n_cls_visit_, SpMatU64s&& n_cls_trans_,
              VectorAF&& cls_cum_rewards_, SpMats&& i_cls_trans_probs_,
              VectorAF&& i_cls_rewards_, ArrayF&& q_, Tensor3F&& qs_,
              VectorAI&& q_policy_, VectorMF&& v_, VectorAI&& v_policy_,
              VectorMF&& t_env_probs_, SpMats&& t_cls_trans_probs_,
              VectorAF&& t_cls_rewards_);

  friend class cereal::access;
  void save(cereal::BinaryOutputArchive& ar) const;
  void load(cereal::BinaryInputArchive& ar);
};

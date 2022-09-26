#pragma once

#include "Eigen/Dense"
#include "cereal/access.hpp"
#include "cereal/cereal.hpp"
#include "cereal/types/vector.hpp"
#include "planning/config.hpp"
#include "planning/serialize.hpp"

class CartesianProduct {
 public:
  explicit CartesianProduct(const VectorAI& lens);
  CartesianProduct() = default;

  const VectorAI lens;

  const index_type n{};
  const index_type d{};

  const ArrayI a;

  bool operator==(const CartesianProduct& other) const;

 private:
  CartesianProduct(VectorAI&& lens, index_type&& n, index_type&& d, ArrayI&& a);

  friend class cereal::access;
  void save(cereal::BinaryOutputArchive& ar) const;
  void load(cereal::BinaryInputArchive& ar);
};

class State {
 public:
  State(index_type n_env, const VectorAI& limits);
  State() = default;

  const index_type n_env{};
  const index_type n_cls{};

  const VectorAI cls_dims;

  const CartesianProduct cls{};
  const CartesianProduct env{};
  const CartesianProduct sys{};

  const VectorAI lin_spaced;

  const VectorAI to_cls;
  const VectorAI to_env;
  const ArrayI to_sys;

  index_type to_cls_action(index_type i, index_type a) const;

  index_type to_sys_action(index_type i, index_type a) const;

  std::optional<index_type> next_to(const ConstRowXprAI& s1,
                                    const ConstRowXprAI& s2,
                                    index_type a) const;

  bool operator==(const State& other) const;

 private:
  State(index_type&& n_env, index_type&& n_cls, VectorAI&& cls_dims,
        CartesianProduct&& cls, CartesianProduct&& env, CartesianProduct&& sys,
        VectorAI&& lin_spaced, VectorAI&& to_cls, VectorAI&& to_env,
        ArrayI&& to_sys);

  friend class cereal::access;
  void save(cereal::BinaryOutputArchive& ar) const;
  void load(cereal::BinaryInputArchive& ar);
};

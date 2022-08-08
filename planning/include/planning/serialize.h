#pragma once

#include <memory>
#include <random>

#include "Eigen/Dense"
#include "cereal/archives/binary.hpp"
#include "cereal/cereal.hpp"
#include "cereal/types/array.hpp"
#include "cereal/types/utility.hpp"
#include "planning/config.h"
#include "planning/state.h"
#include "tsl/robin_map.h"

namespace cereal {

template <class Archive, class T>
concept is_output_binary_serializable =
    traits::is_output_serializable<BinaryData<T>, Archive>::value;

template <class Archive, class T>
concept is_input_binary_serializable =
    traits::is_input_serializable<BinaryData<T>, Archive>::value;

template <class Archive, class Derived>
requires is_output_binary_serializable<Archive, typename Derived::Scalar>
void save(Archive &ar, Eigen::PlainObjectBase<Derived> const &m) {
  using DenseType = Eigen::PlainObjectBase<Derived>;

  if (DenseType::RowsAtCompileTime == Eigen::Dynamic) ar(m.rows());
  if (DenseType::ColsAtCompileTime == Eigen::Dynamic) ar(m.cols());
  ar(binary_data(m.data(), m.size() * sizeof(typename Derived::Scalar)));
}

template <class Archive, class Derived>
requires is_input_binary_serializable<Archive, typename Derived::Scalar>
void load(Archive &ar, Eigen::PlainObjectBase<Derived> &m) {
  using DenseType = Eigen::PlainObjectBase<Derived>;
  using Index = Derived::Index;

  Index rows = DenseType::RowsAtCompileTime;
  Index cols = DenseType::ColsAtCompileTime;
  if (rows == Eigen::Dynamic) ar(rows);
  if (cols == Eigen::Dynamic) ar(cols);
  m.resize(rows, cols);
  ar(binary_data(m.data(), rows * cols * sizeof(typename Derived::Scalar)));
}

template <class Archive, typename Derived>
requires is_output_binary_serializable<Archive, typename Derived::Scalar>
void save(Archive &ar, const Eigen::SparseCompressedBase<Derived> &m) {
  using SparseType = Eigen::SparseCompressedBase<Derived>;
  using StorageIndex = typename SparseType::StorageIndex;

  if (!m.isCompressed()) return;

  ar(m.rows(), m.cols(), m.nonZeros(), m.outerSize());

  ar(binary_data(m.outerIndexPtr(),
                 (m.outerSize() + 1) * sizeof(StorageIndex)));
  ar(binary_data(m.innerIndexPtr(), m.nonZeros() * sizeof(StorageIndex)));
  ar(binary_data(m.valuePtr(),
                 m.nonZeros() * sizeof(typename Derived::Scalar)));
}

template <class Archive, typename Derived>
requires is_input_binary_serializable<Archive, typename Derived::Scalar>
void load(Archive &ar, Eigen::SparseCompressedBase<Derived> &m) {
  using SparseType = Eigen::SparseCompressedBase<Derived>;
  using StorageIndex = typename SparseType::StorageIndex;
  using Index = Derived::Index;

  Index rows;
  Index cols;
  Index nnz;
  Index outer_size;
  ar(rows, cols, nnz, outer_size);

  auto outer = std::make_unique_for_overwrite<StorageIndex[]>(outer_size + 1);
  auto inner = std::make_unique_for_overwrite<StorageIndex[]>(nnz);
  auto value = std::make_unique_for_overwrite<typename Derived::Scalar[]>(nnz);

  ar(binary_data(outer.get(), (outer_size + 1) * sizeof(StorageIndex)));
  ar(binary_data(inner.get(), nnz * sizeof(StorageIndex)));
  ar(binary_data(value.get(), nnz * sizeof(typename Derived::Scalar)));

  m.derived().resize(rows, cols);
  m.derived().resizeNonZeros(nnz);
  std::copy(outer.get(), outer.get() + m.outerSize() + 1, m.outerIndexPtr());
  std::copy(inner.get(), inner.get() + nnz, m.derived().data().indexPtr());
  std::copy(value.get(), value.get() + nnz, m.derived().data().valuePtr());
}

template <class Archive, typename Derived>
requires is_output_binary_serializable<Archive, typename Derived::Scalar>
void save(Archive &ar, const Eigen::TensorBase<Derived> &t) {
  using Index = typename Derived::Index;

  const auto &d = *static_cast<const Derived *>(&t);

  ar(d.size());

  using DimBaseType = Eigen::array<Index, Derived::NumIndices>;
  if constexpr (std::derived_from<typename Derived::Dimensions, DimBaseType>) {
    ar(DimBaseType(d.dimensions()));
  }

  ar(binary_data(d.data(), d.size() * sizeof(typename Derived::Scalar)));
}

template <class Archive, typename Derived>
requires is_input_binary_serializable<Archive, typename Derived::Scalar>
void load(Archive &ar, Eigen::TensorBase<Derived> &t) {
  using Index = typename Derived::Index;

  auto &d = *static_cast<Derived *>(&t);

  Index size;
  ar(size);

  using DimBaseType = Eigen::array<Index, Derived::NumIndices>;
  if constexpr (std::derived_from<typename Derived::Dimensions, DimBaseType>) {
    DimBaseType dims;
    ar(dims);
    d.resize(dims);
  }

  ar(binary_data(d.data(), size * sizeof(typename Derived::Scalar)));
}

template <class Archive, class Key, class T>
struct specialize<Archive, tsl::robin_map<Key, T>,
                  cereal::specialization::non_member_load_save> {};

template <class Archive, class Key, class T>
void save(Archive &ar, const tsl::robin_map<Key, T> &map) {
  auto serializer = [&ar](const auto &v) { ar &v; };
  map.serialize(serializer);
}

template <class Archive, class Key, class T>
void load(Archive &ar, tsl::robin_map<Key, T> &map) {
  auto deserializer = [&ar]<typename U>() {
    U u;
    ar &u;
    return u;
  };
  map = tsl::robin_map<Key, T>::deserialize(deserializer);
}

}  // namespace cereal

#pragma once

#include <dbg.h>

#include <functional>
#include <iostream>

#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "planning/xoshiro.hpp"
#include "unsupported/Eigen/CXX11/Tensor"

static constexpr auto storage_order = Eigen::RowMajor;

using index_type = Eigen::Index;
using MatrixI =
    Eigen::Matrix<index_type, Eigen::Dynamic, Eigen::Dynamic, storage_order>;
using ArrayI =
    Eigen::Array<index_type, Eigen::Dynamic, Eigen::Dynamic, storage_order>;
using VectorMI = Eigen::Matrix<index_type, 1, Eigen::Dynamic, storage_order>;
using VectorAI = Eigen::Array<index_type, 1, Eigen::Dynamic, storage_order>;
using ConstRowXprMI = typename MatrixI::ConstRowXpr;
using ConstRowXprAI = typename ArrayI::ConstRowXpr;

using float_type = double;
using MatrixF =
    Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, storage_order>;
using ArrayF =
    Eigen::Array<float_type, Eigen::Dynamic, Eigen::Dynamic, storage_order>;
using VectorMF = Eigen::Matrix<float_type, 1, Eigen::Dynamic, storage_order>;
using VectorAF = Eigen::Array<float_type, 1, Eigen::Dynamic, storage_order>;

using MatrixB =
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, storage_order>;
using ArrayB =
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, storage_order>;
using VectorMB = Eigen::Matrix<bool, 1, Eigen::Dynamic, storage_order>;
using VectorAB = Eigen::Array<bool, 1, Eigen::Dynamic, storage_order>;

using VectorAS = Eigen::Array<size_t, 1, Eigen::Dynamic, storage_order>;

using MatrixU64 =
    Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, storage_order>;
using ArrayU64 =
    Eigen::Array<uint64_t, Eigen::Dynamic, Eigen::Dynamic, storage_order>;
using VectorMU64 = Eigen::Matrix<uint64_t, 1, Eigen::Dynamic, storage_order>;
using VectorAU64 = Eigen::Array<uint64_t, 1, Eigen::Dynamic, storage_order>;

using SpMat = Eigen::SparseMatrix<float_type, storage_order, index_type>;
using SpMatIt = typename SpMat::InnerIterator;

using sp_mat_type = Eigen::SparseMatrix<float_type, storage_order, index_type>;
using sp_mat_it = typename sp_mat_type::InnerIterator;

using Tensor2F = Eigen::Tensor<float_type, 2, storage_order>;
using Tensor3F = Eigen::Tensor<float_type, 3, storage_order>;

template <typename XprType>
typename XprType::Scalar to_scalar(const XprType& xpr) {
  return Eigen::Tensor<typename XprType::Scalar, 0, storage_order>(xpr)(0);
}

using reward_func_type = std::function<float_type(
    const Tensor2F&, const ConstRowXprAI&, index_type)>;
using SpMats = std::vector<SpMat>;
using SpMatU64 = Eigen::SparseMatrix<uint64_t, storage_order, index_type>;
using SpMatU64s = std::vector<SpMatU64>;
using dists_type = std::vector<std::discrete_distribution<index_type>>;

static constexpr auto inf_v = std::numeric_limits<float_type>::infinity();
static constexpr auto eps_v = std::numeric_limits<float_type>::epsilon();

static XoshiroCpp::Xoshiro256Plus rng;

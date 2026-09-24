#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>

namespace takt {

// Fixed-size, stack-allocated matrix for the tracker's 8-state Kalman filter.
//
// Dimensions are template parameters, so multiplying an 8x4 by a 8x8 is a compile error rather
// than a runtime assert, and every temporary lives on the stack - no allocation in the hot loop,
// no Eigen dependency for ~200 lines of linear algebra.
template <std::size_t Rows, std::size_t Cols, class T = float>
struct Matrix {
  std::array<T, Rows * Cols> values{};

  static constexpr std::size_t kRows = Rows;
  static constexpr std::size_t kCols = Cols;

  [[nodiscard]] static constexpr Matrix identity() noexcept
    requires(Rows == Cols)
  {
    Matrix m;
    for (std::size_t i = 0; i < Rows; ++i) m(i, i) = T{1};
    return m;
  }

  [[nodiscard]] static constexpr Matrix diagonal(const std::array<T, Rows>& diag) noexcept
    requires(Rows == Cols)
  {
    Matrix m;
    for (std::size_t i = 0; i < Rows; ++i) m(i, i) = diag[i];
    return m;
  }

  [[nodiscard]] constexpr T& operator()(std::size_t r, std::size_t c) noexcept {
    return values[r * Cols + c];
  }
  [[nodiscard]] constexpr const T& operator()(std::size_t r, std::size_t c) const noexcept {
    return values[r * Cols + c];
  }

  [[nodiscard]] constexpr Matrix<Cols, Rows, T> transposed() const noexcept {
    Matrix<Cols, Rows, T> t;
    for (std::size_t r = 0; r < Rows; ++r) {
      for (std::size_t c = 0; c < Cols; ++c) t(c, r) = (*this)(r, c);
    }
    return t;
  }

  // Copies the BlockRows x BlockCols sub-matrix starting at (row, col).
  template <std::size_t BlockRows, std::size_t BlockCols>
    requires(BlockRows <= Rows && BlockCols <= Cols)
  [[nodiscard]] constexpr Matrix<BlockRows, BlockCols, T> block(
      std::size_t row = 0, std::size_t col = 0) const noexcept {
    Matrix<BlockRows, BlockCols, T> b;
    for (std::size_t r = 0; r < BlockRows; ++r) {
      for (std::size_t c = 0; c < BlockCols; ++c) b(r, c) = (*this)(row + r, col + c);
    }
    return b;
  }

  constexpr Matrix& operator+=(const Matrix& other) noexcept {
    for (std::size_t i = 0; i < values.size(); ++i) values[i] += other.values[i];
    return *this;
  }
  constexpr Matrix& operator-=(const Matrix& other) noexcept {
    for (std::size_t i = 0; i < values.size(); ++i) values[i] -= other.values[i];
    return *this;
  }

  [[nodiscard]] friend constexpr Matrix operator+(Matrix a, const Matrix& b) noexcept {
    return a += b;
  }
  [[nodiscard]] friend constexpr Matrix operator-(Matrix a, const Matrix& b) noexcept {
    return a -= b;
  }
  friend constexpr bool operator==(const Matrix&, const Matrix&) = default;
};

template <std::size_t R, std::size_t K, std::size_t C, class T>
[[nodiscard]] constexpr Matrix<R, C, T> operator*(const Matrix<R, K, T>& a,
                                                  const Matrix<K, C, T>& b) noexcept {
  Matrix<R, C, T> out;
  for (std::size_t r = 0; r < R; ++r) {
    for (std::size_t k = 0; k < K; ++k) {
      const T a_rk = a(r, k);
      for (std::size_t c = 0; c < C; ++c) out(r, c) += a_rk * b(k, c);
    }
  }
  return out;
}

// Solves A * X = B for symmetric positive-definite A using a Cholesky factorisation. Returns
// nullopt if A is not positive definite. Used instead of an explicit inverse, which is both slower
// and less numerically stable.
template <std::size_t N, std::size_t M, class T>
[[nodiscard]] std::optional<Matrix<N, M, T>> cholesky_solve(const Matrix<N, N, T>& a,
                                                            const Matrix<N, M, T>& b) noexcept {
  Matrix<N, N, T> l;  // lower-triangular factor, A = L * L^T
  for (std::size_t j = 0; j < N; ++j) {
    T diag = a(j, j);
    for (std::size_t k = 0; k < j; ++k) diag -= l(j, k) * l(j, k);
    if (!(diag > T{0})) return std::nullopt;  // also rejects NaN
    l(j, j) = std::sqrt(diag);
    for (std::size_t i = j + 1; i < N; ++i) {
      T sum = a(i, j);
      for (std::size_t k = 0; k < j; ++k) sum -= l(i, k) * l(j, k);
      l(i, j) = sum / l(j, j);
    }
  }

  Matrix<N, M, T> x = b;
  for (std::size_t col = 0; col < M; ++col) {
    for (std::size_t i = 0; i < N; ++i) {  // forward substitution: L * y = b
      T sum = x(i, col);
      for (std::size_t k = 0; k < i; ++k) sum -= l(i, k) * x(k, col);
      x(i, col) = sum / l(i, i);
    }
    for (std::size_t i = N; i-- > 0;) {  // back substitution: L^T * x = y
      T sum = x(i, col);
      for (std::size_t k = i + 1; k < N; ++k) sum -= l(k, i) * x(k, col);
      x(i, col) = sum / l(i, i);
    }
  }
  return x;
}

}  // namespace takt

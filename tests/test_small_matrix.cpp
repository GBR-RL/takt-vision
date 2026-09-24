#include <gtest/gtest.h>

#include "takt/track/small_matrix.hpp"

namespace takt {
namespace {

TEST(SmallMatrix, MultiplyTransposeAndBlocksAreCompileTimeSized) {
  Matrix<2, 3> a;
  a.values = {1, 2, 3, 4, 5, 6};
  const Matrix<3, 2> at = a.transposed();
  const Matrix<2, 2> product = a * at;  // Matrix<2,2> - a mismatch would not compile
  EXPECT_FLOAT_EQ(product(0, 0), 14.0f);
  EXPECT_FLOAT_EQ(product(0, 1), 32.0f);
  EXPECT_FLOAT_EQ(product(1, 1), 77.0f);
  const Matrix<2, 2> right = a.block<2, 2>(0, 1);
  EXPECT_FLOAT_EQ(right(1, 0), 5.0f);
  EXPECT_FLOAT_EQ(right(1, 1), 6.0f);
}

TEST(SmallMatrix, IsUsableInConstantExpressions) {
  constexpr auto i3 = Matrix<3, 3>::identity();
  static_assert(i3(0, 0) == 1.0f && i3(0, 1) == 0.0f);
  constexpr auto twice = i3 + i3;
  static_assert(twice(2, 2) == 2.0f);
  SUCCEED();
}

TEST(SmallMatrix, CholeskySolveMatchesKnownSolution) {
  // SPD matrix and right-hand side with solution x = (1, 2, 3).
  Matrix<3, 3, double> a;
  a.values = {4, 2, 0, 2, 5, 3, 0, 3, 6};
  Matrix<3, 1, double> b;
  b.values = {8, 21, 24};
  const auto x = cholesky_solve(a, b);
  ASSERT_TRUE(x.has_value());
  EXPECT_NEAR((*x)(0, 0), 1.0, 1e-12);
  EXPECT_NEAR((*x)(1, 0), 2.0, 1e-12);
  EXPECT_NEAR((*x)(2, 0), 3.0, 1e-12);
}

TEST(SmallMatrix, CholeskyRejectsIndefiniteMatrix) {
  Matrix<2, 2> a;
  a.values = {1, 2, 2, 1};  // eigenvalues 3 and -1
  EXPECT_FALSE(cholesky_solve(a, Matrix<2, 1>{}).has_value());
}

}  // namespace
}  // namespace takt

#include "takt/track/kalman_filter.hpp"

#include <array>

namespace takt {
namespace {

constexpr float square(float v) noexcept {
  return v * v;
}

// x' = F x with unit time step: position += velocity.
constexpr Matrix<8, 8> motion_model() noexcept {
  auto f = Matrix<8, 8>::identity();
  for (std::size_t i = 0; i < 4; ++i) f(i, i + 4) = 1.0f;
  return f;
}

}  // namespace

BoxKalmanFilter::State BoxKalmanFilter::initiate(const Measurement& z) noexcept {
  State s;
  for (std::size_t i = 0; i < 4; ++i) s.mean(i, 0) = z(i, 0);  // velocities start at zero

  const float h = z(3, 0);
  const float pos = kStdWeightPosition * h;
  const float vel = kStdWeightVelocity * h;
  s.covariance =
      Covariance::diagonal({square(2 * pos), square(2 * pos), square(1e-2f), square(2 * pos),
                            square(10 * vel), square(10 * vel), square(1e-5f), square(10 * vel)});
  return s;
}

void BoxKalmanFilter::predict(State& s) noexcept {
  static constexpr Matrix<8, 8> kF = motion_model();

  const float h = s.mean(3, 0);
  const float pos = kStdWeightPosition * h;
  const float vel = kStdWeightVelocity * h;
  const auto process_noise =
      Covariance::diagonal({square(pos), square(pos), square(1e-2f), square(pos), square(vel),
                            square(vel), square(1e-5f), square(vel)});

  s.mean = kF * s.mean;
  s.covariance = kF * s.covariance * kF.transposed() + process_noise;
}

void BoxKalmanFilter::update(State& s, const Measurement& z) noexcept {
  const float h = s.mean(3, 0);
  const float pos = kStdWeightPosition * h;
  const auto measurement_noise =
      Matrix<4, 4>::diagonal({square(pos), square(pos), square(1e-1f), square(pos)});

  // The measurement matrix H just selects the first four state entries, so H P H^T and P H^T are
  // sub-blocks of P - no need to form H explicitly.
  const Matrix<8, 4> p_ht = s.covariance.block<8, 4>();
  const Matrix<4, 4> innovation_cov = s.covariance.block<4, 4>() + measurement_noise;

  // K = P H^T S^-1  <=>  S K^T = (P H^T)^T, since S is symmetric.
  const auto k_transposed = cholesky_solve(innovation_cov, p_ht.transposed());
  if (!k_transposed) return;
  const Matrix<8, 4> gain = k_transposed->transposed();

  const Measurement innovation = z - s.mean.block<4, 1>();
  s.mean += gain * innovation;
  s.covariance -= gain * p_ht.transposed();  // P - K S K^T == P - K (P H^T)^T
}

BoxKalmanFilter::Measurement BoxKalmanFilter::to_measurement(const Box& box) noexcept {
  Measurement z;
  const float h = box.height();
  z(0, 0) = box.center_x();
  z(1, 0) = box.center_y();
  z(2, 0) = h > 0.0f ? box.width() / h : 0.0f;
  z(3, 0) = h;
  return z;
}

Box BoxKalmanFilter::to_box(const Mean& mean) noexcept {
  const float h = mean(3, 0);
  const float w = mean(2, 0) * h;
  return Box::from_center(mean(0, 0), mean(1, 0), w, h);
}

}  // namespace takt

/*
  Copyright (C) 2008-2011 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#ifndef BOOM_SPARSE_KALMAN_TOOLS_HPP
#define BOOM_SPARSE_KALMAN_TOOLS_HPP

#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/Types.hpp>
#include <Models/StateSpace/Filters/SparseVector.hpp>
#include <Models/StateSpace/Filters/SparseMatrix.hpp>

namespace BOOM{
  // Returns the likelihood contribution of y given previous y's.
  // Uses notation from Durbin and Koopman (2001):
  //
  //     y[t] = Z[t].dot(alpha[t]) + epsilon[t] ~ N(0, H[t])
  // alpha[t] = T[t] * alpha[t-1] + R[t]eta[t] ~ N(0, RQR.transpose())
  //
  // In the code below...
  // a[t] = E(alpha[t]| Y^{t-1})
  // P[t] = V(alpha[t]| Y^{t-1})
  // v[t] is the one step forecast error of y[t] | Y^{t-1}
  // K[t] is the "Kalman Gain:"  a[t+1] = T*a[t] + K*v
  // F[t] = Var(y[t] | Y^{t-1})
  double sparse_scalar_kalman_update(
      double y,                        // y[t]
      Vec &a,                          // a[t] -> a[t+1]
      Spd &P,                          // P[t] -> P[t+1]
      Vec &kalman_gain,          // output as K[t]
      double &forecast_error_variance, // output as F[t]
      double &forecast_error,          // output as v[t]
      bool missing,  // was y observed?
      const SparseVector &Z,  // input
      double observation_variance,
      const SparseKalmanMatrix &T,
      const SparseKalmanMatrix &RQR);   // state transition error variance

  // Updates a[t] and P[t] to condition on all Y, and sets up r and N
  // for use in the next recursion.
  void sparse_scalar_kalman_smoother_update(
      Vec &a,         // a[t] -> E(alpha[t] | Y)
      Spd &P,         // P[t] -> V(alpha[t] | Y)
      const Vec & K,  // K[t] As produced by Kalman filter
      double F,       // F[t] "
      double v,       // v[t] "
      const Vec &Z,   // Z[t] "
      const Mat &T,   // T[t] "
      Vec & r,        // backward Kalman variable, local
      Mat & N);       // backward Kalman variance, local

}
#endif// BOOM_SPARSE_KALMAN_TOOLS_HPP

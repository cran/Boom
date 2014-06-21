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

#include <Models/StateSpace/Filters/KalmanTools.hpp>
#include <Models/StateSpace/Filters/SparseKalmanTools.hpp>
#include <Models/StateSpace/Filters/SparseVector.hpp>
#include <Models/StateSpace/Filters/SparseMatrix.hpp>
#include <distributions.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM{
  double sparse_scalar_kalman_update(
      double y,                         // New observation at time t
      Vec &a,                           // Input a[t].  Output a[t+1]
      Spd &P,                           // Input P[t].  Ouput P[t+1]
      Vec &K,                           // Output K[t] Kalman gain
      double &F,                        // Output F[t] Forecast variance
      double &v,                        // Output v[t] Forecast error
      bool missing,                     // true if y[t] is missing
      const SparseVector & Z,           // Model matrix for obs. equation
      double H,                         // Var(Y | state)
      const SparseKalmanMatrix & T,     // State transition matrix
      const SparseKalmanMatrix & RQR){  // State variance matrix

    Vec PZ = P*Z;
    F = Z.dot(PZ) + H;
    if(F <= 0) {
      std::ostringstream err;
      err << "Found a zero forecast variance:" << endl
          << "missing = " << missing << endl
          << "a = " << a << endl
          << "P = " << endl << P << endl
          << "y = " << y << endl
          << "H = " << H << endl
          << "ZPZ = " << Z.dot(PZ) << endl
          << "Z = " << Z.dense() << endl;
      report_error(err.str());
    }
    Vec TPZ = T * PZ;

    double loglike=0;
    if(!missing){
      K = TPZ/F;
      double mu = Z.dot(a);
      v = y-mu;
      loglike = dnorm(y, mu, sqrt(F), true);
    }else{
      K = a.zero();
      v = 0;
    }

    a = T * a;                      // Sparse multiplication
    if(!missing) a.axpy(K, v);      // a += K * v
    T.sandwich_inplace(P);          // P = T P T.transpose()
    if(!missing){                   // K is zero if missing, so skip this
      P.Mat::add_outer(TPZ, K, -1); // P-= T*P*Z*K.transpose();
    }
    RQR.add_to(P);                  // P += RQR

    return loglike;
  }

}

/*
  Copyright (C) 2005 Steven L. Scott

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
#include <LinAlg/Vector.hpp>
#include <LinAlg/Types.hpp>
#include <cmath>
#include <cpputil/math_utils.hpp>

namespace BOOM{

  double lse_safe(const Vec &eta){
    double m = eta.max();
    if (m == negative_infinity()) return m;
    double tmp=0;
    uint n = eta.size();
    for (uint i=0; i<n; ++i) tmp+= exp(eta[i]-m);
    return m + log(tmp);
  }

  double lse_fast(const Vec & eta){
    double ans = 0;
    uint n = eta.size();
    const double *d(eta.data());
    for (uint i=0; i<n; ++i) {
      ans += exp(d[i]);
    }
    if (ans <= 0) {
      return negative_infinity();
    }
    return log(ans);
  }

  double lse(const Vec &eta){
    return lse_safe(eta);
  }
}

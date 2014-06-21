/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_LSE_HPP
#define BOOM_LSE_HPP
#include <LinAlg/Types.hpp>
#include <cmath>
//#include <tr1/cmath>

namespace BOOM{
  double lse(const Vec &v);
  double lse_safe(const Vec &v);
  double lse_fast(const Vec &v);
  inline double lse2(double x, double y){
    // returns log( exp(x) + exp(y));
    if(x<y){ double tmp(x); x=y; y=tmp; }
    return x + ::log1p(::exp(y-x));
  }
}
#endif // BOOM_LSE_HPP

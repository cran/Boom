/*
  Copyright (C) 2008 Steven L. Scott

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

#ifndef BOOM_SCALAR_KALMAN_STORAGE_HPP
#define BOOM_SCALAR_KALMAN_STORAGE_HPP

#include <LinAlg/Vector.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/Types.hpp>

namespace BOOM{

  // LightKalmanStorage is 'light' because it does not keep a copy of
  // 'a' (the state forecast/state value) or P (variance of state
  // forecast/value).
  struct LightKalmanStorage{
    Vec K;
    double F;
    double v;
    LightKalmanStorage(){}
    LightKalmanStorage(int dim) : K(dim) {}
  };

  struct ScalarKalmanStorage : public LightKalmanStorage{
    Vec a;
    Spd P;
    ScalarKalmanStorage() : LightKalmanStorage() {}
    ScalarKalmanStorage(int dim) : LightKalmanStorage(dim), a(dim), P(dim) {}
  };
}
#endif// BOOM_SCALAR_KALMAN_STORAGE_HPP

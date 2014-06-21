/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#ifndef BOOM_MATH_SPECIAL_FUNCTIONS_HPP_
#define BOOM_MATH_SPECIAL_FUNCTIONS_HPP_

namespace BOOM {

  namespace Cephes {
    double polylog(int n, double x);
    double spence(double x);
  }

  //======================================================================
  // Special functions from the cephes math library:
  inline double polylog(int n, double x) {
    return Cephes::polylog(n, x);
  }

  // The dilogarithm.  Equivalent to polylog(2, x), but faster.
  inline double dilog(double x) {
    //    return Cephes::spence(1 - x);
    return Cephes::polylog(2, x);
  }

}  // namespace BOOM
#endif // BOOM_MATH_SPECIAL_FUNCTIONS_HPP_

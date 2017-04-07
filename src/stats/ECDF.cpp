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

#include <stats/ECDF.hpp>
#include <algorithm>

namespace BOOM{

  ECDF::ECDF(const std::vector<double> &unsorted)
    : sorted_(unsorted)
  {
    std::sort(sorted_.begin(), sorted_.end());
    b = sorted_.begin();
    e = sorted_.end();
    n = static_cast<double>(sorted_.size());
  }

  double ECDF::fplus(double x)const{
    std::vector<double>::const_iterator it = std::upper_bound(b, e, x);
    unsigned pos = it - b;  // it is the first element >= x
    return pos / n;
  }

  double ECDF::fminus(double x)const{
    std::vector<double>::const_iterator it = std::lower_bound(b, e, x);
    unsigned pos = it - b;
    return pos / n;
  }

}  // namespace BOOM

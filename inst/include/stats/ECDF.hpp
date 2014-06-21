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
#include <vector>

namespace BOOM{
  class ECDF{
    // empirical CDF
  public:
    ECDF(const std::vector<double> &unsorted);
    double fplus(double x)const;  // fraction of data <= x
    double fminus(double x)const; // fraction of data < x;
    double operator()(double x, bool leq = true)const{
      return leq ? fplus(x) : fminus(x);}
    const std::vector<double> & sorted_data()const{return sorted_;}
  private:
    std::vector<double> sorted_;
    std::vector<double>::const_iterator b, e;
    double n;
  };
}

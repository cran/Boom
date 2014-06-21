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
#ifndef BOOM_SPLINE_HPP
#define BOOM_SPLINE_HPP
#include <LinAlg/Vector.hpp>
#include <LinAlg/Types.hpp>
#include <vector>
#include <boost/shared_ptr.hpp>

namespace BOOM{
  class Spline{
    //cubic "natural" spline
  public:
    Spline(const Vec & Knots, uint Order=4);
    int nknots()const;
    Vec basis(double x)const;
    const Vec & basis(double x, Vec &ans)const;
    double eval(double x, const Vec &beta)const;
    double evaluate_derivs(double x, int nder)const;
  private:
    int order_;
    int ordm1_; // order -1 (3 for cubic splines)

    mutable int curs; // current position in knots vector
    mutable bool boundary;  // must have knots[curs] <= x < knots[curs+1]
                            // except for the boundary case

    Vec knots;       // knots
    mutable Vec rdel;
    mutable Vec ldel;
    mutable Vec a;   // scratch array

    void set_cursor(double x)const;
    void basis_funcs(double x, Vec &ans)const;
    void diff_table(double x, int ordm1)const;
  };

}
#endif// BOOM_SPLINE_HPP

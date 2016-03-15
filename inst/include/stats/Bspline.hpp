/*
  Copyright (C) 2015 Steven L. Scott

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

#ifndef BOOM_STATS_BSPLINE_HPP_
#define BOOM_STATS_BSPLINE_HPP_

#include <LinAlg/Vector.hpp>

namespace BOOM {

  // Compute a Bspline basis expansion of a scalar value x.  A Bspline
  // is a spline formed by a set of local basis functions (the
  // b-spline basis) determined by a set of knots.  The knots
  // partition an interval [lo, hi] over which the spline function
  // nonzero.  The spline basis is zero over (-infinity, lo) and (hi,
  // infinity).
  //
  // To make the B-spline theory work, the notional set of knots is
  // supposed to be infinite, but in practice a B-spline basis
  // function of degree d is nonzero over at most d+1 knot spans.
  // This means we can add d+1 fake knots at the beginning and the end
  // of the knot sequence, and the knots can be in any arbitrary
  // positions.  This class follows an established convention of
  // adding the fake knots at the first and last elements of the knot
  // vector.
  class Bspline {
   public:
    // Args:
    //   knots: The set of knots for the spline.  In between pairs of
    //     knots, the spline is a piecewise polynomial whose degree is
    //     given by the second argument.
    //   degree: The degree of the piecewise polynomial in between
    //     pairs of interior knots.
    Bspline(const Vector &knots, int degree = 3);

    // The Bspline basis expansion at the value x.  If x lies outside
    // the range [knots.begin(), knots.end()] then all basis elements
    // are zero.
    //
    // If there are fewer than 2 knots then the return value is empty.
    Vector basis(double x) const;

    // The dimension of the spline basis, which is one for every
    // distinct interval covered by knots(), plus one for every degree
    // of the piecewise polynomial.  Normally this is number_of_knots
    // - 1 + degree, though it can be less if knots() contains
    // duplicate elements.  If knots().size <= 1 then the
    // basis_dimension is 0.
    int basis_dimension() const {return basis_dimension_;}

    // The order of the piecewise polynomial connecting the knots.
    int order() const {return order_;}

    // The degree of the piecewise polynomial connecting the knots.
    int degree() const {return order_ - 1;}

    // Adds a knot at the given location.  If knot_location lies
    // before the first or after the last current knot, then the
    // domain of the Bspline is extended to cover knot_location.
    void add_knot(double knot_location);

    // Remove the specified knot.  An exception will be thrown if
    // which_knot is outside the range of knots_.  If which_knot == 0
    // or which_knot == number_of_knots() - 1 then the domain of the
    // spline basis will be reduced.
    void remove_knot(int which_knot);

    // The vector of knots.  Implicit boundary knots are not included.
    const Vector &knots() const {return knots_;}
    int number_of_knots() const {return knots_.size();}

    // If the argument is in the interior of the knots vector, return
    // knots_[i].  If it is off the end to the left return knots_[0].
    // If it is off the end to the right then include knots_.back().
    // The implicit assumption is that we have an infinite set of
    // knots piled up on the beginning and end of the actual knot
    // sequence.
    double knot(int i) const;

    // Compute the coefficient C for combining two splines of order degree-1
    // into a spline of order degree.  The recursion is
    // basis[i, degree](x) =
    //     C(x, i, degree) * basis[i, degree-1]
    // + (1 - C(x, i + 1, degree)) * basis[i + 1, degree - 1].
    // See deBoor, page 90, chapter IX, formulas (15) and (16).
    double compute_coefficient(double x, int knot_span, int degree) const;

   private:
    // The vector of knots defining the spline.
    Vector knots_;

    // The order (1 + degree) of the piecewise polynomial connecting
    // the knots.
    int order_;

    // The dimension of the spline basis expansion.
    int basis_dimension_;
  };

} // namespace BOOM

#endif  //  BOOM_STATS_BSPLINE_HPP_

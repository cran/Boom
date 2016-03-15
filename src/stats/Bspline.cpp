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
#include <stats/Bspline.hpp>
#include <LinAlg/Matrix.hpp>
#include <cpputil/math_utils.hpp>
#include <cpputil/report_error.hpp>
#include <sstream>

namespace BOOM{

  Bspline::Bspline(const Vector &knots, int degree)
      : knots_(knots),
        order_(degree + 1)
  {
    knots_.sort();
    if (degree < 0) {
      report_error("Spline degree must be non-negative.");
    }
    if (knots_.size() <= 1) {
      // If the knot vector contains 0 or 1 knots then the "spline
      // basis expansion" of x is just the empty vector.
      basis_dimension_ = 0;
    } else {
      // There are number_of_knots() - 1 knot spans, each with its own
      // basis function.  Each degree of the spline adds one more
      // basis element, from the basis functions corresponding to the
      // additional fake knots at knots[0].
      basis_dimension_ = knots_.size() - 1 + degree;
    }
    for (int i = 1; i < knots_.size(); ++i) {
      // Any duplicate knots reduce the number of distinct knot spans.
      if (knots_[i] == knots_[i-1]) {
        --basis_dimension_;
      }
    }
    if (basis_dimension_ < 0) {
      basis_dimension_ = 0;
    }
  }

  Vector Bspline::basis(double x) const {
    if (basis_dimension_ == 0) {
      return Vector(0);
    }

    // Each basis function looks forward 'degree' knots to include x.
    Vector ans(basis_dimension(), 0.0);
    if (x < knots_[0] || x > knots_.back()) {
      return ans;
    }

    // To find the knot in the left endpoint of the knot span, we first find the
    Vector::const_iterator terminal_knot_position = std::upper_bound(
        knots_.begin(), knots_.end(), x);
    int terminal_knot = terminal_knot_position - knots_.begin();
    int knot_span_for_x = terminal_knot - 1;

    // Each row of the basis_function_table corresponds to one knot
    // span.  Row zero corresponds to [knots_[0], knots_[1]), row 1 to
    // [knots_[1], knots_[2]), etc.  The columns correspond to the
    // degree of the spline.  We first compute the zero-degree bases.
    // We can use the zero-degree bases to get the first degree bases,
    // etc.
    //
    // The zero-degree basis is 1 in exactly one position (the knot
    // span containing x).  The linear basis is nonzero in two
    // positions: knot span containing x and the the one prior to
    // that.  Each additional degree increases the number of nonzero
    // knot-spans by 1.
    //
    // Given the initial (zero-degree) basis, we can compute each
    // higher degree basis using the recurrence relation:
    //   basis[knot_span, degree](x) =
    //       L(x, knot, degree) * basis(knot, degree - 1)
    //      +R(x, knot, degree) * basis(knot + 1, degree - 1)
    // The left coefficient L is
    //
    //                        (x - knots_[knot])
    // L(x, knot, degree) =   ---------------------
    //                  knots_[knot + degree] - knots_[knot]
    //
    // if the denominator is zero (because of knot multiplicity) then
    // the value of L is arbitrary, so we set it to zero.
    //
    // and the right coefficient R is
    //     R(x, knot, degree) = 1 - L(x, knot + 1, degree).
    //
    // All this is explained in deBoor ("A Practical Guide to
    // Splines") chapter IX, and online at
    // www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html

    int number_of_knot_spans = number_of_knots() - 1;

    // The entries of basis_function_table, column d, are the spline
    // basis functions of degree d.
    ArbitraryOffsetMatrix basis_function_table(
        -degree(), number_of_knot_spans + order_,
        0, order_,
        0.0);

    basis_function_table(knot_span_for_x, 0) = 1.0;
    for (int d = 1; d <= degree(); ++d) {
      // For a given knot_span containing x, each spline basis of
      // degree d is nonzero over its knot span, and the next d spans.
      // Thus to find the set of nonzero bases containing x, we must
      // look _back_ d spaces.
      for (int lag = 0; lag <= d; ++lag) {
        int span = knot_span_for_x - lag;
        double left_coefficient = compute_coefficient(x, span, d);
        double right_coefficient =
            1 - compute_coefficient(x, span + 1, d);
        double left_basis = basis_function_table(span, d - 1);
        double right_basis =
            span < number_of_knot_spans ?
                   basis_function_table(span + 1, d - 1) : 0.0;
        basis_function_table(span, d) = left_coefficient * left_basis
            + right_coefficient * right_basis;
      }
    }
    if (number_of_knots() > 1) {
      for (int i = -degree(); i < number_of_knot_spans; ++i) {
        ans[i + degree()] = basis_function_table(i, degree());
      }
    }
    return(ans);
  }

  void Bspline::add_knot(double knot_location) {
    knots_.insert(std::lower_bound(knots_.begin(),
                                   knots_.end(),
                                   knot_location),
                  knot_location);
    ++basis_dimension_;
  }

  void Bspline::remove_knot(int which_knot) {
    if (which_knot < 0 || which_knot >= number_of_knots()) {
      report_error("Requested knot is not in range.");
    }
    knots_.erase(knots_.begin() + which_knot);
    --basis_dimension_;
  }

  double Bspline::knot(int i) const {
    if (knots_.empty()) {
      return negative_infinity();
    } else {
      if (i <= 0) {
        return knots_[0];
      } else if (i >= knots_.size()) {
        return knots_.back();
      } else {
        return knots_[i];
      }
    }
  }

  double Bspline::compute_coefficient(
      double x, int knot_span, int degree) const {
    if (knot(knot_span) < knot(knot_span + degree)) {
      double dknot = knot(knot_span + degree) - knot(knot_span);
      return (x - knot(knot_span)) / dknot;
    } else {
      // In this case the result is arbitrary, because the coefficient
      // will only multiply basis functions with the value zero.
      // http://math.stackexchange.com/questions/52157/can-cox-de-boor-recursion-formula-apply-to-b-splines-with-multiple-knots
      return 0;
    }
  }


}

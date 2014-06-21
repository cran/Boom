/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#ifndef BOOM_BOUNDED_ADAPTIVE_REJECTION_SAMPLER_HPP
#define BOOM_BOUNDED_ADAPTIVE_REJECTION_SAMPLER_HPP

#include <boost/function.hpp>
#include <distributions.hpp>
#include <vector>

namespace BOOM{

// for sampling from a log-concave strictly decreasing function
  class BoundedAdaptiveRejectionSampler{
   public:
    typedef boost::function<double(double)> Fun;

    // logf must be a concave function( e.g -x^2) with derivative
    // given by dlogf.  lower_bound must be to the right of the mode
    // of logf
    BoundedAdaptiveRejectionSampler(double lower_bound, Fun logf, Fun dlogf);
    double draw(RNG & );              // simluate a value
    void add_point(double x);         // adds the point to the hull
    double f(double x)const;          // log of the target distribution
    double df(double x)const;         // derivative of logf at x
    double h(double x, uint k)const;  // evaluates the outer hull at x
    std::ostream & print(std::ostream & out)const;
   private:
    Fun logf_;
    Fun dlogf_;

    std::vector<double> x;
    // points that have been tried thus far, stored in ascending
    // order

    std::vector<double> logf;
    // function values corresponding to values in x

    std::vector<double> dlogf;
    // derivatives of the log target density evaluated at x

    std::vector<double> knots;
    // contains the points of intersection between the tangent lines
    // to logf at x.  First knot is x[0].  Later knots satisfy x[i-1]
    // < knots[i] < x[i].

    std::vector<double> cdf;     // cdf[i] = cdf[i-1] + the integral of
    // the hull from knots[i] to
    // knots[i+1].  cdf.back() assumes a
    // final knot at infinity

    void update_cdf();
    void refresh_knots();
    double compute_knot(uint k)const;
    typedef std::vector<double>::iterator IT;
  };

}
#endif // BOOM_BOUNDED_ADAPTIVE_REJECTION_SAMPLER_HPP

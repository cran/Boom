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

#include <distributions/BoundedAdaptiveRejectionSampler.hpp>
#include <cpputil/report_error.hpp>
#include <sstream>
#include <stdexcept>

namespace BOOM{

  typedef BoundedAdaptiveRejectionSampler BARS;
  BARS:: BoundedAdaptiveRejectionSampler(double a, Fun Logf, Fun Dlogf)
      : logf_(Logf),
        dlogf_(Dlogf),
        x(1, a),
        logf(1, f(a)),
        dlogf(1, df(a)),
        knots(1,a)
  {
    if(dlogf[0] >=0){
      std::ostringstream err;
      err << "lower bound of " << a << " must be to the right of the mode of "
          << "logf in BoundedAdaptiveRejectionSampler" << std::endl
          << "a        = " << a << std::endl
          << "logf(a)  = " << logf[0] << std::endl
          << "dlogf(a) = " << dlogf[0] << std::endl;
      report_error(err.str());
    }
    update_cdf();
  }

  double BARS::f(double x)const{return logf_(x);}
  double BARS::df(double x)const{return dlogf_(x);}

  //----------------------------------------------------------------------
  void BARS::add_point(double z){
    IT it = std::lower_bound(knots.begin(), knots.end(), z);

    if(it==knots.end()){
      //    cout << "inserting at end " << endl;
      x.push_back(z);
      logf.push_back(f(z));
      dlogf.push_back(df(z));
    }else{
      uint k = it - knots.begin();
      x.insert(x.begin()+k, z);
      logf.insert(logf.begin()+k, f(z));
      dlogf.insert(dlogf.begin() + k, df(z));
    }

    refresh_knots();
    update_cdf();
  }
  //----------------------------------------------------------------------
  void BARS::refresh_knots(){
    // wasteful!  should only update a knot between the x's, but
    // adding an x will change two knots
    knots.resize(x.size());
    knots[0] = x[0];
    for(uint i=1; i<knots.size(); ++i)
      knots[i] = compute_knot(i);
  }
  //----------------------------------------------------------------------
  // returns the location of the intersection of the tanget line at
  // x[k] and x[k-1]
  double BARS::compute_knot(uint k)const{

  double y2 = logf[k];
  double y1 = logf[k-1];
  double d2 = dlogf[k];
  double d1 = dlogf[k-1];
  double x2 = x[k];
  double x1 = x[k-1];

  // If d2 == d1 then you've reached a spot of exponential decay, or
  // else x2 == x1.
  if(d2 == d1) return x1;

  double ans = ( y1 - d1*x1) - (y2 - d2 * x2);
  ans /= (d2-d1);
  return ans;
}
  //----------------------------------------------------------------------
  void BARS::update_cdf(){
    // cdf[i] is the integral of the outer hull from knots[i] to
    // knots[i+1], where the last value is implicitly infinity.

    // cdf is un-normalized, so we divide everything by exp(y0)
    uint n = knots.size();
    cdf.resize(n);
    double y0 = logf[0];
    double last = 0;
    for(uint k=0; k<knots.size(); ++k){
      double d= dlogf[k];
      double y = logf[k] - y0;
      double z = x[k];
      double dinv = 1.0/d;
      double inc1 = k==n-1? 0 : dinv * exp(y - d * z + d * knots[k+1]);
      double inc2 = dinv * exp(y - d * z + d * knots[k]);
      cdf[k] = last + inc1-inc2;
      last = cdf[k];
    }
  }
  //----------------------------------------------------------------------
  double BARS::h(double z, uint k)const{
    double xk = x[k];
    double dk = dlogf[k];
    double yk = logf[k];
    return yk + dk*(z-xk);
  }
 //----------------------------------------------------------------------
 double BARS::draw(RNG & rng){

   double u= runif_mt(rng, 0, cdf.back());
   IT pos = std::lower_bound(cdf.begin(), cdf.end(), u);
   uint k = pos - cdf.begin();
   double cand;
   if(k+1 == cdf.size()){
     // one sided draw..................
     cand = knots.back() + rexp_mt(rng, -1*dlogf.back());
   }else{
     // draw from the doubly truncated exponential distribution
     double lo = knots[k];
     double hi = knots[k+1];
     double lam = -1*dlogf[k];
     cand = rtrun_exp_mt(rng, lam, lo, hi);
   }
   double target = f(cand);
   double hull = h(cand, k);
   double logu = hull - rexp_mt(rng, 1);
   // The <= in the following statement is important in edge cases
   // where you're very close to the boundary.
   if(logu <= target) return cand;
   add_point(cand);
   return draw(rng);
 }

}

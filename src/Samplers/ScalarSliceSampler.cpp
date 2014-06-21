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

#include <Samplers/ScalarSliceSampler.hpp>
#include <distributions.hpp>
#include <cpputil/math_utils.hpp>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <BOOM.hpp>
#include <algorithm>

namespace BOOM{
  typedef ScalarSliceSampler SSS;

//  SSS::ScalarSliceSampler(const ScalarTargetFun &F, bool Unimodal, double dx)
  SSS::ScalarSliceSampler(const Fun &F, bool Unimodal, double dx)
    : logf_(F),
      suggested_dx_(dx),
      min_dx_(-1),
      lo_set_manually_(false),
      hi_set_manually_(false),
      unimodal_(Unimodal),
      estimate_dx_(true)
  {}

  void SSS::set_suggested_dx(double dx){suggested_dx_ = dx;}
  void SSS::set_min_dx(double dx){min_dx_ = dx;}
  void SSS::estimate_dx(bool yn){ estimate_dx_ = yn; }

  void SSS::set_limits(double Lo, double Hi){
    assert(Hi>Lo);
    lo_ = lower_bound_ = Lo;
    hi_ = upper_bound_ = Hi;
    lo_set_manually_ = hi_set_manually_ = true;
  }
  void SSS::set_lower_limit(double Lo){
    lo_ = lower_bound_ = Lo;
    lo_set_manually_=true;
    hi_set_manually_ = false;
  }
  void SSS::set_upper_limit(double Hi){
    hi_ = upper_bound_ = Hi;
    lo_set_manually_ = false;
    hi_set_manually_=true;
  }

  void SSS::unset_limits(){hi_set_manually_=lo_set_manually_=false;}

  double SSS::logp(double x)const{ return logf_(x);}

  double SSS::draw(double x){
    find_limits(x);
    double logp_cand = 0;
    int number_of_tries = 0;
    do{
      double x_cand = runif_mt(rng(), lo_, hi_);
      logp_cand = logf_(x_cand);
      if(logp_cand < logp_slice_){
        contract(x,x_cand, logp_cand);
        ++number_of_tries;
      } else return x_cand;
      if(number_of_tries > 100){
        ostringstream err;
        err << "number of tries exceeded.  candidate value is "
            << x_cand << " with logp_cand = " << logp_cand << endl;
        throw_exception(err.str(), x);
      }
    }while(logp_cand < logp_slice_);
    throw_exception("should never get here", x);
    return 0;
  }

// If a candidate draw winds up out of the slice then the pseudo-slice
// can be made narrower to increase the chance of success next time.
// See Neal (2003).
  void SSS::contract(double x, double x_cand, double logp){
    if(x_cand > x){
      hi_ = x_cand;
      logphi_ = logp;
    }else{
      lo_ = x_cand;
      logplo_ = logp;
    }
    if(estimate_dx_){
      suggested_dx_ = hi_ - lo_;
      if (suggested_dx_ < min_dx_) suggested_dx_ = min_dx_;
    }
  }

// driver function to find the limits of a slice containing 'x'.
// Logic varies according to whether the distribution is bounded
// above, below, both, or neither.
  void SSS::find_limits(double x){
    logp_slice_ = logf_(x) - rexp_mt(rng(), 1.0);
    check_finite(x,logp_slice_);
    if(doubly_bounded()){
      lo_ = lower_bound_;
      logplo_ = logf_(lo_);
      hi_ = upper_bound_;
      logphi_ = logf_(hi_);
    }else if (lower_bounded()){
      lo_ = lower_bound_;
      logplo_ = logf_(lo_);
      find_upper_limit(x);
    }else if(upper_bounded()){
      find_lower_limit(x);
      hi_ = upper_bound_;
      logphi_ = logf_(hi_);
    }else{ // unbounded
      find_limits_unbounded(x);
    }
    check_slice(x);
    check_probs(x);
  }

// find the upper and lower limits of a slice containing x for a
// potentially multimodal distribution.  Uses Neal's (2003 Annals of
// Statistics) doubling algorithm
  void SSS::find_limits_unbounded(double x){
    hi_ = x + suggested_dx_;
    lo_ = x - suggested_dx_;
    if(unimodal_){
      find_limits_unbounded_unimodal(x);
    }else{
      while(!done_doubling()){
        double u = runif_mt(rng(), -1, 1);
        if(u>0) double_hi(x);
        else double_lo(x);
      }
    }
    check_upper_limit(x);
    check_lower_limit(x);
  }

// utility function used by find_limits_unbounded
  bool SSS::done_doubling()const{
    return (logphi_ < logp_slice_) && (logplo_ < logp_slice_);
  }

// find the upper and lower limits of a slice when the target
// distribution is known to be unimodal
  void SSS::find_limits_unbounded_unimodal(double x){
    hi_ = x + suggested_dx_;
    logphi_ = logf_(hi_);
    while(logphi_ >= logp_slice_) double_hi(x);
    check_upper_limit(x);

    lo_ = x - suggested_dx_;
    logplo_ = logf_(lo_);
    while(logplo_ >= logp_slice_) double_lo(x);
    check_lower_limit(x);
  }

  void SSS::find_upper_limit(double x){
    hi_ = x + suggested_dx_;
    logphi_ = logf_(hi_);
    while(logphi_ >= logp_slice_ || (!unimodal_ && runif_mt(rng()) > .5)){
      double_hi(x);
    }
    check_upper_limit(x);
  }

  void SSS::find_lower_limit(double x){
    lo_ = x - suggested_dx_;
    logplo_ = logf_(lo_);
    while(logplo_ >= logp_slice_ || (!unimodal_ && runif_mt(rng()) > .5)){
      double_lo(x);
    }
    check_lower_limit(x);
  }

  std::string SSS::error_message(double lo, double hi, double x,
                                 double logplo, double logphi,
                                 double logp_slice) const {
    ostringstream err;
    err << endl
        << "lo = " << lo << "  logp(lo) = " << logplo << endl
	<< "hi = " << hi << "  logp(hi) = " << logphi << endl
	<< "x  = " << x  << "  logp(x)  = " << logp_slice << endl;
    return err.str().c_str();
  }

  void SSS::throw_exception(const std::string & msg, double x)const{
    BOOM::throw_exception<std::runtime_error>(
        msg + " in ScalarSliceSampler" +
        error_message(lo_, hi_, x, logplo_, logphi_, logp_slice_));
  }

// makes the upper end of the slice twice as far away from x, and
// updates the density value
  void SSS::double_hi(double x){
    double dx = hi_ - x;
    hi_ = x + 2 * dx;
    if(!std::isfinite(hi_)){
      throw_exception("infinite upper limit", x);
    }
    logphi_ = logf_(hi_);
  }

// makes the lower end of the slice twice as far away from x, and
// updates the density value
  void SSS::double_lo(double x){
    double dx = x - lo_;
    lo_  = x-2*dx;
    if(!std::isfinite(lo_)) throw_exception("infinite lower limit", x);
    logplo_ = logf_(lo_);
  }

  //------ Quality assurance and error handling  ---------------------
  void SSS::check_slice(double x){
    if(x<lo_ || x>hi_)
      throw_exception("problem building slice:  x out of bounds", x);
    if(lo_>hi_)
      throw_exception("problem building slice:  lo > hi", x);
  }

  void SSS::check_probs(double x){
    // logp may be infinite at the upper or lower bound
    bool logood = lower_bounded() || (logplo_ <= logp_slice_);
    bool higood = upper_bounded() || (logphi_ <= logp_slice_);
    if( logood  && higood) return;
    throw_exception("problem with probabilities", x);
  }

  bool SSS::lower_bounded()const{return lo_set_manually_;}
  bool SSS::upper_bounded()const{return hi_set_manually_;}
  bool SSS::doubly_bounded()const{return lo_set_manually_ && hi_set_manually_;}
  bool SSS::unbounded()const{ return !(lo_set_manually_ || hi_set_manually_);}

  void SSS::check_finite(double x, double logp_slice_){
    if(std::isfinite(logp_slice_)) return;
    throw_exception("initial value leads to infinite probability", x);
  }

  void SSS::check_upper_limit(double x){
    if(x>hi_) throw_exception("x beyond upper limit", x);
    if(!std::isfinite(hi_)) throw_exception("upper limit is infinite", x);
    if(isnan(logphi_)) throw_exception("upper limit givs NaN probability", x);
  }

  void SSS::check_lower_limit(double x){
    if(x<lo_) throw_exception("x beyond lower limit", x);
    if(!std::isfinite(lo_)) throw_exception("lower limit is infininte", x);
    if(isnan(logplo_)) throw_exception("lower limit givs NaN probability", x);
  }

}

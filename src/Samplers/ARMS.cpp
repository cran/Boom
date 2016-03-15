/*
  Copyright (C) 2006 Steven L. Scott

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
#include <Samplers/ARMS.hpp>
#include <Samplers/Gilks/arms.hpp>
#include <sstream>
#include <stdexcept>
#include <cpputil/math_utils.hpp>
#include <cpputil/report_error.hpp>
#include <cstdlib>
#include <limits>
#include <numopt.hpp>

namespace BOOM{

  typedef ArmsSampler ARMS;

  double localfun(double x, void *Obj){
    ARMS * obj = static_cast<ARMS *>(Obj) ;
    obj->set(x);
    return obj->eval();
  }

  ARMS::ArmsSampler(Target target, const Vector & initial_value, bool lc)
    : f(target),
      x(initial_value),
      lower_limits(initial_value),
      upper_limits(initial_value),
      ninit(4),
      log_convex(lc)
  {
    find_limits();
  }

  void ARMS::find_limits(){
    max_nd0(x, f);
    lower_limits = x-1.0;
    upper_limits = x+1.0;  // these get adjusted later
  }

  Vector ARMS::draw(const Vector &old){
    using std::endl;
    x=old;
    for(uint i=0; i<x.size(); ++i){
      which = i;
      double lo = lower_limits[i];
      double hi = upper_limits[i];
      double now = x[i];
      double ans = now;
      void * this_ptr(this);
      int err = GilksArms::arms_simple(rng(), ninit, &lo, &hi, localfun, this_ptr,
				       !log_convex, &now, &ans);
      if(err){
	std::ostringstream msg;
	msg << "Error signal recieved in ARMS::draw "
	    << "ninit = " << ninit << endl
	    << "lo    = " << lo << endl
	    << "hi    = " << hi << endl
	    << "log_convex = " << log_convex << endl
	    << "now   = " << now << endl
	    << "ans   = " << ans <<endl;
	report_error(msg.str());
      }
      double width = hi-lo;
      if( fabs(hi-ans) < 1.0) upper_limits[i] += .5*width;
      if( fabs(ans-lo) < 1.0) lower_limits[i] -= .5*width;
      x[which]=ans;
    }
    return x;
  }

  double ARMS::logp(const Vector &v)const{ return f(v);}
  double ARMS::eval()const{return logp(x);}
  void ARMS::set(double y){x[which]=y;}
  void ARMS::set_limits(const Vector &lo, const Vector &hi){
    lower_limits = lo;
    upper_limits = hi;
  }
  void ARMS::set_lower_limits(const Vector &lo){
    lower_limits = lo;}
  void ARMS::set_upper_limits(const Vector &hi){
    upper_limits=hi;}

}

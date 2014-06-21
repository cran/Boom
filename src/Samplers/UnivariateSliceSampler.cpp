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
#include <Samplers/UnivariateSliceSampler.hpp>
#include <distributions.hpp>
#include <cmath>
#include <cassert>
#include <cpputil/math_utils.hpp>

namespace BOOM{
  typedef UnivariateSliceSampler USS;

  USS::UnivariateSliceSampler(const TargetFun &F, bool UniModal)
    : f(F),
      unimodal(UniModal)
  {
    lo = rnorm();
    hi = lo+runif_mt(rng());
  }

  Vec USS::draw(const Vec &x){
    theta = x;
    //    wsp = theta;
    uint n = x.size();
    for(uint i=0; i<n; ++i){
      which = i;
      draw_1();
    }
    return theta;
  }

  double USS::logp(const Vec &x)const{return f(x);}

  //--------------------------------------------------------------

  void USS::draw_1(){
    y = theta[which];
    wsp = theta;
    initialize();
    find_limits();
    double p = pstar-1;
    do{
      double z = runif_mt(rng(), lo, hi);
      p=f1(z);
      if(p < pstar) contract(z, p);
      else theta[which] = z;
    }while(p<pstar);
  }


  void USS::contract(double z, double p){
    if(z<y){
      lo = z;
      plo = p;
    }else if(z>y){
      hi = z;
      phi = p;}}


  void USS::doubling(bool upper){
    double & val( upper ? hi : lo);
    double & p ( upper ? phi : plo);
    int sgn = upper ? 1 : -1;
    double old = val;

    double d = hi-lo;
    val += sgn*d;
    validate(val, p, old);
  }

  void USS::find_limits(){
    if(unimodal){
      // extend upper and lower limits until they exceed the slice
      while(phi > pstar) doubling(true);
      while(plo > pstar) doubling(false);
      return;
    }

    // if not unimodal....
    // randomly extend limits until both are out of the slice
    while(phi > pstar || plo > pstar){
      double tmp = runif_mt(rng(), -1, 1);
      doubling(tmp > 0);
    }
  }

  void USS::initialize(){
    // will be called only by draw_1
    // will be defeated if y is a boundary value.
    pstar = f(theta) - rexp(1);
    assert(std::isfinite(pstar) &&
	   "invalid condition used to initialize UniformSliceSampler");
    if(lo>=y) lo = y-1.0;
    validate(lo, plo, y);

    if(hi <=y) hi = y+1.0;
    validate(hi, phi, y);
  }

  void USS::validate(double &value, double &prob, double anchor){
    prob = f1(value);
    while(!std::isfinite(prob)){
      value = .5*(value+anchor);
      prob = f1(value);
    }
  }

  double USS::f1(double x){
    wsp[which] =x ;
    return f(wsp);
  }

}

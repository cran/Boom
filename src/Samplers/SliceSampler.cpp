/*
  Copyright (C) 2005 Steven L. Scott

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
#include <Samplers/SliceSampler.hpp>
#include <cpputil/math_utils.hpp>
#include <cpputil/report_error.hpp>
#include <distributions.hpp>
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace BOOM{
  SliceSampler::SliceSampler(Func F, bool Unimodal)
    : unimodal(Unimodal),
      f(F)
  {
    hi = lo = scale =1.0;
    assert(scale>0);
  }

  void SliceSampler::initialize(){
    random_direction();
    pstar = f(theta);
    if(!std::isfinite(pstar)){
      std::string msg = "invalid condition used to initialize SliceSampler";
      report_error(msg);
    }

    plo = f(theta-lo*z);
    while(!std::isfinite(plo)){
      lo/=2.0;
      plo = f(theta-lo*z);
    }

    phi = f(theta+hi*z);
    while(!std::isfinite(phi)){
      hi/=2.0;
      phi = f(theta+hi*z);
    }
  }

  void SliceSampler::random_direction(){
    for(uint i=0; i<z.size(); ++i) z[i] = scale*rnorm(); }

  void SliceSampler::doubling(bool upper){
    int sgn = upper ? 1 : -1;
    double & val(upper ? hi : lo);
    double old = val;
    double &p(upper ? phi : plo);

    val *= 2.0;
    p = f(theta + sgn*val*z);
    while(isnan(p)){
      val = (val + old)/2;
      p = f(theta + sgn*val*z);
    }
  }

  void SliceSampler::find_limits(){
    if(unimodal){
      while(phi > pstar) doubling(true);
      while(plo > pstar) doubling(false);
    }else{
      while(phi > pstar || plo > pstar){
	double tmp = runif_mt(rng(), -1,1);
	doubling(tmp>0);}}}


  void SliceSampler::contract(double lam, double p){
    if(lam<0){
      lo = fabs(lam);
      plo = p;
    }else if(lam>0){
      hi = lam;
      phi = p;}}


  double SliceSampler::logp(const Vector &x)const{
    return f(x);
  }

  Vector SliceSampler::draw(const Vector &t){
    theta = t;
    z = t;

    initialize();

    pstar = f(theta) - rexp(1);
    find_limits();
    Vector tstar(theta.size(), 0.0);
    double p = pstar -1;
    do{
      double lam = runif_mt(rng(), -lo, hi);
      tstar = theta + lam*z;   // randomly chosen point in the slice
      p = f(tstar);
      if(p<pstar) contract(lam,p);
      else theta  = tstar;
    }while(p < pstar);
    scale = hi+lo;  // both hi and lo >0
    return theta;
  }


}

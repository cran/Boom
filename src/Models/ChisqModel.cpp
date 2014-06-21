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
#include <Models/ChisqModel.hpp>
#include <cmath>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <distributions.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM{

  typedef ChisqModel CSM;
  typedef GammaModelBase GMB;

  CSM::ChisqModel(double df, double sigsq)
    : GMB(),
      ParamPolicy(new UnivParams(df), new UnivParams(sigsq)),
      PriorPolicy()
  { }

  CSM::ChisqModel(const CSM & rhs)
    : Model(rhs),
      MLE_Model(rhs),
      GMB(rhs),
      ParamPolicy(rhs),
      PriorPolicy(rhs)
  {}

  CSM * CSM::clone()const{return new CSM(*this);}

  Ptr<UnivParams> CSM::Df_prm(){return ParamPolicy::prm1();}
  Ptr<UnivParams> CSM::Sigsq_prm(){return ParamPolicy::prm2();}
  const Ptr<UnivParams> CSM::Df_prm()const{return ParamPolicy::prm1();}
  const Ptr<UnivParams> CSM::Sigsq_prm()const{return ParamPolicy::prm2();}

  double CSM::df()const{return Df_prm()->value();}
  double CSM::sigsq()const{return Sigsq_prm()->value();}
  double CSM::sum_of_squares()const{return sigsq() * df();}

  double CSM::alpha()const{return df()/2.0;}
  double CSM::beta()const{return sigsq()*df()/2.0;}
  double CSM::Loglike(Vec &g, Mat &h, uint nd)const{

    double n = suf()->n();
    double sum = suf()->sum();
    double sumlog = suf()->sumlog();

    double d = df();
    double s = sigsq();

    if(d<=0 || s<=0){
      if(nd>0){
	g[0] = (d <=0) ? d : 0;
	g[1] = (s <= 0) ? s : 0;
	if(nd>1) h.set_diag(-1);
      }
      return BOOM::negative_infinity();
    }

    //-----
    double logds2 = log(d*s/2);
    double halfn = n/2.0;
    double halfd  = d/2.0;

    double ans = d*halfn * logds2 - n*lgamma(halfd) +
      sumlog*(halfd-1) - s*halfd*sum;

    if(nd>0){
      g[0] = halfn*(logds2 + 1 - digamma(halfd)) + .5*(sumlog-s*sum);
      g[1] = halfn*d/s  - halfd*sum;

      if(nd>1){
	h(0,0) = halfn/d - trigamma(halfd)*n/4;
	h(0,1) = h(1,0) = halfn/s - .5*sum;
	h(1,1) = -halfn*d/pow(s,2);
      }
    }
    return ans;
  }

}

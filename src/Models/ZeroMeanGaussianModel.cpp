/*
  Copyright (C) 2008-2011 Steven L. Scott

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

#include <Models/ZeroMeanGaussianModel.hpp>
#include <Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp>
#include <Models/GammaModel.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM{

  typedef ZeroMeanGaussianModel ZGM;

  ZGM::ZeroMeanGaussianModel(double sigma)
      : ParamPolicy(new UnivParams(sigma * sigma))
  {}

  ZGM::ZeroMeanGaussianModel(const std::vector<double> &y)
      : GaussianModelBase(y),
        ParamPolicy(new UnivParams(1.0))
  {
    mle();
  }

  ZGM * ZGM::clone()const{return new ZGM(*this);}

  void ZGM::set_sigsq(double s2){
    Sigsq_prm()->set(s2);}

  Ptr<UnivParams> ZGM::Sigsq_prm(){
    return ParamPolicy::prm();}

  const Ptr<UnivParams> ZGM::Sigsq_prm()const{
    return ParamPolicy::prm();}

  double ZGM::sigsq()const{return Sigsq_prm()->value();}
  double ZGM::sigma()const{return sqrt(sigsq());}

  void ZGM::mle(){
    double n = suf()->n();
    double ss = suf()->sumsq();
    if(n>0) set_sigsq(ss/n);
    else set_sigsq(1.0);
  }

  void ZGM::set_conjugate_prior(double df, double sigma_guess){
    double ss = pow(sigma_guess, 2) * df;
    NEW(ZeroMeanGaussianConjSampler, pri)(this, df/2, ss/2);
    set_conjugate_prior(pri);
  }

  void ZGM::set_conjugate_prior(Ptr<GammaModelBase> ivar){
    NEW(ZeroMeanGaussianConjSampler, pri)(this, ivar);
    set_conjugate_prior(pri);
  }

  void ZGM::set_conjugate_prior(Ptr<ZeroMeanGaussianConjSampler> pri){
    ConjPriorPolicy::set_conjugate_prior(pri);
  }

  double ZGM::Loglike(Vec &g, Mat &h, uint nd)const{
    double sigsq = this->sigsq();
    if(sigsq<0) return BOOM::negative_infinity();

    const double log2pi = 1.8378770664093453;
    double n = suf()->n();
    double sumsq = suf()->sumsq();
    double SS = sumsq;
    double ans = -0.5*(n*(log2pi + log(sigsq)) + SS/sigsq);

    if(nd>0){
      double sigsq_sq = sigsq*sigsq;
      g[0] = -0.5*n/sigsq + 0.5*SS/sigsq_sq;
      if(nd>1) h(0,0) = (n/2 - SS/sigsq)/sigsq_sq;
    }
    return ans;
  }
} // namespace BOOM

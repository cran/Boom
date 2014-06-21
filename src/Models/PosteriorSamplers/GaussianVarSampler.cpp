/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#include <Models/PosteriorSamplers/GaussianVarSampler.hpp>
#include <cpputil/math_utils.hpp>
#include <cpputil/report_error.hpp>
#include <distributions.hpp>
#include <distributions/trun_gamma.hpp>
#include <Models/GaussianModelBase.hpp>
#include <Models/GammaModel.hpp>

namespace BOOM{

  typedef GaussianVarSampler GVS;
  GVS::GaussianVarSampler(GaussianModelBase * m, Ptr<GammaModelBase> g)
    : gam(g),
      mod(m),
      upper_truncation_point_(BOOM::infinity())
  {}

  inline double sumsq(double nu, double sig){ return nu*sig*sig;}

  GVS::GaussianVarSampler(GaussianModelBase* m,
                          double prior_df,
                          double prior_sigma_guess)
    : gam(new GammaModel(prior_df/2.0, sumsq(prior_df,prior_sigma_guess)/2.0)),
      mod(m),
      upper_truncation_point_(BOOM::infinity())
  {}

  void GVS::set_sigma_upper_limit(double max_sigma){
    if(max_sigma <= 0) {
      ostringstream err;
      err << "GaussianVarSampler::set_sigma_upper_limit expects a "
          << "positive argument.  It was given " << max_sigma;
      report_error(err.str());
    }
    upper_truncation_point_ = max_sigma;
  }

  void GVS::draw(){
    double n = mod->suf()->n();
    double ybar = mod->suf()->ybar();
    double mu = mod->mu();

    double sumsq = mod->suf()->sumsq() - 2*n*ybar*mu + n*mu*mu;

    double df = n + 2*gam->alpha();  // alpha = df/2
    double ss = sumsq + 2*gam->beta();

    double ans;
    if(upper_truncation_point_ == BOOM::infinity()){
      ans = rgamma_mt(rng(), df/2,ss/2);
    }else{
      ans = rtrun_gamma_mt(rng(), df/2, ss/2,
                           1.0/pow(upper_truncation_point_, 2));
    }

    mod->set_sigsq(1.0/ans);
  }

  double GVS::logpri()const{
    return gam->logp(1.0/mod->sigsq());
  }

  const Ptr<GammaModelBase> GVS::ivar()const{ return gam;}
}

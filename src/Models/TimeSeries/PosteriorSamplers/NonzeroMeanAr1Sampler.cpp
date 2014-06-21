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
#include <Models/TimeSeries/PosteriorSamplers/NonzeroMeanAr1Sampler.hpp>
#include <distributions.hpp>
#include <distributions/trun_gamma.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM{

  NonzeroMeanAr1Sampler::NonzeroMeanAr1Sampler(NonzeroMeanAr1Model *model,
                                               Ptr<GaussianModelBase> mean_prior,
                                               Ptr<GaussianModelBase> phi_prior,
                                               Ptr<GammaModelBase> siginv_prior)
      : m_(model),
        mean_prior_(mean_prior),
        phi_prior_(phi_prior),
        siginv_prior_(siginv_prior),
        truncate_phi_(false),
        force_ar1_positive_(false),
        sigma_upper_limit_(-1)
  {}
  //----------------------------------------------------------------------
  void NonzeroMeanAr1Sampler::force_stationary(){truncate_phi_ = true;}
  //----------------------------------------------------------------------
  void NonzeroMeanAr1Sampler::force_ar1_positive(){force_ar1_positive_ = true;}
  //----------------------------------------------------------------------
  void NonzeroMeanAr1Sampler::set_sigma_upper_limit(double sigma_upper_limit){
    sigma_upper_limit_ = sigma_upper_limit;
  }
  //----------------------------------------------------------------------
  void NonzeroMeanAr1Sampler::draw(){
    draw_mu();
    draw_phi();
    draw_sigma();
  }
  //----------------------------------------------------------------------
  double NonzeroMeanAr1Sampler::logpri()const{
    double ans = mean_prior_->logp(m_->mu());
    ans += phi_prior_->logp(m_->phi());
    ans += siginv_prior_->logp(1.0 / m_->sigsq());
    return ans;
  }
  //----------------------------------------------------------------------
  void NonzeroMeanAr1Sampler::draw_mu(){
    double phi = m_->phi();
    double sigsq = m_->sigsq();

    Ptr<Ar1Suf> suf = m_->suf();
    double n = suf->n();
    double current_sum = suf->sum_excluding_first();
    double lag_sum = suf->lag_sum();
    double y1 = suf->first_value();

    double ivar = (1 + (n-1)*pow(1-phi, 2)) / sigsq;
    ivar += 1.0 / mean_prior_->sigsq();

    double mean =  (1-phi)*(current_sum - phi * lag_sum) + y1;
    mean /= sigsq;
    mean += mean_prior_->mu() / mean_prior_->sigsq();
    mean /= ivar;

    double sd = sqrt(1.0/ivar);
    double mu = rnorm_mt(rng(), mean, sd);
    m_->set_mu(mu);
  }
  //----------------------------------------------------------------------
  void NonzeroMeanAr1Sampler::draw_phi(){
    Ptr<Ar1Suf> suf = m_->suf();
    double mu = m_->mu();
    double sigsq = m_->sigsq();
    double ivar = suf->centered_lag_sumsq(mu);  // sum((y[t-1] - mean)^2)
    ivar /= sigsq;
    ivar += 1.0 / phi_prior_->sigsq();

    double mean = suf->centered_cross(mu);
    mean /= sigsq;
    mean += phi_prior_->mu() / phi_prior_->sigsq();

    mean /= ivar;
    double sd = sqrt(1.0/ivar);

    double phi = 0;

    if(truncate_phi_){
      double lower_limit = force_ar1_positive_ ? 0 : -1;
      try {
        phi = rtrun_norm_2_mt(rng(), mean, sd, lower_limit, 1);
      } catch(std::exception &e) {
        ostringstream err;
        err << "NonzeroMeanAr1Sampler::draw_phi() caught an exception "
            << "when called with the following other parameters and "
            << "sufficient statistics:"
            << "mu = " << mu << endl
            << "sigma = " << sqrt(sigsq) << endl
            << "sufficient statistics:" << endl
            << *suf << endl
            << "The error message of the captured exception is " << endl
            << e.what() << endl
            ;
        report_error(err.str());
      } catch(...) {
        report_error("Unknown error caught in NonzeroMeanAr1Sampler::draw_phi().");
      }
    }else if(force_ar1_positive_){
      try {
        phi = rtrun_norm_mt(rng(), mean, sd, 0, true);
      } catch(std::exception &e) {
        ostringstream err;
        err << "NonzeroMeanAr1Sampler::draw_phi() caught an exception "
            << "when called with the following other parameters and "
            << "sufficient statistics:"
            << "mu = " << mu << endl
            << "sigma = " << sqrt(sigsq) << endl
            << "sufficient statistics:" << endl
            << *suf << endl
            << "The error message of the captured exception is " << endl
            << e.what() << endl
            ;
        report_error(err.str());
      } catch(...) {
        report_error("Unknown error caught in NonzeroMeanAr1Sampler::draw_phi().");
      }
    }else{
      phi = rnorm_mt(rng(), mean, sd);
    }
    m_->set_phi(phi);
  }
  //----------------------------------------------------------------------
  void NonzeroMeanAr1Sampler::draw_sigma(){
    double mu = m_->mu();
    double phi = m_->phi();
    Ptr<Ar1Suf> suf = m_->suf();
    double n = suf->n();

    // start with the prior
    double sumsq = 2 * siginv_prior_->beta() + suf->model_sumsq(mu, phi);
    double df = n + 2 * siginv_prior_->alpha();

    double ivar;
    if(sigma_upper_limit_ <= 0) {
      ivar = rgamma_mt(rng(), df/2, sumsq/2);
    }else{
      double lo = 1.0 / pow(sigma_upper_limit_, 2);
      ivar = rtrun_gamma_mt(rng(), df/2, sumsq/2, lo);
    }
    m_->set_sigsq(1.0/ivar);
  }

}

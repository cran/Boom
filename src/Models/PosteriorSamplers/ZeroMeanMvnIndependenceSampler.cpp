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

#include <Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp>
#include <distributions.hpp>
#include <distributions/trun_gamma.hpp>
#include <cpputil/math_utils.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM{
  typedef ZeroMeanMvnIndependenceSampler ZMMI;

  ZMMI::ZeroMeanMvnIndependenceSampler(ZeroMeanMvnModel *model,
                                       Ptr<GammaModelBase> prior,
                                       int which_variable)
      : m_(model),
        prior_(prior),
        which_variable_(which_variable),
        upper_truncation_point_(infinity())
  {}

  ZMMI::ZeroMeanMvnIndependenceSampler(ZeroMeanMvnModel *model,
                                       double prior_df,
                                       double prior_sigma_guess,
                                       int which_variable)
      : m_(model),
        prior_(new GammaModel(prior_df/2,
                              pow(prior_sigma_guess, 2) * prior_df / 2)),
        which_variable_(which_variable),
        upper_truncation_point_(infinity())
  {}

  void ZMMI::set_sigma_upper_limit(double max_sigma){
    if(max_sigma <= 0) {
      ostringstream err;
      err << "ZeroMeanMvnIndependenceSampler::set_sigma_upper_limit "
          << "expects a positive argument, given " << max_sigma;
      report_error(err.str());
    }
    upper_truncation_point_ = max_sigma;
  }

  void ZMMI::draw(){
    Spd siginv = m_->siginv();
    int i = which_variable_;
    double df = 2 * prior_->alpha() + m_->suf()->n();
    Spd sumsq = m_->suf()->center_sumsq(m_->mu());
    double ss = 2 * prior_->beta() + sumsq(i,i);
    if(upper_truncation_point_ == infinity()){
      siginv(i, i) = rgamma_mt(rng(), df/2, ss/2);
    }else{
      double cutpoint = 1.0/pow(upper_truncation_point_, 2);
      siginv(i, i) = rtrun_gamma_mt(rng(), df/2, ss/2, cutpoint);
    }
    m_->set_siginv(siginv);
  }

  double ZMMI::logpri()const{
    int i = which_variable_;
    double siginv = m_->siginv()(i, i);
    return prior_->logp(siginv);
  }

  //======================================================================
  typedef ZeroMeanMvnCompositeIndependenceSampler ZMMCIS;
  ZMMCIS::ZeroMeanMvnCompositeIndependenceSampler(
      ZeroMeanMvnModel *model,
      const std::vector<Ptr<GammaModelBase> > & siginv_priors,
      const Vec & sigma_upper_truncation_points)
      : model_(model),
        priors_(siginv_priors),
        sigma_upper_truncation_point_(sigma_upper_truncation_points)
  {
    if (model_->dim() != priors_.size()) {
      report_error("'model' and 'siginv_priors' arguments are not compatible "
                   "in "
                   "ZeroMeanMvnCompositeIndependenceSampler constructor.");
    }

    if (model_->dim() != sigma_upper_truncation_points.size()) {
      report_error("'model' and 'sigma_upper_truncation_points' arguments "
                   "are not compatible in "
                   "ZeroMeanMvnCompositeIndependenceSampler constructor.");
    }

    for (int i = 0; i < sigma_upper_truncation_points.size(); ++i) {
      if (sigma_upper_truncation_points[i] < 0) {
        ostringstream err;
        err << "Element " << i << " (counting from 0) of "
            << "sigma_upper_truncation_points is negative in "
            << "ZeroMeanMvnCompositeIndependenceSampler constructor."
            << endl
            << sigma_upper_truncation_points << endl;
        report_error(err.str());
      }
    }
  }

  void ZMMCIS::draw() {
    Spd Sigma = model_->Sigma();
    Spd sumsq = model_->suf()->center_sumsq(model_->mu());
    for (int i = 0; i < model_->dim(); ++i) {
      double df = 2 * priors_[i]->alpha() + model_->suf()->n();
      double ss = 2 * priors_[i]->beta() + sumsq(i,i);
      if (sigma_upper_truncation_point_[i] == 0) {
        Sigma(i, i) = 0.0;
      } if(sigma_upper_truncation_point_[i] == infinity()){
        Sigma(i, i) = 1.0 / rgamma_mt(rng(), df/2, ss/2);
      } else {
        double cutpoint = 1.0/pow(sigma_upper_truncation_point_[i], 2);
        Sigma(i, i) = 1.0 / rtrun_gamma_mt(rng(), df/2, ss/2, cutpoint);
      }
    }
    model_->set_Sigma(Sigma);
  }

  double ZMMCIS::logpri()const {
    const Spd & Sigma(model_->Sigma());
    double ans = 0;
    for (int i = 0; i < Sigma.nrow(); ++i) {
      if (sigma_upper_truncation_point_[i] > 0) {
        ans += priors_[i]->logp(1.0/Sigma(i, i));
      }
    }
    return ans;
  }

}  // namespace BOOM

/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include <Models/PosteriorSamplers/MvnIndependentVarianceSampler.hpp>

#include <distributions.hpp>
#include <distributions/trun_gamma.hpp>

namespace BOOM {

  MvnIndependentVarianceSampler::MvnIndependentVarianceSampler(
      MvnModel * model,
      const std::vector<Ptr<GammaModelBase> > & siginv_priors,
      const Vec & upper_sigma_truncation_point)
      : model_(model),
        priors_(siginv_priors),
        upper_sigma_truncation_point_(upper_sigma_truncation_point)
  {
    if (model->dim() != siginv_priors.size()) {
      report_error("The model and siginv_priors arguments do not conform in "
                   "the MvnIndependentVarianceSampler constructor.");
    }

    if (model->dim() != upper_sigma_truncation_point.size()) {
      report_error("The model and upper_sigma_truncation_point arguments do "
                   "not conform in the MvnIndependentVarianceSampler "
                   "constructor.");
    }

    for (int i  = 0; i < model->dim(); ++i) {
      if (upper_sigma_truncation_point_[i] < 0) {
        report_error("All elements of upper_sigma_truncation_point must be "
                     "non-negative in "
                     "MvnIndependentVarianceSampler constructor.");
      }
    }
  }

  void MvnIndependentVarianceSampler::draw(){
    Spd siginv = model_->siginv();
    Spd sumsq = model_->suf()->center_sumsq(model_->mu());
    for (int i = 0; i < model_->dim(); ++i) {
      double df = 2 * priors_[i]->alpha() + model_->suf()->n();
      double ss = 2 * priors_[i]->beta() + sumsq(i, i);
      if (upper_sigma_truncation_point_[i] == 0.0) {
        siginv(i, i) = 0.0;
      } else if(upper_sigma_truncation_point_[i] == infinity()) {
        siginv(i, i) = rgamma_mt(rng(), df/2, ss/2);
      } else {
        double cutpoint = 1.0/pow(upper_sigma_truncation_point_[i], 2);
        siginv(i, i) = rtrun_gamma_mt(rng(), df/2, ss/2, cutpoint);
      }
    }
    model_->set_siginv(siginv);
  }

  double MvnIndependentVarianceSampler::logpri()const{
    double ans = 0;
    for (int i = 0; i < priors_.size(); ++i) {
      double siginv = model_->siginv()(i, i);
      ans += priors_[i]->logp(siginv);
    }
    return ans;
  }


} // namespace BOOM

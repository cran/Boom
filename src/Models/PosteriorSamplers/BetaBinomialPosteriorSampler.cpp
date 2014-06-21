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

#include <Models/PosteriorSamplers/BetaBinomialPosteriorSampler.hpp>
#include <boost/bind.hpp>
#include <distributions.hpp>
#include <stats/logit.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM{

  typedef BetaBinomialPosteriorSampler BBPS;

  BBPS::BetaBinomialPosteriorSampler(
      BetaBinomialModel *model,
      Ptr<BetaModel> probability_prior_distribution,
      Ptr<DoubleModel> sample_size_prior_distribution)
      : model_(model),
        probability_prior_distribution_(probability_prior_distribution),
        sample_size_prior_distribution_(sample_size_prior_distribution),
        probability_sampler_(boost::bind(
            &BetaBinomialPosteriorSampler::logp_prob, this, _1)),
        sample_size_sampler_(boost::bind(
            &BetaBinomialPosteriorSampler::logp_sample_size, this, _1)),
        sampling_method_(DATA_AUGMENTATION)
  {
    probability_sampler_.set_limits(0,1);
    sample_size_sampler_.set_lower_limit(0);
  }

  double BBPS::logpri()const{
    double prob = model_->prior_mean();
    double sample_size = model_->prior_sample_size();
    return probability_prior_distribution_->logp(prob) +
        sample_size_prior_distribution_->logp(sample_size);
  }

  void BBPS::draw(){
    switch (sampling_method_){
      case SLICE:
        draw_slice();
        return;

      case DATA_AUGMENTATION:
        draw_data_augmentation();
        return;

      default:
        draw_data_augmentation();
        return;
    }
  }

  void BBPS::draw_data_augmentation(){
    double a = model_->a();
    double b = model_->b();
    complete_data_suf_.clear();

    const std::vector<Ptr<BinomialData> > & data(model_->dat());
    int nobs = data.size();
    for (int i = 0; i < nobs; ++i) {
      int y = data[i]->y();
      int n = data[i]->n();
      double theta;
      int failure_count = 0;
      do {
        // In obscure corner cases where either a or b is very close
        // to zero you can get theta == 0.0 or theta == 1.0
        // numerically.  In that case just keep trying.  If it takes
        // more than 100 tries then something is really wrong.
        theta = rbeta_mt(rng(), y + a, n - y + b);
        if (++failure_count > 100) {
          report_error(
              "Too many attempts at rbeta in "
              "BetaBinomialPosteriorSampler::draw_data_augmentation");
        }
      } while (theta == 0.0 || theta == 1.0 || !std::isfinite(theta));
      complete_data_suf_.update_raw(theta);
    }
    draw_slice();
  }

  void BBPS::draw_slice(){
    double prob = model_->prior_mean();
    prob = probability_sampler_.draw(prob);
    model_->set_prior_mean(prob);

    double sample_size = model_->prior_sample_size();
    sample_size  = sample_size_sampler_.draw(sample_size);
    model_->set_prior_sample_size(sample_size);
  }

  double BBPS::logp(double prob, double sample_size)const{
    double a = prob * sample_size;
    double b = sample_size - a;
    double ans = probability_prior_distribution_->logp(prob)
        + sample_size_prior_distribution_->logp(sample_size);
    if (sampling_method_ == DATA_AUGMENTATION) {
      ans += beta_log_likelihood(a, b, complete_data_suf_);
    } else {
      ans += model_->loglike(a, b);
    }
    return ans;
  }

  double BBPS::logp_sample_size(double sample_size)const{
    double prob = model_->prior_mean();
    return logp(prob, sample_size);
  }

  double BBPS::logp_prob(double prob)const{
    double sample_size = model_->prior_sample_size();
    return logp(prob, sample_size);
  }

}  // namespace BOOM
